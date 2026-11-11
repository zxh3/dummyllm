import { toast } from "sonner"
import { create } from "zustand"

import { API_BASE, ApiError, IMAGE_MODEL, apiHeaders, assistantToApi, buildMessages, readSSE, requestParams, toApiError } from "./api"
import { chatDB } from "./db"
import { runTool } from "./tools"
import type { AssistantMessage, Chat, Round, SendOptions, Settings, UserMessage } from "./types"

export const DEFAULT_SETTINGS: Settings = {
  model: "kiana",
  system: "",
  temperature: 1,
  maxTokens: "",
  seed: "",
  reasoning: "none",
  format: "text",
  stream: true,
  tools: false,
  toolChoice: "auto",
  image: false,
  ttft: 200,
  tps: 150,
  replyTokens: 0,
  apiKey: "",
}

const CAPTIONS = [
  "*proudly presents a drawing* Mrrp!",
  "*drops a picture at your feet* Meow~",
  "*paints with its paws* Prrr. Behold!",
  "*knocks over the paint* …Meow. Art.",
]

export const uid = () => Math.random().toString(36).slice(2, 10) + Date.now().toString(36)
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms))

// ---------------------------------------------------------------- persistence
//
// Chats live in IndexedDB (one record per chat, written only when that chat
// changes). Settings and the open chat id are tiny and needed synchronously at
// startup, so they stay in localStorage.

const PREFS_KEY = "dummyllm.prefs"
const LEGACY_KEY = "dummyllm.v2" // everything-in-localStorage format, migrated on first load

function loadPrefs(): { settings: Settings; currentId: string | null } {
  try {
    const stored = localStorage.getItem(PREFS_KEY)
    const raw = JSON.parse(stored ?? localStorage.getItem(LEGACY_KEY) ?? "{}")
    const saved = { ...raw.settings }
    // Older builds saved slower latency defaults (450 ms / 40 tok/s); move those to
    // the current defaults but keep values someone actually chose.
    if (saved.ttft === 450) delete saved.ttft
    if (saved.tps === 40) delete saved.tps
    if (saved.reasoning === "default") delete saved.reasoning
    // Older builds had several models (meow-*, kiana-*); there is now just `kiana`.
    // Someone who picked a reasoning model keeps thinking via the effort setting.
    if (/reason/.test(saved.model ?? "") && !saved.reasoning) saved.reasoning = "medium"
    if (/^(meow-|kiana-)/.test(saved.model ?? "")) saved.model = "kiana"
    const prefs = { settings: { ...DEFAULT_SETTINGS, ...saved }, currentId: raw.currentId ?? null }
    savePrefs(prefs.settings, prefs.currentId) // persist migrations (and carry legacy settings over)
    return prefs
  } catch {
    return { settings: DEFAULT_SETTINGS, currentId: null }
  }
}

function savePrefs(settings: Settings, currentId: string | null) {
  try {
    localStorage.setItem(PREFS_KEY, JSON.stringify({ settings, currentId }))
  } catch {
    /* storage disabled: settings just won't stick */
  }
}

/** Turns that were still streaming when the tab closed become "interrupted". Returns the chats it changed. */
function repairInterrupted(chats: Chat[]): Chat[] {
  const changed: Chat[] = []
  for (const c of chats) {
    let touched = false
    for (const m of c.messages)
      if (m.role === "assistant" && !m.done) {
        m.done = true
        m.meta = { ...m.meta, stopped: true, interrupted: true }
        if (m.image?.pending) m.image = null
        touched = true
      }
    if (touched) changed.push(c)
  }
  return changed
}

async function loadChats(): Promise<Chat[]> {
  const legacy = localStorage.getItem(LEGACY_KEY)
  if (legacy) {
    let chats: Chat[] = []
    try {
      const raw = JSON.parse(legacy)
      if (Array.isArray(raw.chats)) chats = raw.chats
    } catch {
      /* corrupt: nothing to migrate */
    }
    if (chats.length) await chatDB.put(chats)
    localStorage.removeItem(LEGACY_KEY)
  }
  const chats = await chatDB.all()
  const repaired = repairInterrupted(chats)
  if (repaired.length) await chatDB.put(repaired)
  return chats.sort((a, b) => b.updated - a.updated)
}

// ---------------------------------------------------------------- store

type State = {
  /** False until chats have been loaded from IndexedDB. */
  ready: boolean
  chats: Chat[]
  currentId: string | null
  settings: Settings
  attachments: string[]
  busy: boolean
  setSettings: (patch: Partial<Settings>) => void
  resetSettings: () => void
  newChat: () => void
  selectChat: (id: string) => void
  deleteChat: (id: string) => void
  addAttachments: (urls: string[]) => void
  removeAttachment: (index: number) => void
  send: (text: string, opts?: SendOptions) => Promise<void>
  stop: () => void
  regenerate: (assistantId: string) => void
}

let controller: AbortController | null = null

export const useChat = create<State>()((set, get) => {
  const prefs = loadPrefs()

  const updateChat = (chatId: string, fn: (c: Chat) => Chat) =>
    set((s) => ({ chats: s.chats.map((c) => (c.id === chatId ? fn(c) : c)) }))

  const snapshot = (m: AssistantMessage): AssistantMessage => ({
    ...m,
    meta: { ...m.meta },
    image: m.image ? { ...m.image } : m.image,
    rounds: m.rounds.map((r) => ({ ...r, toolCalls: r.toolCalls.map((c) => ({ ...c })) })),
  })

  async function respond(chatId: string, opts: SendOptions & { prompt: string }) {
    const settings = get().settings
    const draft: AssistantMessage = {
      id: uid(),
      role: "assistant",
      model: opts.image ? IMAGE_MODEL : settings.model,
      rounds: [],
      image: null,
      meta: {},
      done: false,
    }
    updateChat(chatId, (c) => ({ ...c, updated: Date.now(), messages: [...c.messages, snapshot(draft)] }))
    controller = new AbortController()
    set({ busy: true })

    // Stream into a mutable draft; publish at most once per animation frame.
    // Background/occluded tabs may not run animation frames at all, so a timer
    // races the frame; otherwise a backgrounded reply wouldn't reach the store
    // (or storage) until it finished.
    let cancel: (() => void) | null = null
    const write = () => updateChat(chatId, (c) => ({ ...c, messages: c.messages.map((m) => (m.id === draft.id ? snapshot(draft) : m)) }))
    const run = () => {
      cancel?.()
      cancel = null
      write()
    }
    const commit = (now = false) => {
      if (now) run()
      else if (!cancel) {
        const f = requestAnimationFrame(run)
        const t = setTimeout(run, 250)
        cancel = () => {
          cancelAnimationFrame(f)
          clearTimeout(t)
        }
      }
    }

    try {
      if (opts.image) await generateImage(draft, opts.prompt, settings, controller.signal, commit)
      else await complete(chatId, draft, opts, settings, controller.signal, commit)
    } catch (e) {
      if ((e as Error).name === "AbortError") draft.meta.stopped = true
      else draft.error = { message: (e as Error).message || String(e), status: (e as ApiError).status }
    } finally {
      draft.done = true
      for (const r of draft.rounds) for (const c of r.toolCalls) c.running = false
      commit(true)
      controller = null
      set({ busy: false })
    }
  }

  async function complete(
    chatId: string,
    draft: AssistantMessage,
    opts: SendOptions,
    settings: Settings,
    signal: AbortSignal,
    commit: (now?: boolean) => void,
  ) {
    const t0 = performance.now()
    let firstAt: number | undefined
    const usage = { prompt: 0, completion: 0, reasoning: 0 }
    let toolChoice = opts.toolChoice

    for (let round = 0; round < 6; round++) {
      const r: Round = { content: "", toolCalls: [], finish: null }
      let thinkStart: number | undefined
      draft.rounds.push(r)
      const chat = get().chats.find((c) => c.id === chatId)!
      const messages = buildMessages(chat, chat.messages.findIndex((m) => m.id === draft.id), settings)
      assistantToApi(draft, messages, draft.rounds.slice(0, -1))
      const body: Record<string, unknown> = { ...requestParams(settings, { ...opts, toolChoice }), messages }
      if (settings.stream) {
        body.stream = true
        body.stream_options = { include_usage: true }
      }
      const resp = await fetch(`${API_BASE}/chat/completions`, {
        method: "POST",
        signal,
        body: JSON.stringify(body),
        headers: apiHeaders(settings, {
          // Ask the server for a believable typing speed (API clients are unaffected).
          "x-dummyllm-ttft-ms": String(round ? Math.min(settings.ttft, 300) : settings.ttft),
          "x-dummyllm-tokens-per-second": String(settings.tps),
          ...(settings.replyTokens > 0 ? { "x-dummyllm-reply-tokens": String(settings.replyTokens) } : {}),
        }),
      })
      if (!resp.ok) throw await toApiError(resp)

      if (settings.stream) {
        for await (const chunk of readSSE(resp)) {
          if (chunk.usage) addUsage(usage, chunk.usage)
          const choice = chunk.choices?.[0]
          if (!choice) continue
          const delta = choice.delta ?? {}
          const now = performance.now()
          if (delta.reasoning_content) {
            r.reasoning = (r.reasoning ?? "") + delta.reasoning_content
            thinkStart ??= now
            firstAt ??= now
          }
          // The first answer token ends the thinking phase.
          if ((delta.content || delta.tool_calls) && thinkStart !== undefined && r.thinkingMs === undefined) r.thinkingMs = now - thinkStart
          if (delta.content) {
            r.content += delta.content
            firstAt ??= now
          }
          for (const tc of delta.tool_calls ?? []) {
            firstAt ??= now
            const call = (r.toolCalls[tc.index] ??= { id: "", name: "", arguments: "" })
            if (tc.id) call.id = tc.id
            if (tc.function?.name) call.name = tc.function.name
            if (tc.function?.arguments) call.arguments += tc.function.arguments
          }
          if (choice.finish_reason) r.finish = choice.finish_reason
          commit()
        }
      } else {
        const data = await resp.json()
        firstAt ??= performance.now()
        const choice = data.choices[0]
        r.content = choice.message.content ?? ""
        if (choice.message.reasoning_content) r.reasoning = choice.message.reasoning_content
        r.toolCalls = (choice.message.tool_calls ?? []).map((c: { id: string; function: { name: string; arguments: string } }) => ({
          id: c.id,
          name: c.function.name,
          arguments: c.function.arguments,
        }))
        r.finish = choice.finish_reason
        addUsage(usage, data.usage)
      }

      draft.meta = { ...draft.meta, ttft: firstAt && firstAt - t0, firstAt, end: performance.now(), ...usage, finish: r.finish }
      commit(true)
      if (r.finish !== "tool_calls" || !r.toolCalls.length) break

      // "Execute" the tools, then hand the results back to the model.
      for (const call of r.toolCalls) {
        call.running = true
        commit(true)
        await sleep(300 + Math.random() * 500)
        if (signal.aborted) throw new DOMException("aborted", "AbortError")
        call.result = runTool(call.name, call.arguments)
        call.running = false
        commit(true)
      }
      // Forcing a call every round would loop forever; let the model decide after the first.
      if (toolChoice === "required" || settings.toolChoice === "required") toolChoice = "auto"
    }
  }

  async function generateImage(
    draft: AssistantMessage,
    prompt: string,
    settings: Settings,
    signal: AbortSignal,
    commit: (now?: boolean) => void,
  ) {
    const t0 = performance.now()
    draft.image = { pending: true }
    commit(true)
    const resp = await fetch(`${API_BASE}/images/generations`, {
      method: "POST",
      signal,
      headers: apiHeaders(settings, { "x-dummyllm-ttft-ms": String(Math.max(1800, settings.ttft * 3)) }),
      body: JSON.stringify({ model: IMAGE_MODEL, prompt: prompt || "a cat", size: "1024x1024", response_format: "b64_json" }),
    })
    if (!resp.ok) throw await toApiError(resp)
    const data = await resp.json()
    draft.image = { b64: data.data[0].b64_json, size: data.size ?? "1024x1024" }
    draft.caption = CAPTIONS[Math.floor(Math.random() * CAPTIONS.length)]
    draft.meta = { ttft: performance.now() - t0, completion: data.usage?.output_tokens, imageGen: true }
  }

  return {
    ready: false,
    chats: [],
    currentId: prefs.currentId,
    settings: prefs.settings,
    attachments: [],
    busy: false,

    setSettings: (patch) => set((s) => ({ settings: { ...s.settings, ...patch } })),
    resetSettings: () => set((s) => ({ settings: { ...DEFAULT_SETTINGS, model: s.settings.model } })),

    newChat: () => {
      if (get().busy) return
      set({ currentId: null, attachments: [] })
    },
    selectChat: (id) => {
      if (get().busy) return
      set({ currentId: id })
    },
    deleteChat: (id) => {
      const { busy, currentId, chats } = get()
      if (busy && id === currentId) return
      set({ chats: chats.filter((c) => c.id !== id), currentId: currentId === id ? null : currentId })
    },

    addAttachments: (urls) => set((s) => ({ attachments: [...s.attachments, ...urls].slice(0, 4) })),
    removeAttachment: (index) => set((s) => ({ attachments: s.attachments.filter((_, i) => i !== index) })),

    send: async (text, opts = {}) => {
      const { busy, attachments, settings } = get()
      text = text.trim()
      if (busy || (!text && !attachments.length)) return
      let chatId = get().currentId
      const user: UserMessage = { id: uid(), role: "user", content: text, images: attachments }
      if (!chatId || !get().chats.some((c) => c.id === chatId)) {
        const now = Date.now()
        const chat: Chat = { id: uid(), title: (text || "Image").replace(/\s+/g, " ").slice(0, 60), created: now, updated: now, messages: [user] }
        chatId = chat.id
        set((s) => ({ chats: [chat, ...s.chats], currentId: chat.id, attachments: [] }))
      } else {
        const id = chatId
        set((s) => {
          const chat = s.chats.find((c) => c.id === id)!
          const updated = { ...chat, updated: Date.now(), messages: [...chat.messages, user] }
          return { chats: [updated, ...s.chats.filter((c) => c.id !== id)], attachments: [] }
        })
      }
      await respond(chatId, { ...opts, image: opts.image ?? settings.image, prompt: text })
    },

    stop: () => controller?.abort(),

    regenerate: (assistantId) => {
      const { busy, currentId, chats } = get()
      const chat = chats.find((c) => c.id === currentId)
      if (busy || !chat) return
      const idx = chat.messages.findIndex((m) => m.id === assistantId)
      const prev = chat.messages[idx - 1]
      const old = chat.messages[idx] as AssistantMessage
      updateChat(chat.id, (c) => ({ ...c, messages: c.messages.slice(0, idx) }))
      void respond(chat.id, { image: !!old.image, prompt: prev?.role === "user" ? prev.content : "" })
    },
  }
})

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function addUsage(acc: { prompt: number; completion: number; reasoning: number }, u?: Record<string, any>) {
  if (!u) return
  acc.prompt += u.prompt_tokens ?? 0
  acc.completion += u.completion_tokens ?? 0
  acc.reasoning += u.completion_tokens_details?.reasoning_tokens ?? 0
}

// ---------------------------------------------------------------- load + save

loadChats()
  .then((chats) => {
    const { currentId } = useChat.getState()
    useChat.setState({ chats, ready: true, currentId: chats.some((c) => c.id === currentId) ? currentId : null })
  })
  .catch((err) => {
    console.error("dummyllm: couldn't open IndexedDB", err)
    toast.error("Chat history isn't available in this browser, so new chats won't be saved.")
    useChat.setState({ ready: true })
  })

// Only chats whose object changed are written. Streaming replaces the active
// chat every frame, so writes are throttled: at most one per second while a
// reply streams (a debounce would starve until the stream ends).
const dirty = new Set<string>()
let flushTimer: ReturnType<typeof setTimeout> | undefined
let warnedQuota = false
let askedPersist = false

function flush() {
  clearTimeout(flushTimer)
  flushTimer = undefined
  if (!dirty.size) return
  const batch = useChat.getState().chats.filter((c) => dirty.has(c.id))
  dirty.clear()
  chatDB
    .put(batch)
    .then(() => {
      // Ask the browser not to evict our data under storage pressure (once).
      if (!askedPersist) {
        askedPersist = true
        void navigator.storage?.persist?.()
      }
    })
    .catch((err: DOMException) => {
      console.error("dummyllm: saving chats failed", err)
      if (err?.name === "QuotaExceededError" && !warnedQuota) {
        warnedQuota = true
        toast.error("Browser storage is full. Delete some chats to keep saving new ones.")
      }
    })
}

useChat.subscribe((s, prev) => {
  if (s.settings !== prev.settings || s.currentId !== prev.currentId) savePrefs(s.settings, s.currentId)
  if (!s.ready || !prev.ready || s.chats === prev.chats) return

  const before = new Map(prev.chats.map((c) => [c.id, c]))
  for (const c of s.chats) if (before.get(c.id) !== c) dirty.add(c.id)
  const alive = new Set(s.chats.map((c) => c.id))
  const removed = prev.chats.filter((c) => !alive.has(c.id)).map((c) => c.id)
  if (removed.length) {
    for (const id of removed) dirty.delete(id)
    chatDB.delete(removed).catch((err) => console.error("dummyllm: deleting chats failed", err))
  }
  flushTimer ??= setTimeout(flush, s.busy ? 1000 : 250)
})

// Write pending changes when the tab is hidden or closed.
document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "hidden") flush()
})
window.addEventListener("pagehide", flush)

export function currentChat(s: Pick<State, "chats" | "currentId">): Chat | null {
  return s.chats.find((c) => c.id === s.currentId) ?? null
}
