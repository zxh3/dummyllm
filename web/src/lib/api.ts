import { TOOLS } from "./tools"
import type { AssistantMessage, Chat, Round, SendOptions, Settings, UserMessage } from "./types"

export const API_BASE = `${window.location.origin}/v1`
export const IMAGE_MODEL = "meow-image-1"

export class ApiError extends Error {
  status?: number
  constructor(message: string, status?: number) {
    super(message)
    this.name = "ApiError"
    this.status = status
  }
}

export function apiHeaders(settings: Settings, extra: Record<string, string> = {}): HeadersInit {
  const h: Record<string, string> = { "content-type": "application/json", ...extra }
  if (settings.apiKey) h.authorization = `Bearer ${settings.apiKey}`
  return h
}

export async function toApiError(resp: Response): Promise<ApiError> {
  let message = resp.statusText || "Request failed"
  try {
    const body = await resp.json()
    message = body?.error?.message ?? message
  } catch {
    /* not JSON */
  }
  return new ApiError(message, resp.status)
}

/** Parse a `text/event-stream` response into JSON payloads (stops at `[DONE]`). */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export async function* readSSE(resp: Response): AsyncGenerator<any> {
  const reader = resp.body!.getReader()
  const decoder = new TextDecoder()
  let buffer = ""
  for (;;) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    let i: number
    while ((i = buffer.indexOf("\n\n")) >= 0) {
      const block = buffer.slice(0, i)
      buffer = buffer.slice(i + 2)
      const data = block
        .split("\n")
        .filter((l) => l.startsWith("data:"))
        .map((l) => l.slice(5).trimStart())
        .join("\n")
      if (!data) continue
      if (data === "[DONE]") return
      yield JSON.parse(data)
    }
  }
}

type ApiMessage = Record<string, unknown>

function userToApi(m: UserMessage, redactImages: boolean): ApiMessage {
  if (!m.images.length) return { role: "user", content: m.content }
  return {
    role: "user",
    content: [
      ...(m.content ? [{ type: "text", text: m.content }] : []),
      ...m.images.map((url) => ({
        type: "image_url",
        image_url: { url: redactImages ? "data:image/jpeg;base64,…" : url },
      })),
    ],
  }
}

/** Append an assistant turn (including tool calls + tool results) as API messages. */
export function assistantToApi(m: AssistantMessage, out: ApiMessage[], rounds: Round[] = m.rounds) {
  if (m.image) {
    out.push({ role: "assistant", content: m.caption || "*presents a drawing*" })
    return
  }
  for (const r of rounds) {
    const calls = r.toolCalls
    if (calls.length && calls.every((c) => c.result !== undefined)) {
      out.push({
        role: "assistant",
        content: r.content || null,
        tool_calls: calls.map((c) => ({ id: c.id, type: "function", function: { name: c.name, arguments: c.arguments } })),
      })
      for (const c of calls) out.push({ role: "tool", tool_call_id: c.id, content: JSON.stringify(c.result) })
    } else if (r.content) {
      out.push({ role: "assistant", content: r.content })
    }
  }
}

export function buildMessages(chat: Chat, uptoIndex: number, settings: Settings, redactImages = false): ApiMessage[] {
  const out: ApiMessage[] = []
  if (settings.system.trim()) out.push({ role: "system", content: settings.system.trim() })
  for (const m of chat.messages.slice(0, uptoIndex)) {
    if (m.role === "user") out.push(userToApi(m, redactImages))
    else if (!m.error) assistantToApi(m, out)
  }
  return out
}

export function requestParams(settings: Settings, opts: SendOptions = {}): Record<string, unknown> {
  const p: Record<string, unknown> = { model: settings.model }
  if (settings.temperature !== 1) p.temperature = settings.temperature
  if (settings.maxTokens) p.max_completion_tokens = Number(settings.maxTokens)
  if (settings.seed !== "") p.seed = Number(settings.seed)
  if (settings.reasoning !== "none") p.reasoning_effort = settings.reasoning
  if (settings.format === "json_object") p.response_format = { type: "json_object" }
  if (opts.tools ?? settings.tools) {
    p.tools = TOOLS
    p.tool_choice = opts.toolChoice ?? settings.toolChoice
  }
  return p
}

export async function fetchModels(settings: Settings): Promise<string[]> {
  const resp = await fetch(`${API_BASE}/models`, { headers: apiHeaders(settings) })
  if (!resp.ok) throw await toApiError(resp)
  const body = await resp.json()
  return (body.data as { id: string }[]).map((m) => m.id).filter((id) => !/embed|image/.test(id))
}
