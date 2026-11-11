export type ToolCall = {
  id: string
  name: string
  arguments: string
  result?: unknown
  running?: boolean
}

/** One model call. A turn with tool use has several: calls, then the answer. */
export type Round = {
  /** Streamed `reasoning_content` (the model's thinking, shown before the answer). */
  reasoning?: string
  /** How long the thinking phase took, for "Thought for 1.2s". */
  thinkingMs?: number
  content: string
  toolCalls: ToolCall[]
  finish?: string | null
}

export type Meta = {
  ttft?: number
  firstAt?: number
  end?: number
  prompt?: number
  completion?: number
  reasoning?: number
  finish?: string | null
  stopped?: boolean
  interrupted?: boolean
  imageGen?: boolean
}

export type UserMessage = {
  id: string
  role: "user"
  content: string
  images: string[]
}

export type GeneratedImage = { b64?: string; size?: string; pending?: boolean }

export type AssistantMessage = {
  id: string
  role: "assistant"
  model: string
  rounds: Round[]
  image?: GeneratedImage | null
  caption?: string
  meta: Meta
  error?: { message: string; status?: number } | null
  done: boolean
}

export type Message = UserMessage | AssistantMessage

export type Chat = {
  id: string
  title: string
  created: number
  updated: number
  messages: Message[]
}

/** "default" omits reasoning_effort and lets the model decide (kiana-reasoning thinks). */
export type ReasoningEffort = "default" | "none" | "low" | "medium" | "high"
export type ToolChoice = "auto" | "required" | "none"

export type Settings = {
  model: string
  system: string
  temperature: number
  maxTokens: string
  seed: string
  reasoning: ReasoningEffort
  format: "text" | "json_object"
  stream: boolean
  tools: boolean
  toolChoice: ToolChoice
  image: boolean
  ttft: number
  tps: number
  apiKey: string
}

export type SendOptions = {
  tools?: boolean
  toolChoice?: ToolChoice
  image?: boolean
}
