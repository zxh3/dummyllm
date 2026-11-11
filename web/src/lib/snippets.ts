import { API_BASE, buildMessages, requestParams } from "./api"
import type { Chat, Settings } from "./types"

export type SnippetLang = "python" | "node" | "curl"

type Json = null | boolean | number | string | Json[] | { [k: string]: Json }

/** Render a value on one line in the given language's literal syntax. */
function inline(v: Json, lang: SnippetLang): string {
  if (v === null || v === undefined) return lang === "python" ? "None" : "null"
  if (typeof v === "boolean") return lang === "python" ? (v ? "True" : "False") : String(v)
  if (typeof v === "number" || typeof v === "string") return JSON.stringify(v)
  if (Array.isArray(v)) return `[${v.map((x) => inline(x, lang)).join(", ")}]`
  const key = (k: string) => (lang === "node" && /^[A-Za-z_$][\w$]*$/.test(k) ? k : JSON.stringify(k))
  const body = Object.entries(v)
    .map(([k, x]) => `${key(k)}: ${inline(x, lang)}`)
    .join(", ")
  return lang === "node" ? `{ ${body} }` : `{${body}}`
}

/** Arrays (messages, tools) get one element per line; everything else stays inline. */
function block(v: Json, lang: SnippetLang, indent: string, step: string): string {
  if (!Array.isArray(v) || v.length === 0) return inline(v, lang)
  const comma = lang === "curl" ? "" : ","
  return `[\n${v.map((x, i) => `${indent}${step}${inline(x, lang)}${i < v.length - 1 ? "," : comma}`).join("\n")}\n${indent}]`
}

/** Turn the current conversation + settings into short, runnable client code. */
export function snippet(lang: SnippetLang, chat: Chat | null, settings: Settings): string {
  const messages = (
    chat?.messages.length ? buildMessages(chat, chat.messages.length, settings, true) : [{ role: "user", content: "Hello!" }]
  ) as Json
  const { model, ...params } = requestParams(settings) as Record<string, Json>
  const key = settings.apiKey ? "YOUR_API_KEY" : "anything"
  const args: [string, Json][] = [["model", model], ["messages", messages], ...Object.entries(params)]
  // With tools the reply may be a tool call (no text), so show the whole message.
  const out = params.tools ? "message" : "message.content"

  if (lang === "python") {
    return [
      "from openai import OpenAI",
      "",
      "client = OpenAI(",
      `    base_url="${API_BASE}",`,
      `    api_key="${key}",`,
      ")",
      "",
      "response = client.chat.completions.create(",
      ...args.map(([k, v]) => `    ${k}=${block(v, lang, "    ", "    ")},`),
      ")",
      `print(response.choices[0].${out})`,
    ].join("\n")
  }

  if (lang === "node") {
    return [
      'import OpenAI from "openai";',
      "",
      "const client = new OpenAI({",
      `  baseURL: "${API_BASE}",`,
      `  apiKey: "${key}",`,
      "});",
      "",
      "const response = await client.chat.completions.create({",
      ...args.map(([k, v]) => `  ${k}: ${block(v, lang, "  ", "  ")},`),
      "});",
      `console.log(response.choices[0].${out});`,
    ].join("\n")
  }

  const body = args.map(([k, v]) => `    ${JSON.stringify(k)}: ${block(v, "curl", "    ", "  ")}`).join(",\n")
  return [
    `curl ${API_BASE}/chat/completions \\`,
    '  -H "Content-Type: application/json" \\',
    ...(settings.apiKey ? ['  -H "Authorization: Bearer YOUR_API_KEY" \\'] : []),
    `  -d '{\n${body.replace(/'/g, "'\\''")}\n  }'`,
  ].join("\n")
}
