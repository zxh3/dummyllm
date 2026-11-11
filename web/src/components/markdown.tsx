import { CheckIcon, CopyIcon } from "lucide-react"
import { Fragment, useState, type ReactNode } from "react"

import { Button } from "@/components/ui/button"
import { copyText } from "@/lib/browser"

/**
 * A deliberately small Markdown renderer: paragraphs, ``` code fences, `inline code`,
 * **bold**, and *cat actions* (rendered as muted italics). JSON replies become code blocks.
 */
export function Markdown({ text, streaming }: { text: string; streaming?: boolean }) {
  const caret = streaming ? <Caret /> : null
  const json = prettyJson(text)
  if (json) return <CodeBlock code={json} lang="json" />

  type Block = { kind: "p"; text: string } | { kind: "code"; code: string; lang: string }
  const blocks: Block[] = []
  text.split("```").forEach((part, i) => {
    if (i % 2) {
      const nl = part.indexOf("\n")
      blocks.push({
        kind: "code",
        lang: nl >= 0 ? part.slice(0, nl).trim() : "",
        code: nl >= 0 ? part.slice(nl + 1).replace(/\n$/, "") : part,
      })
    } else {
      for (const p of part.split(/\n{2,}/)) if (p.trim()) blocks.push({ kind: "p", text: p.trim() })
    }
  })
  const lastIsParagraph = blocks.at(-1)?.kind === "p"
  return (
    <div className="space-y-2.5 text-[15px] leading-[1.65] break-words">
      {blocks.map((b, i) =>
        b.kind === "code" ? (
          <CodeBlock key={i} code={b.code} lang={b.lang} />
        ) : (
          <p key={i}>
            {inline(b.text)}
            {i === blocks.length - 1 && caret}
          </p>
        ),
      )}
      {caret && !lastIsParagraph && <p>{caret}</p>}
    </div>
  )
}

/** JSON replies (e.g. response_format=json_object) render as a code block. */
function prettyJson(text: string): string | null {
  const trimmed = text.trim()
  if (!/^[[{]/.test(trimmed) || trimmed.length < 2) return null
  try {
    return JSON.stringify(JSON.parse(trimmed), null, 2)
  } catch {
    return null // partial JSON while streaming
  }
}

function inline(text: string): ReactNode[] {
  const out: ReactNode[] = []
  const re = /(`[^`\n]+`|\*\*[^*\n]+\*\*|\*[^*\n]+\*|\n)/g
  let last = 0
  let m: RegExpExecArray | null
  let k = 0
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(<Fragment key={k++}>{text.slice(last, m.index)}</Fragment>)
    const tok = m[0]
    if (tok === "\n") out.push(<br key={k++} />)
    else if (tok.startsWith("`"))
      out.push(
        <code key={k++} className="rounded-md bg-muted px-1.5 py-0.5 font-mono text-[0.85em]">
          {tok.slice(1, -1)}
        </code>,
      )
    else if (tok.startsWith("**"))
      out.push(
        <strong key={k++} className="font-semibold">
          {tok.slice(2, -2)}
        </strong>,
      )
    else
      out.push(
        <em key={k++} className="text-muted-foreground">
          {tok.slice(1, -1)}
        </em>,
      )
    last = m.index + tok.length
  }
  if (last < text.length) out.push(<Fragment key={k++}>{text.slice(last)}</Fragment>)
  return out
}

function Caret() {
  return <span className="ml-0.5 inline-block h-[1.1em] w-2 translate-y-[3px] rounded-[2px] bg-foreground animate-[caret-blink_1s_infinite]" />
}

export function CodeBlock({ code, lang }: { code: string; lang?: string }) {
  const [copied, setCopied] = useState(false)
  return (
    <div className="overflow-hidden rounded-2xl bg-muted/60 ring-1 ring-border/60">
      <div className="flex items-center justify-between py-0.5 pr-1 pl-3.5">
        <span className="font-mono text-xs text-muted-foreground">{lang || "text"}</span>
        <Button
          variant="ghost"
          size="xs"
          className="text-muted-foreground"
          onClick={() => {
            void copyText(code)
            setCopied(true)
            setTimeout(() => setCopied(false), 1500)
          }}
        >
          {copied ? <CheckIcon /> : <CopyIcon />}
          {copied ? "Copied" : "Copy"}
        </Button>
      </div>
      <pre className="overflow-x-auto px-3.5 pt-0.5 pb-3 font-mono text-[12.5px] leading-relaxed">
        <code>{code}</code>
      </pre>
    </div>
  )
}
