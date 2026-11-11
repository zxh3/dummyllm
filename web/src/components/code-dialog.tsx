import { CheckIcon, CopyIcon } from "lucide-react"
import { Fragment, useMemo, useState, type ReactNode } from "react"
import { useShallow } from "zustand/react/shallow"

import { Button } from "@/components/ui/button"
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { copyText } from "@/lib/browser"
import { snippet, type SnippetLang } from "@/lib/snippets"
import { currentChat, useChat } from "@/lib/store"

export function CodeDialog({ open, onOpenChange }: { open: boolean; onOpenChange: (open: boolean) => void }) {
  const [lang, setLang] = useState<SnippetLang>("python")
  const [copied, setCopied] = useState(false)
  const { chat, settings } = useChat(useShallow((s) => ({ chat: currentChat(s), settings: s.settings })))
  // Only build the snippet while open, but keep the last one through the close
  // animation; blanking it there would collapse and re-center the dialog mid-fade.
  const live = useMemo(() => (open ? snippet(lang, chat, settings) : null), [open, lang, chat, settings])
  const [code, setCode] = useState("")
  if (live !== null && live !== code) setCode(live)

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="flex max-h-[min(85vh,760px)] flex-col gap-4 sm:max-w-3xl">
        <DialogHeader>
          <DialogTitle>Use this conversation in code</DialogTitle>
          <DialogDescription>dummyllm speaks the OpenAI API, so any OpenAI SDK works. Just point the base URL here.</DialogDescription>
        </DialogHeader>

        <div className="flex items-center justify-between gap-3">
          <Tabs value={lang} onValueChange={(v) => setLang(v as SnippetLang)}>
            <TabsList>
              <TabsTrigger value="python">Python</TabsTrigger>
              <TabsTrigger value="node">Node.js</TabsTrigger>
              <TabsTrigger value="curl">curl</TabsTrigger>
            </TabsList>
          </Tabs>
          <Button
            variant="outline"
            size="sm"
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

        <CodeView code={code} lang={lang} />
      </DialogContent>
    </Dialog>
  )
}

/**
 * Wraps long lines instead of scrolling sideways. Each line keeps its indent and
 * wrapped continuations hang a few columns further in, so structure stays readable.
 */
function CodeView({ code, lang }: { code: string; lang: SnippetLang }) {
  const lines = useMemo(() => code.split("\n"), [code])
  return (
    // The fade mask lives on the inner scroller so the box's border stays crisp.
    <div className="flex min-h-0 flex-col overflow-hidden rounded-xl bg-muted/50 ring-1 ring-border/70">
      <div className="scroll-fade-y min-h-0 overflow-y-auto py-3.5 font-mono text-[12.5px] leading-[1.65]">
        {lines.map((line, i) => {
          const indent = line.length - line.trimStart().length
          return (
            <div key={i} className="pr-4 break-words whitespace-pre-wrap" style={{ paddingLeft: `calc(1rem + ${indent + 4}ch)`, textIndent: "-4ch" }}>
              {line.trim() ? highlight(line.trimStart(), lang) : "\u00a0"}
            </div>
          )
        })}
      </div>
    </div>
  )
}

const KEYWORDS: Record<SnippetLang, string[]> = {
  python: ["from", "import", "print", "True", "False", "None"],
  node: ["import", "from", "const", "await", "new", "true", "false", "null", "console"],
  curl: ["curl", "true", "false", "null"],
}

/** Tiny highlighter: strings, numbers, keywords, and flags. Enough for these snippets. */
function highlight(line: string, lang: SnippetLang): ReactNode[] {
  const kw = KEYWORDS[lang].join("|")
  // Numbers after ":" or "/" are ports/paths in URLs, not values.
  const re = new RegExp(`("(?:[^"\\\\]|\\\\.)*")|((?<![\\w:/.])\\d+(?:\\.\\d+)?\\b)|\\b(${kw})\\b|(\\s-[A-Za-z]\\b)`, "g")
  const out: ReactNode[] = []
  let last = 0
  let m: RegExpExecArray | null
  while ((m = re.exec(line))) {
    if (m.index > last) out.push(<Fragment key={last}>{line.slice(last, m.index)}</Fragment>)
    const cls = m[1] ? "text-emerald-700 dark:text-emerald-400" : m[2] ? "text-amber-700 dark:text-amber-400" : "text-violet-700 dark:text-violet-400"
    out.push(
      <span key={m.index} className={cls}>
        {m[0]}
      </span>,
    )
    last = m.index + m[0].length
  }
  if (last < line.length) out.push(<Fragment key={last}>{line.slice(last)}</Fragment>)
  return out
}
