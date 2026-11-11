import { CheckIcon, ChevronRightIcon, CopyIcon, DownloadIcon, OctagonAlertIcon, RefreshCwIcon } from "lucide-react"
import { cn } from "cn"
import { memo, useState } from "react"

import { Markdown } from "@/components/markdown"
import { useSmoothText } from "@/hooks/use-smooth-text"
import { Button } from "@/components/ui/button"
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { copyText } from "@/lib/browser"
import { useChat } from "@/lib/store"
import type { AssistantMessage, Meta, ToolCall, UserMessage } from "@/lib/types"

const enter = "animate-in fade-in-0 slide-in-from-bottom-2 duration-300 ease-snappy"

export const UserBubble = memo(function UserBubble({ message, animate }: { message: UserMessage; animate?: boolean }) {
  return (
    <div className={cn("flex flex-col items-end gap-2", animate && enter)}>
      {message.images.length > 0 && (
        <div className="flex flex-wrap justify-end gap-2">
          {message.images.map((src, i) => (
            <img key={i} src={src} alt="" className="size-28 rounded-2xl object-cover ring-1 ring-border sm:size-32" />
          ))}
        </div>
      )}
      {message.content && (
        <div className="max-w-[80%] rounded-[18px] bg-muted px-3.5 py-2 text-[15px] leading-[1.6] break-words whitespace-pre-wrap">{message.content}</div>
      )}
    </div>
  )
})

export const AssistantTurn = memo(function AssistantTurn({
  message,
  isLast,
  animate,
}: {
  message: AssistantMessage
  isLast: boolean
  animate?: boolean
}) {
  const regenerate = useChat((s) => s.regenerate)
  const busy = useChat((s) => s.busy)
  const reasoningOn = useChat((s) => s.settings.reasoning !== "none")
  const streaming = !message.done
  const rounds = message.rounds
  const lastRound = rounds.at(-1)
  const hasOutput = rounds.some((r) => r.content || r.toolCalls.length) || !!message.image
  const waiting =
    streaming &&
    !message.image &&
    (!lastRound || (!lastRound.content && !lastRound.toolCalls.length) || (lastRound.toolCalls.length > 0 && lastRound.toolCalls.every((c) => c.result !== undefined)))
  const text = message.image ? message.caption ?? "" : rounds.map((r) => r.content).filter(Boolean).join("\n\n")

  return (
    <div className={cn("group/turn", animate && "animate-in fade-in-0 duration-300 ease-snappy")}>
      <div className="space-y-2.5">
        {!!message.meta.reasoning && <p className="text-sm text-muted-foreground">Thought for {message.meta.reasoning} tokens</p>}

        {rounds.map((r, ri) => (
          <div key={ri} className="space-y-2.5">
            {r.toolCalls.length > 0 && (
              <div className="space-y-1">
                {r.toolCalls.map((call, ci) => (
                  <ToolCallRow key={call.id || ci} call={call} streaming={streaming} />
                ))}
              </div>
            )}
            {r.content && <RoundText text={r.content} streaming={streaming && ri === rounds.length - 1 && !r.toolCalls.length} />}
          </div>
        ))}

        {message.image && <GeneratedImageView image={message.image} />}
        {message.image && message.caption && <Markdown text={message.caption} />}

        {waiting && (
          <div className="flex h-6 items-center gap-2.5">
            <span className="size-2 animate-pulse rounded-full bg-foreground" />
            {reasoningOn && <span className="shimmer text-sm text-muted-foreground">Thinking</span>}
          </div>
        )}

        {message.error && (
          <div className="flex items-center gap-2.5 rounded-xl bg-destructive/8 px-3 py-2 text-sm">
            <OctagonAlertIcon className="size-4 shrink-0 text-destructive" />
            <span className="min-w-0 flex-1 text-foreground/80">{message.error.message}</span>
            {isLast && (
              <Button variant="ghost" size="sm" disabled={busy} onClick={() => regenerate(message.id)}>
                Retry
              </Button>
            )}
          </div>
        )}

        {!streaming && !hasOutput && !message.error && (
          <p className="text-muted-foreground italic">{message.meta.interrupted ? "Interrupted" : message.meta.stopped ? "Stopped" : "…"}</p>
        )}
      </div>

      {!streaming && (
        <div
          className={cn(
            "mt-0.5 -ml-1.5 flex items-center transition-opacity duration-200",
            isLast ? "opacity-100" : "opacity-0 group-hover/turn:opacity-100 focus-within:opacity-100 [@media(hover:none)]:opacity-100",
          )}
        >
          <ActionButton label="Copy" onClick={() => void copyText(text)} disabled={!text}>
            <CopyIcon />
          </ActionButton>
          <ActionButton label="Regenerate" onClick={() => regenerate(message.id)} disabled={busy}>
            <RefreshCwIcon />
          </ActionButton>
          <MetaLine meta={message.meta} />
        </div>
      )}
    </div>
  )
})

/** A round's text, revealed smoothly while it streams. */
function RoundText({ text, streaming }: { text: string; streaming: boolean }) {
  const shown = useSmoothText(text, streaming)
  return <Markdown text={shown} streaming={streaming || shown.length < text.length} />
}

function ActionButton({ label, children, ...props }: React.ComponentProps<typeof Button> & { label: string }) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Button variant="ghost" size="icon-sm" className="text-muted-foreground hover:text-foreground max-md:size-9" aria-label={label} {...props}>
          {children}
        </Button>
      </TooltipTrigger>
      <TooltipContent>{label}</TooltipContent>
    </Tooltip>
  )
}

function MetaLine({ meta }: { meta: Meta }) {
  const parts: string[] = []
  if (meta.ttft != null) parts.push(meta.ttft >= 1000 ? `${(meta.ttft / 1000).toFixed(1)}s` : `${Math.round(meta.ttft)}ms`)
  if (meta.completion) parts.push(`${meta.completion} tok`)
  if (meta.completion && meta.firstAt && meta.end && meta.end > meta.firstAt + 30)
    parts.push(`${Math.round(meta.completion / ((meta.end - meta.firstAt) / 1000))} tok/s`)
  if (meta.finish === "length") parts.push("max tokens")
  if (!parts.length) return null
  return (
    <span className="ml-2 font-mono text-[11px] text-muted-foreground/70 tabular-nums opacity-0 transition-opacity group-hover/turn:opacity-100">
      {parts.join("  ·  ")}
    </span>
  )
}

function ToolCallRow({ call, streaming }: { call: ToolCall; streaming: boolean }) {
  const [open, setOpen] = useState(false)
  const done = call.result !== undefined
  const active = !done && (call.running || streaming)
  let args = call.arguments
  try {
    args = JSON.stringify(JSON.parse(call.arguments), null, 2)
  } catch {
    /* still streaming */
  }
  return (
    <Collapsible open={open} onOpenChange={setOpen}>
      <CollapsibleTrigger className="group/tool inline-flex items-center gap-2 rounded-md py-1 text-sm text-muted-foreground transition-colors hover:text-foreground">
        {done ? <CheckIcon className="size-3.5" /> : <span className={cn("size-1.5 rounded-full bg-brand", active && "animate-pulse")} />}
        <span className={cn(active && "shimmer")}>
          {done ? "Called" : "Calling"} <span className="font-mono text-[13px]">{call.name || "…"}</span>
        </span>
        <ChevronRightIcon className={cn("size-3.5 opacity-0 transition group-hover/tool:opacity-100 [@media(hover:none)]:opacity-100", open && "rotate-90 opacity-100")} />
      </CollapsibleTrigger>
      <CollapsibleContent>
        <div className="mt-1.5 mb-2 grid gap-px overflow-hidden rounded-xl bg-border ring-1 ring-border">
          <Pane label="Arguments" value={args || " "} />
          {done && <Pane label="Result" value={JSON.stringify(call.result, null, 2)} />}
        </div>
      </CollapsibleContent>
    </Collapsible>
  )
}

function Pane({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-card px-3.5 py-3">
      <div className="mb-1.5 text-[11px] font-medium text-muted-foreground">{label}</div>
      <pre className="overflow-x-auto font-mono text-xs leading-relaxed break-words whitespace-pre-wrap">{value}</pre>
    </div>
  )
}

function GeneratedImageView({ image }: { image: NonNullable<AssistantMessage["image"]> }) {
  if (image.pending || !image.b64) {
    return (
      <div className="grid aspect-square w-full max-w-xs animate-pulse place-items-center rounded-[18px] bg-muted">
        <span className="shimmer text-sm text-muted-foreground">Creating image</span>
      </div>
    )
  }
  const src = `data:image/png;base64,${image.b64}`
  return (
    <figure className="group/img relative w-full max-w-xs animate-in fade-in-0 zoom-in-[0.98] duration-500">
      <img src={src} alt="A drawing of a cat" className="aspect-square w-full rounded-[18px] object-cover ring-1 ring-border [image-rendering:pixelated]" />
      <Button
        variant="secondary"
        size="icon-sm"
        asChild
        className="absolute top-2.5 right-2.5 rounded-full bg-black/55 text-white opacity-0 backdrop-blur transition-opacity group-hover/img:opacity-100 hover:bg-black/70 [@media(hover:none)]:opacity-100"
      >
        <a href={src} download="dummyllm-cat.png" aria-label="Download image">
          <DownloadIcon />
        </a>
      </Button>
    </figure>
  )
}
