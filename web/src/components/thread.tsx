import { ArrowDownIcon } from "lucide-react"
import { useEffect, useLayoutEffect, useRef, useState } from "react"

import { AssistantTurn, UserBubble } from "@/components/message"
import { Button } from "@/components/ui/button"
import type { Chat } from "@/lib/types"

/** The scrolling message list. Sticks to the bottom while streaming unless you scroll up. Key it by chat id. */
export function Thread({ chat }: { chat: Chat }) {
  const scrollRef = useRef<HTMLDivElement>(null)
  const pinned = useRef(true)
  const [showJump, setShowJump] = useState(false)
  // Messages already present when the chat opens render without an entrance animation.
  // (The parent keys this component by chat id, so this runs once per chat.)
  const [initialIds] = useState(() => new Set(chat.messages.map((m) => m.id)))
  const animate = (id: string) => !initialIds.has(id)

  const nearBottom = () => {
    const el = scrollRef.current
    return !el || el.scrollHeight - el.scrollTop - el.clientHeight < 80
  }

  // Follow new content while pinned (and start at the bottom).
  useLayoutEffect(() => {
    const el = scrollRef.current
    if (el && pinned.current) el.scrollTop = el.scrollHeight
  }, [chat.messages])

  // A new user message always re-pins.
  const count = chat.messages.length
  useEffect(() => {
    pinned.current = true
  }, [count])

  return (
    <div className="relative min-h-0 flex-1">
      <div
        ref={scrollRef}
        className="scroll-fade-y h-full overflow-y-auto [scrollbar-gutter:stable]"
        onScroll={() => {
          const near = nearBottom()
          if (near) pinned.current = true
          setShowJump(!near)
        }}
        // Only explicit input un-pins; scroll events lag behind streamed content.
        onWheel={(e) => {
          if (e.deltaY < 0) pinned.current = false
        }}
        onTouchMove={() => {
          if (!nearBottom()) pinned.current = false
        }}
      >
        <div className="mx-auto flex w-full max-w-[44rem] flex-col gap-6 px-4 pt-3 pb-8 sm:px-5">
          {chat.messages.map((m, i) =>
            m.role === "user" ? (
              <UserBubble key={m.id} message={m} animate={animate(m.id)} />
            ) : (
              <AssistantTurn key={m.id} message={m} isLast={i === chat.messages.length - 1} animate={animate(m.id)} />
            ),
          )}
        </div>
      </div>
      {showJump && (
        <Button
          variant="outline"
          size="icon"
          aria-label="Scroll to bottom"
          className="absolute bottom-3 left-1/2 -translate-x-1/2 rounded-full bg-background shadow-sm animate-in fade-in-0 zoom-in-95"
          onClick={() => {
            pinned.current = true
            scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" })
          }}
        >
          <ArrowDownIcon />
        </Button>
      )}
    </div>
  )
}
