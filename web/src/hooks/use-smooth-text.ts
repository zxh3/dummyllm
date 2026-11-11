import { useEffect, useLayoutEffect, useRef, useState } from "react"

/**
 * Reveal streamed text smoothly instead of a word at a time. Each frame shows a
 * slice of the backlog proportional to its size, so the text flows at a steady
 * pace and catches up quickly when chunks arrive in bursts. Text that is already
 * complete when it mounts (chat history) is shown immediately.
 */
export function useSmoothText(text: string, streaming: boolean): string {
  const [shown, setShown] = useState(() => (streaming ? 0 : text.length))
  const target = useRef(text)
  useLayoutEffect(() => {
    target.current = text
  }, [text])
  const behind = shown < text.length

  useEffect(() => {
    if (!behind) return
    let frame = 0
    const tick = () => {
      setShown((n) => {
        const backlog = target.current.length - n
        return backlog <= 0 ? n : n + Math.max(1, Math.ceil(backlog / 6))
      })
      frame = requestAnimationFrame(tick)
    }
    frame = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(frame)
  }, [behind])

  return text.slice(0, Math.max(shown, 0))
}
