import { flushSync } from "react-dom"
import { toast } from "sonner"

/** Downscale an image file to a JPEG data URL (keeps requests and localStorage small). */
export function downscaleImage(file: File, max = 768): Promise<string> {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.onload = () => {
      const s = Math.min(1, max / Math.max(img.width, img.height))
      const canvas = document.createElement("canvas")
      canvas.width = Math.max(1, Math.round(img.width * s))
      canvas.height = Math.max(1, Math.round(img.height * s))
      canvas.getContext("2d")!.drawImage(img, 0, 0, canvas.width, canvas.height)
      URL.revokeObjectURL(img.src)
      resolve(canvas.toDataURL("image/jpeg", 0.85))
    }
    img.onerror = reject
    img.src = URL.createObjectURL(file)
  })
}

export async function copyText(text: string, label = "Copied to clipboard") {
  try {
    await navigator.clipboard.writeText(text)
    toast.success(label)
  } catch {
    toast.error("Couldn't copy")
  }
}

export const isMac = typeof navigator !== "undefined" && /Mac|iPhone|iPad/.test(navigator.platform)

/** Phones and tablets: no hover, finger-sized pointer. */
export const isTouch = () => matchMedia("(hover: none) and (pointer: coarse)").matches

const prefersReducedMotion = () => matchMedia("(prefers-reduced-motion: reduce)").matches

/**
 * Run a state update inside a View Transition (crossfade + shared-element morphs,
 * e.g. the composer gliding between the welcome screen and a conversation).
 * Falls back to a plain update where unsupported or when motion is reduced.
 */
export function viewTransition(update: () => void): ViewTransition | null {
  if (!document.startViewTransition || prefersReducedMotion()) {
    update()
    return null
  }
  return startGuarded(update)
}

/**
 * startViewTransition defers the update until the old frame is captured. If the
 * page isn't rendering (e.g. an occluded or zero-size embed) that never happens,
 * so skip the animation after a moment; skipping still runs the update.
 */
function startGuarded(update: () => void): ViewTransition {
  let ran = false
  const t = document.startViewTransition(() => {
    ran = true
    flushSync(update)
  })
  setTimeout(() => {
    if (!ran) t.skipTransition()
  }, 300)
  t.finished.catch(() => {})
  return t
}

/** Switch theme with a circular reveal that grows from (x, y). */
export function revealTheme(apply: () => void, x: number, y: number) {
  if (!document.startViewTransition || prefersReducedMotion()) {
    apply()
    return
  }
  const root = document.documentElement
  const radius = Math.hypot(Math.max(x, innerWidth - x), Math.max(y, innerHeight - y))
  root.classList.add("theme-transition")
  const t = startGuarded(apply)
  t.ready
    .then(() =>
      root.animate(
        { clipPath: [`circle(0px at ${x}px ${y}px)`, `circle(${radius}px at ${x}px ${y}px)`] },
        { duration: 520, easing: "cubic-bezier(0.32, 0.72, 0, 1)", pseudoElement: "::view-transition-new(root)" },
      ),
    )
    .catch(() => {})
  const done = () => root.classList.remove("theme-transition")
  t.finished.then(done, done)
}
