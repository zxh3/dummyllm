import { cn } from "cn"

import logo from "@/assets/kiana.jpg"

/** The dummyllm mark: Kiana the cat (the original image, unedited). */
export function CatLogo({ className }: { className?: string }) {
  return (
    <img
      src={logo}
      alt=""
      draggable={false}
      className={cn("size-7 shrink-0 rounded-[9px] object-cover ring-1 ring-black/5 select-none dark:ring-white/10", className)}
    />
  )
}
