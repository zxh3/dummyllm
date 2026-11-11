import { ArrowUpIcon, ImageIcon, PaperclipIcon, PlusIcon, WrenchIcon, XIcon } from "lucide-react"
import { cn } from "cn"
import { useEffect, useRef, useState } from "react"
import { toast } from "sonner"
import { useShallow } from "zustand/react/shallow"

import { Button } from "@/components/ui/button"
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { Textarea } from "@/components/ui/textarea"
import { downscaleImage, isTouch, viewTransition } from "@/lib/browser"
import { useChat } from "@/lib/store"

/** `centered` is the new-chat layout: sending morphs the composer down into the conversation. */
export function Composer({ autoFocus, centered }: { autoFocus?: boolean; centered?: boolean }) {
  const [text, setText] = useState("")
  const [dragging, setDragging] = useState(false)
  const inputRef = useRef<HTMLTextAreaElement>(null)
  const fileRef = useRef<HTMLInputElement>(null)
  const { busy, attachments, tools, image, send, stop, setSettings, addAttachments, removeAttachment } = useChat(
    useShallow((s) => ({
      busy: s.busy,
      attachments: s.attachments,
      tools: s.settings.tools,
      image: s.settings.image,
      send: s.send,
      stop: s.stop,
      setSettings: s.setSettings,
      addAttachments: s.addAttachments,
      removeAttachment: s.removeAttachment,
    })),
  )

  // Keep the caret in the box on desktop; on phones that would pop the keyboard.
  useEffect(() => {
    if ((autoFocus || !busy) && !isTouch()) inputRef.current?.focus()
  }, [autoFocus, busy])

  const canSend = !busy && (text.trim().length > 0 || attachments.length > 0)
  const submit = () => {
    if (!canSend) return
    const go = () => {
      void send(text)
      setText("")
    }
    if (centered) viewTransition(go)
    else go()
  }

  async function addFiles(files: FileList | File[]) {
    const images = [...files].filter((f) => f.type.startsWith("image/"))
    if (!images.length) return
    if (attachments.length + images.length > 4) toast("Up to 4 images per message")
    try {
      addAttachments(await Promise.all(images.slice(0, 4 - attachments.length).map((f) => downscaleImage(f))))
    } catch {
      toast.error("Couldn't read that image")
    }
  }

  return (
    <form
      onSubmit={(e) => {
        e.preventDefault()
        submit()
      }}
      onDragOver={(e) => {
        e.preventDefault()
        setDragging(true)
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault()
        setDragging(false)
        void addFiles(e.dataTransfer.files)
      }}
      style={{ viewTransitionName: "composer" }}
      className={cn(
        "w-full rounded-[22px] border bg-composer p-1.5 shadow-[0_1px_2px_rgb(0_0_0/0.04),0_12px_32px_-12px_rgb(0_0_0/0.14)] transition-[border-color,box-shadow] duration-200 ease-snappy dark:shadow-[0_12px_32px_-12px_rgb(0_0_0/0.6)]",
        "focus-within:border-ring/60 focus-within:ring-4 focus-within:ring-ring/[0.07]",
        dragging && "border-brand ring-4 ring-brand/15",
      )}
    >
      {attachments.length > 0 && (
        <div className="flex flex-wrap gap-1.5 px-1 pt-1 pb-1">
          {attachments.map((src, i) => (
            <div key={i} className="group/thumb relative size-12 overflow-hidden rounded-xl ring-1 ring-border animate-in fade-in-0 zoom-in-95">
              <img src={src} alt="" className="size-full object-cover" />
              <button
                type="button"
                aria-label="Remove image"
                onClick={() => removeAttachment(i)}
                className="absolute top-1 right-1 grid size-5 place-items-center rounded-full bg-black/70 text-white opacity-0 transition-opacity group-hover/thumb:opacity-100 focus:opacity-100"
              >
                <XIcon className="size-3" />
              </button>
            </div>
          ))}
        </div>
      )}

      <Textarea
        ref={inputRef}
        value={text}
        rows={1}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={(e) => {
          // On touch keyboards Enter makes a new line; send with the button.
          if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing && !isTouch()) {
            e.preventDefault()
            submit()
          }
        }}
        onPaste={(e) => {
          const files = [...e.clipboardData.files].filter((f) => f.type.startsWith("image/"))
          if (files.length) {
            e.preventDefault()
            void addFiles(files)
          }
        }}
        placeholder={image ? "Describe an image" : "Ask anything"}
        aria-label="Message"
        className="max-h-52 min-h-9 resize-none rounded-none border-0 bg-transparent px-2 pt-1.5 pb-0.5 text-base shadow-none placeholder:text-muted-foreground/80 focus-visible:ring-0 md:text-[15px] dark:bg-transparent"
      />

      <div className="flex items-center gap-1">
        <input
          ref={fileRef}
          type="file"
          accept="image/png,image/jpeg,image/gif,image/webp"
          multiple
          hidden
          onChange={(e) => {
            if (e.target.files) void addFiles(e.target.files)
            e.target.value = ""
          }}
        />
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button type="button" variant="ghost" size="icon" className="size-8 rounded-full text-muted-foreground max-md:size-9" aria-label="Add">
              <PlusIcon className="size-[18px]" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent side="top" align="start" className="w-52">
            <DropdownMenuItem onSelect={() => fileRef.current?.click()}>
              <PaperclipIcon /> Add photos
            </DropdownMenuItem>
            <DropdownMenuSeparator />
            <DropdownMenuCheckboxItem checked={tools} onCheckedChange={(on) => setSettings({ tools: on, image: on ? false : image })}>
              <WrenchIcon /> Use tools
            </DropdownMenuCheckboxItem>
            <DropdownMenuCheckboxItem checked={image} onCheckedChange={(on) => setSettings({ image: on, tools: on ? false : tools })}>
              <ImageIcon /> Create image
            </DropdownMenuCheckboxItem>
          </DropdownMenuContent>
        </DropdownMenu>

        {tools && <ModeChip icon={<WrenchIcon />} label="Tools" onClear={() => setSettings({ tools: false })} />}
        {image && <ModeChip icon={<ImageIcon />} label="Image" onClear={() => setSettings({ image: false })} />}

        <Button
          type={busy ? "button" : "submit"}
          size="icon"
          onClick={busy ? stop : undefined}
          disabled={!busy && !canSend}
          aria-label={busy ? "Stop" : "Send"}
          className={cn(
            "ml-auto size-8 rounded-full transition-all max-md:size-9",
            busy ? "bg-foreground text-background hover:bg-foreground/85" : "bg-brand text-brand-foreground hover:bg-brand/90",
            "disabled:bg-muted disabled:text-muted-foreground disabled:opacity-100",
          )}
        >
          <span key={busy ? "stop" : "send"} className="grid place-items-center animate-in fade-in-0 zoom-in-50 duration-200 ease-snappy">
            {busy ? <span className="size-2.5 rounded-[2px] bg-current" /> : <ArrowUpIcon className="size-4" strokeWidth={2.4} />}
          </span>
        </Button>
      </div>
    </form>
  )
}

function ModeChip({ icon, label, onClear }: { icon: React.ReactNode; label: string; onClear: () => void }) {
  return (
    <button
      type="button"
      onClick={onClear}
      aria-label={`Turn off ${label}`}
      className="group/chip inline-flex h-7 items-center max-md:h-8 gap-1.5 rounded-full bg-brand/10 pr-2 pl-2.5 text-xs font-medium text-brand transition-colors hover:bg-brand/15 animate-in fade-in-0 zoom-in-95 [&_svg]:size-3.5"
    >
      {icon}
      {label}
      <XIcon className="opacity-60 group-hover/chip:opacity-100" />
    </button>
  )
}
