import { CheckIcon, ChevronDownIcon, CodeXmlIcon, SlidersHorizontalIcon, SquarePenIcon } from "lucide-react"

import { Button } from "@/components/ui/button"
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu"
import { SidebarTrigger, useSidebar } from "@/components/ui/sidebar"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { viewTransition } from "@/lib/browser"
import { useChat } from "@/lib/store"

export function ChatHeader({ models, onOpenSettings, onOpenCode }: { models: string[]; onOpenSettings: () => void; onOpenCode: () => void }) {
  const model = useChat((s) => s.settings.model)
  const setSettings = useChat((s) => s.setSettings)
  const newChat = useChat((s) => s.newChat)
  const { open, isMobile } = useSidebar()
  const options = models.includes(model) ? models : [model, ...models]

  return (
    <header className="flex h-[calc(2.75rem+env(safe-area-inset-top))] shrink-0 items-center gap-0.5 px-2 pt-[env(safe-area-inset-top)] max-md:h-[calc(3.25rem+env(safe-area-inset-top))]">
      <SidebarTrigger className="size-7 text-muted-foreground max-md:size-9" />
      {(!open || isMobile) && (
        <IconButton label="New chat" onClick={() => viewTransition(newChat)}>
          <SquarePenIcon />
        </IconButton>
      )}

      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button variant="ghost" size="sm" className="h-7 gap-1 px-2 text-sm font-medium max-md:h-9 max-md:text-[15px]">
            {model}
            <ChevronDownIcon className="size-3.5 opacity-50" />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="start" className="min-w-44">
          {options.map((id) => (
            <DropdownMenuItem key={id} onSelect={() => setSettings({ model: id })}>
              {id}
              {id === model && <CheckIcon className="ml-auto" />}
            </DropdownMenuItem>
          ))}
        </DropdownMenuContent>
      </DropdownMenu>

      <div className="ml-auto flex items-center gap-0.5">
        <IconButton label="View code" onClick={onOpenCode}>
          <CodeXmlIcon />
        </IconButton>
        <IconButton label="Settings" onClick={onOpenSettings}>
          <SlidersHorizontalIcon />
        </IconButton>
      </div>
    </header>
  )
}

function IconButton({ label, children, onClick }: { label: string; children: React.ReactNode; onClick: () => void }) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Button variant="ghost" size="icon-sm" className="text-muted-foreground max-md:size-9" aria-label={label} onClick={onClick}>
          {children}
        </Button>
      </TooltipTrigger>
      <TooltipContent>{label}</TooltipContent>
    </Tooltip>
  )
}
