import { MoonIcon, SquarePenIcon, SunIcon, XIcon } from "lucide-react"
import { cn } from "cn"
import { useTheme } from "next-themes"
import { useState } from "react"

import { CatLogo } from "@/components/cat-logo"
import { Button } from "@/components/ui/button"
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuAction,
  SidebarMenuButton,
  SidebarMenuItem,
  useSidebar,
} from "@/components/ui/sidebar"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import type { ServerStatus } from "@/hooks/use-server"
import { revealTheme, viewTransition } from "@/lib/browser"
import { useChat } from "@/lib/store"

const STATUS: Record<ServerStatus, { label: string; dot: string }> = {
  connecting: { label: "Connecting", dot: "bg-muted-foreground animate-pulse" },
  online: { label: "Online", dot: "bg-emerald-500" },
  auth: { label: "Needs API key", dot: "bg-amber-500" },
  offline: { label: "Offline", dot: "bg-destructive" },
}

export function AppSidebar({ status }: { status: ServerStatus }) {
  const chats = useChat((s) => s.chats)
  const currentId = useChat((s) => s.currentId)
  const { newChat, selectChat, deleteChat } = useChat.getState()
  const { setOpenMobile } = useSidebar()
  const { resolvedTheme, setTheme } = useTheme()
  // Chats created after the page loaded slide in; the initial list doesn't animate.
  const [initialIds] = useState(() => new Set(chats.map((c) => c.id)))

  return (
    <Sidebar variant="inset">
      <SidebarHeader className="flex-row items-center gap-2 px-3 pt-2 pb-1">
        <CatLogo className="size-6 rounded-[7px]" />
        <span className="text-sm font-semibold tracking-tight">dummyllm</span>
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              variant="ghost"
              size="icon-sm"
              className="ml-auto text-muted-foreground hover:bg-sidebar-accent"
              aria-label="New chat"
              onClick={() => {
                viewTransition(newChat)
                setOpenMobile(false)
              }}
            >
              <SquarePenIcon />
            </Button>
          </TooltipTrigger>
          <TooltipContent>New chat</TooltipContent>
        </Tooltip>
      </SidebarHeader>

      <SidebarContent>
        <SidebarGroup className="pt-1">
          <SidebarMenu className="gap-0.5">
            {chats.map((c) => (
              <SidebarMenuItem
                key={c.id}
                className={cn(!initialIds.has(c.id) && "animate-in fade-in-0 slide-in-from-left-2 duration-300 ease-snappy")}
              >
                <SidebarMenuButton
                  isActive={c.id === currentId}
                  className={cn(
                    "h-8 rounded-lg px-2.5 text-[13px] text-sidebar-foreground/70 transition-colors max-md:h-10 max-md:text-sm",
                    // Hover is a lighter tint than the open chat, so neighbours never fuse into one block.
                    "hover:bg-sidebar-accent/50 hover:text-sidebar-foreground",
                    "data-[active=true]:bg-sidebar-accent data-[active=true]:font-medium data-[active=true]:text-sidebar-foreground data-[active=true]:hover:bg-sidebar-accent",
                  )}
                  onClick={() => {
                    if (c.id !== currentId) viewTransition(() => selectChat(c.id))
                    setOpenMobile(false)
                  }}
                >
                  <span className="truncate">{c.title}</span>
                </SidebarMenuButton>
                <SidebarMenuAction
                  showOnHover
                  className="top-1.5 right-1.5 size-5 rounded-md text-sidebar-foreground/45 hover:bg-transparent hover:text-sidebar-foreground max-md:top-2.5 max-md:right-2 max-md:size-6 [&>svg]:size-3.5"
                  aria-label="Delete chat"
                  onClick={() => deleteChat(c.id)}
                >
                  <XIcon />
                </SidebarMenuAction>
              </SidebarMenuItem>
            ))}
          </SidebarMenu>
        </SidebarGroup>
      </SidebarContent>

      <SidebarFooter className="flex-row items-center px-3 pt-1 pb-2">
        <span className="flex items-center gap-2 text-xs text-muted-foreground">
          <span className={cn("size-1.5 rounded-full", STATUS[status].dot)} />
          {STATUS[status].label}
        </span>
        <Button
          variant="ghost"
          size="icon-sm"
          className="ml-auto text-muted-foreground hover:bg-sidebar-accent"
          aria-label="Toggle theme"
          onClick={(e) => {
            const next = resolvedTheme === "dark" ? "light" : "dark"
            const r = e.currentTarget.getBoundingClientRect()
            revealTheme(
              () => {
                // Apply the class synchronously so the transition captures the new theme.
                document.documentElement.classList.toggle("dark", next === "dark")
                setTheme(next)
              },
              r.left + r.width / 2,
              r.top + r.height / 2,
            )
          }}
        >
          {resolvedTheme === "dark" ? <SunIcon /> : <MoonIcon />}
        </Button>
      </SidebarFooter>
    </Sidebar>
  )
}
