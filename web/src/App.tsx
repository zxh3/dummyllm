import { useEffect, useState } from "react"

import { AppSidebar } from "@/components/app-sidebar"
import { ChatHeader } from "@/components/chat-header"
import { CodeDialog } from "@/components/code-dialog"
import { Composer } from "@/components/composer"
import { SettingsSheet } from "@/components/settings-sheet"
import { Thread } from "@/components/thread"
import { SidebarInset, SidebarProvider } from "@/components/ui/sidebar"
import { Welcome } from "@/components/welcome"
import { useServer } from "@/hooks/use-server"
import { viewTransition } from "@/lib/browser"
import { currentChat, useChat } from "@/lib/store"

export default function App() {
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [codeOpen, setCodeOpen] = useState(false)
  const chat = useChat(currentChat)
  const ready = useChat((s) => s.ready)
  const { models, status } = useServer()

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault()
        if (useChat.getState().currentId) viewTransition(useChat.getState().newChat)
      } else if (e.key === "Escape" && useChat.getState().busy) {
        useChat.getState().stop()
      }
    }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [])

  return (
    <SidebarProvider>
      <AppSidebar status={status} />
      <SidebarInset className="h-dvh min-h-0 overflow-hidden md:h-[calc(100dvh-1rem)] md:ring-1 md:ring-border/70 dark:md:ring-white/[0.06]">
        <ChatHeader models={models} onOpenSettings={() => setSettingsOpen(true)} onOpenCode={() => setCodeOpen(true)} />
        {!ready ? (
          <div className="flex-1" />
        ) : chat && chat.messages.length > 0 ? (
          <>
            <Thread key={chat.id} chat={chat} />
            <div className="mx-auto w-full max-w-[44rem] px-3 pb-[max(0.625rem,env(safe-area-inset-bottom))] sm:px-4 sm:pb-3">
              <Composer />
            </div>
          </>
        ) : (
          <Welcome />
        )}
      </SidebarInset>
      <SettingsSheet open={settingsOpen} onOpenChange={setSettingsOpen} />
      <CodeDialog open={codeOpen} onOpenChange={setCodeOpen} />
    </SidebarProvider>
  )
}
