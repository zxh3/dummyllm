import { useEffect, useState } from "react"

import { ApiError, fetchModels } from "@/lib/api"
import { useChat } from "@/lib/store"

export type ServerStatus = "connecting" | "online" | "auth" | "offline"

/** Models advertised by the server, plus a connection status for the sidebar. */
export function useServer() {
  const apiKey = useChat((s) => s.settings.apiKey)
  const [models, setModels] = useState<string[]>([])
  const [status, setStatus] = useState<ServerStatus>("connecting")

  useEffect(() => {
    let cancelled = false
    fetchModels(useChat.getState().settings)
      .then((ids) => {
        if (cancelled) return
        setModels(ids)
        setStatus("online")
      })
      .catch((e: ApiError) => {
        if (!cancelled) setStatus(e.status === 401 ? "auth" : "offline")
      })
    return () => {
      cancelled = true
    }
  }, [apiKey])

  return { models, status }
}
