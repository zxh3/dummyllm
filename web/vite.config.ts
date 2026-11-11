import path from "node:path"
import tailwindcss from "@tailwindcss/vite"
import react from "@vitejs/plugin-react"
import { defineConfig } from "vite"

// The build is served by the dummyllm Python package (at / and /playground), so it
// goes straight into the package. In dev, API calls are proxied to a local server.
const target = process.env.DUMMYLLM_URL ?? "http://127.0.0.1:8000"

export default defineConfig({
  base: "./",
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: { "@": path.resolve(__dirname, "./src") },
  },
  build: {
    outDir: path.resolve(__dirname, "../src/dummyllm/static/playground"),
    emptyOutDir: true,
    chunkSizeWarningLimit: 1000,
  },
  server: {
    proxy: { "/v1": target, "/health": target, "/info": target },
  },
})
