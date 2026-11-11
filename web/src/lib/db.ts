import type { Chat } from "./types"

/** A tiny IndexedDB wrapper: one database, one `chats` store keyed by chat id. */

const DB_NAME = "dummyllm"
const DB_VERSION = 1
const STORE = "chats"

let opening: Promise<IDBDatabase> | null = null

function open(): Promise<IDBDatabase> {
  opening ??= new Promise<IDBDatabase>((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION)
    req.onupgradeneeded = () => {
      if (!req.result.objectStoreNames.contains(STORE)) req.result.createObjectStore(STORE, { keyPath: "id" })
    }
    req.onsuccess = () => {
      const db = req.result
      // Another tab upgraded the schema: let go so it can proceed, reopen lazily.
      db.onversionchange = () => {
        db.close()
        opening = null
      }
      resolve(db)
    }
    req.onerror = () => {
      opening = null
      reject(req.error)
    }
  })
  return opening
}

async function run<T>(mode: IDBTransactionMode, fn: (store: IDBObjectStore) => IDBRequest<T> | void): Promise<T> {
  const db = await open()
  return new Promise<T>((resolve, reject) => {
    const tx = db.transaction(STORE, mode)
    const req = fn(tx.objectStore(STORE))
    tx.oncomplete = () => resolve(req ? req.result : (undefined as T))
    tx.onerror = () => reject(tx.error)
    tx.onabort = () => reject(tx.error)
  })
}

export const chatDB = {
  all: () => run<Chat[]>("readonly", (s) => s.getAll()),
  put: (chats: Chat[]) =>
    run<void>("readwrite", (s) => {
      for (const c of chats) s.put(c)
    }),
  delete: (ids: string[]) =>
    run<void>("readwrite", (s) => {
      for (const id of ids) s.delete(id)
    }),
}
