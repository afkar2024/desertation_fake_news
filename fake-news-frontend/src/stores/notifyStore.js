import { create } from 'zustand'

let idCounter = 1

export const useNotifyStore = create((set, get) => ({
  toasts: [],
  push: (toast) => {
    const id = toast.id ?? idCounter++
    const item = { id, title: toast.title, message: toast.message, variant: toast.variant || 'info', ttl: toast.ttl || 4000 }
    set((s) => ({ toasts: [...s.toasts, item] }))
    // auto-remove
    setTimeout(() => get().remove(id), item.ttl)
  },
  remove: (id) => set((s) => ({ toasts: s.toasts.filter(t => t.id !== id) })),
}))


