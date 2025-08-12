import { create } from 'zustand'

export const useAppStore = create((set, get) => ({
  // UI state
  sidebarOpen: false,
  theme: 'light',
  loading: false,
  lastError: null,

  // Data caches
  dashboard: null,

  // Actions
  toggleSidebar: () => set((s) => ({ sidebarOpen: !s.sidebarOpen })),
  setTheme: (theme) => set({ theme }),
  setLoading: (loading) => set({ loading }),
  setError: (err) => set({ lastError: err }),

  setDashboard: (data) => set({ dashboard: data }),
}))


