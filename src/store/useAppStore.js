import { create } from "zustand";

export const useAppStore = create((set) => ({
  isAuthenticated: false,
  setIsAuthenticated: (status) => set({ isAuthenticated: status }),

  sidebarOpen: true,
  setSidebarOpen: (isOpen) => set({ sidebarOpen: isOpen }),

  alertCount: 0,
  setAlertCount: (count) => set({ alertCount: count }),

  // App-wide loading/error states could be handled here if needed
}));
