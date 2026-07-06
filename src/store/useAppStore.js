import { create } from "zustand";

export const useAppStore = create((set) => ({
  isAuthenticated: false,
  setIsAuthenticated: (status) => set({ isAuthenticated: status }),

  // Current user's identity/role/team, populated from GET /auth/me after login.
  user: null,
  setUser: (user) => set({ user }),

  sidebarOpen: true,
  setSidebarOpen: (isOpen) => set({ sidebarOpen: isOpen }),

  alertCount: 0,
  setAlertCount: (count) => set({ alertCount: count }),

  // App-wide loading/error states could be handled here if needed
}));
