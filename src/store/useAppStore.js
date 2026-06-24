import { create } from "zustand";

export const useAppStore = create((set) => ({
  token: localStorage.getItem("asr_token") || null,
  setToken: (token) => {
    if (token) {
      localStorage.setItem("asr_token", token);
    } else {
      localStorage.removeItem("asr_token");
    }
    set({ token });
  },

  sidebarOpen: true,
  setSidebarOpen: (isOpen) => set({ sidebarOpen: isOpen }),

  alertCount: 0,
  setAlertCount: (count) => set({ alertCount: count }),

  // App-wide loading/error states could be handled here if needed
}));
