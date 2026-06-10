/**
 * Zustand auth store for global authentication state.
 */

import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { User } from "@/lib/api";
import api from "@/lib/api";

interface AuthState {
  user: User | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;
  login: (username: string, password: string) => Promise<void>;
  logout: () => void;
  setUser: (user: User) => void;
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set) => ({
      user: null,
      isAuthenticated: false,
      isLoading: false,
      error: null,

      login: async (username, password) => {
        set({ isLoading: true, error: null });
        try {
          const { user } = await api.login(username, password);
          set({ user, isAuthenticated: true, isLoading: false });
        } catch (err: any) {
          const msg = err.response?.data?.detail || "Login failed";
          set({ error: msg, isLoading: false, isAuthenticated: false });
          throw new Error(msg);
        }
      },

      logout: () => {
        api.logout();
        set({ user: null, isAuthenticated: false, error: null });
      },

      setUser: (user) => set({ user, isAuthenticated: true }),
    }),
    {
      name: "atmnet-auth",
      partialize: (state) => ({
        user: state.user,
        isAuthenticated: state.isAuthenticated,
      }),
    }
  )
);
