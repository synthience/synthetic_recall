import { create } from 'zustand';
import { queryClient } from './queryClient';
import { refreshAllData } from './api';

interface PollingStoreState {
  pollingRate: number;
  pollingInterval: number | null;
  isPolling: boolean;
  setPollingRate: (rate: number) => void;
  startPolling: () => void;
  stopPolling: () => void;
  refreshAllData: () => void;
}

export const usePollingStore = create<PollingStoreState>((set, get) => ({
  pollingRate: 5000, // Default polling rate: 5 seconds
  pollingInterval: null,
  isPolling: false,
  
  setPollingRate: (rate: number) => {
    set({ pollingRate: rate });
    
    // Restart polling with new rate if currently active
    if (get().isPolling) {
      get().stopPolling();
      get().startPolling();
    }
  },
  
  startPolling: () => {
    const { pollingInterval, pollingRate } = get();
    
    // Don't start another interval if one is already running
    if (pollingInterval !== null) {
      return;
    }
    
    // Create new polling interval
    const interval = window.setInterval(() => {
      get().refreshAllData();
    }, pollingRate);
    
    set({ pollingInterval: interval, isPolling: true });
  },
  
  stopPolling: () => {
    const { pollingInterval } = get();
    
    if (pollingInterval !== null) {
      clearInterval(pollingInterval);
      set({ pollingInterval: null, isPolling: false });
    }
  },
  
  refreshAllData: async () => {
    await refreshAllData(queryClient);
  }
}));

interface ThemeStore {
  isDarkMode: boolean;
  toggleDarkMode: () => void;
}

export const useThemeStore = create<ThemeStore>((set) => ({
  isDarkMode: true, // Default to dark mode for this dashboard
  toggleDarkMode: () => set((state) => ({ isDarkMode: !state.isDarkMode }))
}));

interface SidebarStore {
  isCollapsed: boolean;
  toggleSidebar: () => void;
}

export const useSidebarStore = create<SidebarStore>((set) => ({
  isCollapsed: false,
  toggleSidebar: () => set((state) => ({ isCollapsed: !state.isCollapsed }))
}));
