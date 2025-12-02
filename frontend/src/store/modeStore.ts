import { create } from 'zustand';

export type PlayMode = 'LISTENING' | 'EXERCISE';

interface ModeState {
  mode: PlayMode;
  setMode: (mode: PlayMode) => void;
}

export const useModeStore = create<ModeState>((set) => ({
  mode: 'LISTENING',
  setMode: (mode) => set({ mode }),
}));
