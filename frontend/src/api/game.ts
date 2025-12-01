import type { GameEndResponse } from '@/types/game';
import api from './index';
import { useGameStore } from '@/store/gameStore';
import { mockGameStart } from '@/mocks/gameStart.mock';

// const USE_MOCK = import.meta.env.VITE_USE_MOCK === 'true';

export const gameStartApi = () => {
  // if (USE_MOCK) {
    return mockGameStart();
  // }
  // return api.post<GameStartResponse>('/game/start', { songId }, true);
}

export const gameEndApi = (): Promise<GameEndResponse> => {
  const { sessionId } = useGameStore.getState();

  if (!sessionId) {
    console.warn('세션 ID가 없습니다. gameEndApi 호출 스킵.');
    return Promise.resolve({
      finalScore: 0,
      message: '',
    });
  }

  const path = `/game/end?sessionId=${encodeURIComponent(sessionId)}`;
  return api.post<GameEndResponse, undefined>(
    path,
    undefined,
    false,
  );
};