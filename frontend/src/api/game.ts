import type { GameStartResponse, GameEndResponse } from '@/types/game';
import api from './index';
import { useGameStore } from '@/store/gameStore';

export const gameStartApi = (songId: number) => {
  const userIdStr = localStorage.getItem('userId');

  if (!userIdStr) {
    throw new Error('사용자 ID가 없습니다. 다시 로그인해 주세요.');
  }

  const userId = Number(userIdStr);
  if (Number.isNaN(userId)) {
    throw new Error('잘못된 사용자 ID입니다.');
  }

  return api.post<GameStartResponse, { userId: number; songId: number }>(
    '/game/start',
    { userId, songId },
    true,
  );
};

export const gameEndApi = () => {
  const { sessionId } = useGameStore.getState();

  if (!sessionId) {
    throw new Error('세션 ID가 없습니다. 게임 시작 정보가 제대로 설정되지 않았습니다.');
  }

  const path = `/game/end?sessionId=${encodeURIComponent(sessionId)}`;
  return api.post<GameEndResponse, undefined>(
    path,
    undefined,
    false,
  );
};