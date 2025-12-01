import api from './index';
import type { SongList } from '@/types/song';

/** 노래 감상용 리스트 */
export const getMusicSongList = () =>
  api.get<SongList[]>('/music/list', true);

/** 체조용 리스트 */
export const getGameSongList = () =>
  api.get<SongList[]>('/game/list', true);
