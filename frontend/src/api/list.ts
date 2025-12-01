import api from './index';
import type { Song } from '@/types/song';

// 서버 응답 전용 DTO
interface SongListDto {
  songId: number;
  title: string;
  artist: string;
  playCount: number;
}

/** 노래 감상용 리스트 */
export const getMusicSongList = async (): Promise<Song[]> => {
  const data = await api.get<SongListDto[]>('/music/list', true);

  return data.map((item) => ({
    id: item.songId,
    title: item.title,
    artist: item.artist,
    playCount: item.playCount,
  }));
};

/** 체조용 리스트 */
export const getGameSongList = async (): Promise<Song[]> => {
  const data = await api.get<SongListDto[]>('/game/list', true);

  return data.map((item) => ({
    id: item.songId,
    title: item.title,
    artist: item.artist,
    playCount: item.playCount,
  }));
};
