import api from './index';
import type { SongInfo } from '@/types/song';

interface MusicPlayDto {
  success: boolean;
  message: string;
  songId: number;
  title: string;
  artist: string;
  audioUrl: string;
}

export const playMusicApi = async (songId: number): Promise<SongInfo> => {
  const res = await api.post<MusicPlayDto, { songId: number }>(
    '/music/play',
    { songId },
    true,
  );

  if (!res.success) {
    throw new Error(res.message || '음악 재생에 실패했습니다.');
  }

  // DTO → 도메인 타입으로 변환
  const songInfo: SongInfo = {
    id: res.songId,
    title: res.title,
    artist: res.artist,
    audioUrl: res.audioUrl,
    playCount: 0,        // 리스트에서 받아오면 그 값으로 덮어씌워도 됨
    mediaId: res.songId, // mediaId 별도 없으니 songId로 매핑
    mode: 'LISTENING',
  };

  return songInfo;
};
