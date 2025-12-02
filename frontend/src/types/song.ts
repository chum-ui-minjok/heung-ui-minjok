export type SongMode = 'LISTENING' | 'EXERCISE';

// 리스트·상세 공통으로 쓰는 곡 타입
export interface Song {
  id: number;          // 곡 ID
  title: string;
  artist: string;
  playCount: number;   // 재생 횟수
  rank?: number;       // 차트에서만 사용 (옵션)
}

export type RankedSong = Song & { rank: number };

// 재생용 상세 정보
export interface SongInfo extends Song {
  mediaId: number;
  audioUrl: string;
  mode: SongMode;
}