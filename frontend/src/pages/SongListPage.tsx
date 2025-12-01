import { useEffect, useState } from 'react';
import SongListItem from '@/components/SongListItem';
import type { RankedSong } from '@/types/song';
import { useModeStore } from '@/store/modeStore';
import { getMusicSongList, getGameSongList } from '@/api/list';
import './SongListPage.css';

function SongListPage() {
  const { mode } = useModeStore();
  const [songs, setSongs] = useState<RankedSong[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchSongs() {
      setLoading(true);
      try {
        const baseSongs =
          mode === 'LISTENING'
            ? await getMusicSongList()
            : await getGameSongList();

        // 순위 부여
        const withRank: RankedSong[] = baseSongs.map((song, index) => ({
          ...song,
          rank: index + 1,
        }));

        setSongs(withRank);
      } catch (e) {
        console.error('노래 목록 조회 실패:', e);
        setSongs([]);
      } finally {
        setLoading(false);
      }
    }

    void fetchSongs();
  }, [mode]);

  const now = new Date();
  const year = now.getFullYear();
  const month = now.getMonth() + 1;

  return (
    <div className="song-list-page">
      <div className="song-list__date-title">
        <p>{year}년 {month}월 인기차트</p>
      </div>

      <div className="song-list__container">
        {loading && <div className="song-list__loading">불러오는 중…</div>}
        {!loading &&
          songs.map((song) => (
            <SongListItem key={song.id} song={song} />
          ))}
      </div>
    </div>
  );
}

export default SongListPage;