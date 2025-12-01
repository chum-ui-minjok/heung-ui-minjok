import { useEffect, useState } from 'react';
import SongListItem from '@/components/SongListItem';
import type { Song } from '@/types/song';
import { useModeStore } from '@/store/modeStore';
import { getMusicSongList, getGameSongList } from '@/api/list';
import './SongListPage.css';

function SongListPage() {
  const { mode } = useModeStore();
  const [songs, setSongs] = useState<Song[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchSongs() {
      setLoading(true);
      try {
        const data =
          mode === 'LISTENING'
            ? await getMusicSongList()
            : await getGameSongList();

        const mapped: Song[] = data.map((item, index) => ({
          id: item.songId,
          rank: index + 1,
          title: item.title,
          artist: item.artist,
        }));

        setSongs(mapped);
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
