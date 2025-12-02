import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import SongListItem from '@/components/SongListItem';
import type { RankedSong, SongInfo } from '@/types/song';
import { useModeStore } from '@/store/modeStore';
import { getMusicSongList, getGameSongList } from '@/api/list';
import { playMusicApi } from '@/api/song';
import './SongListPage.css';

function SongListPage() {
  const navigate = useNavigate();
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

  const handleSongClick = async (song: RankedSong) => {
    if (mode === 'LISTENING') {
    // 1) 음악 재생 API 호출 → SongInfo 리턴
    const baseInfo = await playMusicApi(song.id);

    // 2) 리스트에서 알고 있는 재생 수 / 순위를 덮어씀
    const songInfo: SongInfo = {
      ...baseInfo,
      playCount: song.playCount,
      rank: song.rank,
    };

    // 3) SongPage로 이동 (state에 SongInfo + autoPlay 저장)
    navigate('/listening', {
      state: {
        songInfo,
        autoPlay: true,
      },
    });
  } else {
      // EXERCISE 모드: 튜토리얼로 이동, 이후 TutorialPage에서 gameStartApi(song.id) 호출
      navigate('/tutorial', {
        state: {
          songId: song.id,
        },
      });
    }
  };

  return (
    <div className="song-list-page">
      <div className="song-list__date-title">
        <p>{year}년 {month}월 인기차트</p>
      </div>

      <div className="song-list__container">
        {loading && <div className="song-list__loading">불러오는 중…</div>}
        {!loading &&
          songs.map((song) => (
            <SongListItem key={song.id} song={song} onClick={(s) => { void handleSongClick(s); }}/>
          ))}
      </div>
    </div>
  );
}

export default SongListPage;