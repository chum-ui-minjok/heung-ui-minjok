import { useRef, useState, useEffect } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import type { SongInfo } from '@/types/song';
import { useAudioStore } from '@/store/audioStore';
import './SongPage.css';

interface SongPageState {
  songInfo: SongInfo;
  autoPlay?: boolean;
}

function SongPage() {
  const location = useLocation();
  const navigate = useNavigate();
  const state = location.state as SongPageState | null;

  const audioRef = useRef<HTMLAudioElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const { setAudioRef, setIsPlaying: setGlobalPlaying } = useAudioStore();

  // songInfo가 없으면 홈으로 리다이렉트
  useEffect(() => {
    if (!state?.songInfo) {
      navigate('/home', { replace: true });
    }
  }, [state, navigate]);

  // audioRef를 전역 store에 등록
  useEffect(() => {
    if (audioRef.current) {
      setAudioRef(audioRef.current);
    }
    return () => {
      setAudioRef(null);
    };
  }, [setAudioRef]);

  // 로컬 isPlaying ↔ 전역 isPlaying 동기화
  useEffect(() => {
    setGlobalPlaying(isPlaying);
  }, [isPlaying, setGlobalPlaying]);

  // 리다이렉트 중에는 화면 렌더링하지 않음
  if (!state?.songInfo) {
    return null;
  }

  const { songInfo, autoPlay = false } = state;

  return (
    <div className="container">
      <span className="song-title">{songInfo.title}</span>
      <span className="song-artist">{songInfo.artist}</span>

      <div className={`lp-container ${isPlaying ? 'playing' : 'stopped'}`}>
        <div className="lp-disc"></div>
        <div className="tonearm-base"></div>
        <div className="tonearm"></div>
      </div>

      <audio
        ref={audioRef}
        src={songInfo.audioUrl}
        autoPlay={autoPlay}
        onPlay={() => setIsPlaying(true)}
        onPause={() => setIsPlaying(false)}
        onEnded={() => setIsPlaying(false)}
      />
    </div>
  );
}

export default SongPage;
