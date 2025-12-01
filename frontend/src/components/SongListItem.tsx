import type { RankedSong } from '@/types/song';
import './SongListItem.css';

interface Props {
  song: RankedSong;
  onClick?: (song: RankedSong) => void;
}

function SongListItem({ song, onClick }: Props) {
  const handleClick = () => {
    if (onClick) onClick(song);
  };

  return (
    <div className="song-item" onClick={handleClick}>
      <div className="song-item__rank">{song.rank}</div>

      <div className="song-item__card">
        <span className="song-item__title">{song.title}</span>
        <div>
          <span className="song-item__artist">{song.artist}</span>
          <span className="song-item__playCnt">{song.playCount} 회</span>
        </div>
      </div>
    </div>
  );
}

export default SongListItem;
