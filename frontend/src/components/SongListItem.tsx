import type { Song } from '@/types/song';
import './SongListItem.css';

interface Props {
  song: Song & { rank: number };
}

function SongListItem({ song }: Props) {
  return (
    <div className="song-item">
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
