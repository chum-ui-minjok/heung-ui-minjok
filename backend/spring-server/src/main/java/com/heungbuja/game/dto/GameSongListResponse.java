package com.heungbuja.game.dto;

import com.heungbuja.song.entity.Song;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;

@Getter
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class GameSongListResponse {

    private Long songId;
    private String title;
    private String artist;
    private Long playCount;

    public static GameSongListResponse from(Song song, Long playCount) {
        return GameSongListResponse.builder()
                .songId(song.getId())
                .title(song.getTitle())
                .artist(song.getArtist())
                .playCount(playCount != null ? playCount : 0L)
                .build();
    }
}
