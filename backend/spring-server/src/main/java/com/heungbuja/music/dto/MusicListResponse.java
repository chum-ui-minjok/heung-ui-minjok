package com.heungbuja.music.dto;

import com.heungbuja.song.entity.Song;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;

/**
 * 음악 목록 응답 DTO
 */
@Getter
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class MusicListResponse {

    private Long songId;
    private String title;
    private String artist;

    public static MusicListResponse from(Song song) {
        return MusicListResponse.builder()
                .songId(song.getId())
                .title(song.getTitle())
                .artist(song.getArtist())
                .build();
    }
}
