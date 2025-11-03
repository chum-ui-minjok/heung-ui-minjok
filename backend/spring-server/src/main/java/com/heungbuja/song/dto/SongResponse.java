package com.heungbuja.song.dto;

import com.heungbuja.song.entity.Song;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;

@Getter
@Builder
@AllArgsConstructor
public class SongResponse {

    private Long id;
    private String title;
    private String artist;
    private String s3Url;

    public static SongResponse from(Song song) {
        return SongResponse.builder()
                .id(song.getId())
                .title(song.getTitle())
                .artist(song.getArtist())
                .s3Url(song.getS3Url())
                .build();
    }
}
