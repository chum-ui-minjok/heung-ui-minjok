package com.heungbuja.music.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;

/**
 * 음악 재생 응답 DTO
 */
@Getter
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class MusicPlayResponse {

    private boolean success;
    private String message;
    private Long songId;
    private String title;
    private String artist;
    private String audioUrl;  // presigned URL

    public static MusicPlayResponse success(Long songId, String title, String artist, String audioUrl) {
        return MusicPlayResponse.builder()
                .success(true)
                .message("노래가 재생됩니다")
                .songId(songId)
                .title(title)
                .artist(artist)
                .audioUrl(audioUrl)
                .build();
    }

    public static MusicPlayResponse failure(String message) {
        return MusicPlayResponse.builder()
                .success(false)
                .message(message)
                .build();
    }
}
