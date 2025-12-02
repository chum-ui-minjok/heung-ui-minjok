package com.heungbuja.music.dto;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;

/**
 * 음악 재생 요청 DTO
 */
@Getter
@NoArgsConstructor
@AllArgsConstructor
public class MusicPlayRequest {

    private Long songId;
}
