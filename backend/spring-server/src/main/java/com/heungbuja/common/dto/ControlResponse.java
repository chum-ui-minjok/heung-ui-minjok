package com.heungbuja.common.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;

/**
 * 클릭 기반 컨트롤 API 공통 응답 DTO
 */
@Getter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class ControlResponse {

    private boolean success;
    private String message;

    public static ControlResponse success(String message) {
        return ControlResponse.builder()
                .success(true)
                .message(message)
                .build();
    }

    public static ControlResponse failure(String message) {
        return ControlResponse.builder()
                .success(false)
                .message(message)
                .build();
    }
}
