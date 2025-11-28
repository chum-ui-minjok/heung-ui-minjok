package com.heungbuja.game.dto;

import lombok.Getter;
import lombok.NoArgsConstructor;
import java.util.List;

/**
 * MediaPipe Pose 좌표를 받는 WebSocket 요청 DTO (새로운 방식)
 * 기존 WebSocketFrameRequest는 Base64 이미지용으로 유지
 */
@Getter
@NoArgsConstructor
public class WebSocketPoseRequest {
    /**
     * 현재 게임 세션을 식별하는 고유 ID
     */
    private String sessionId;

    /**
     * MediaPipe Pose 좌표 데이터
     * 33개 랜드마크의 [x, y] 좌표 배열
     * 예: [[0.5, 0.3], [0.52, 0.31], ...]
     * x, y는 0~1 사이 값 (이미지 크기 기준 정규화)
     */
    private List<List<Double>> poseData;

    /**
     * 현재 노래 재생 시간 (초 단위, 예: 35.78)
     */
    private double currentPlayTime;
}
