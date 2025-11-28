package com.heungbuja.game.dto;

import lombok.Builder;
import lombok.Getter;
import java.util.List;

/**
 * Motion 서버에 Pose 좌표 데이터를 전송하는 요청 DTO (새로운 방식)
 * 기존 AiAnalyzeRequest는 Base64 이미지용으로 유지
 */
@Getter
@Builder
public class AiPoseAnalyzeRequest {
    /**
     * 동작 코드 (DB 기준)
     */
    private int actionCode;

    /**
     * 동작 이름 (예: "손 박수", "팔 치기")
     */
    private String actionName;

    /**
     * 프레임 수
     */
    private int frameCount;

    /**
     * Pose 좌표 프레임 리스트
     * 각 프레임은 33개 랜드마크의 [x, y] 좌표 배열
     * 형태: List<List<List<Double>>> = [frame1, frame2, ...]
     * 각 frame = [[x0, y0], [x1, y1], ..., [x32, y32]]
     */
    private List<List<List<Double>>> poseFrames;
}
