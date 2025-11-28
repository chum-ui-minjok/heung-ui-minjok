package com.heungbuja.game.domain;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.springframework.data.annotation.Id;
import org.springframework.data.mongodb.core.mapping.Document;

import java.time.LocalDateTime;
import java.util.List;

/**
 * Pose 학습 데이터 (MongoDB)
 *
 * 목적: 게임 플레이 중 수집된 Pose 좌표 데이터를 저장하여 모델 학습에 활용
 *
 * 활용 방안:
 * 1. 팀원들이 게임 플레이하면서 자동으로 학습 데이터 수집
 * 2. 나중에 export해서 모델 재학습에 사용
 * 3. 동작별 데이터 분포 분석
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
@Document(collection = "pose_training_data")
public class PoseTrainingData {

    @Id
    private String id;

    // 세션 정보
    private String sessionId;
    private Long userId;
    private Long songId;

    // 동작 정보
    private Integer actionCode;
    private String actionName;

    // Pose 프레임 데이터 (8프레임 x 33 landmarks x [x, y])
    private List<List<List<Double>>> poseFrames;
    private Integer frameCount;

    // AI 판정 결과 (나중에 업데이트)
    private Integer judgment;           // 0, 1, 2, 3 (AI 판정 결과)
    private Double targetProbability;   // 목표 동작일 확률
    private Boolean verified;           // 수동 검증 여부 (나중에 라벨링용)

    // 메타 정보
    private LocalDateTime createdAt;
    private String verse;               // "verse1" or "verse2"
    private Integer sequenceIndex;      // 타임라인 내 순서
}
