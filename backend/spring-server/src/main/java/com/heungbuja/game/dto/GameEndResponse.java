package com.heungbuja.game.dto;

import lombok.Builder;
import lombok.Getter;
import java.util.List;
import java.util.Map;

@Getter
@Builder
public class GameEndResponse {
    // === 기본 정보 ===
    private double finalScore;          // 최종 점수 (100점 만점)
    private String message;             // 평가 문구
    private Integer finalLevel;         // 최종 결정된 레벨 (1, 2, 3)

    // === 전체 통계 ===
    private OverallStatistics overallStats;

    // === 절별 통계 ===
    private VerseStatistics verse1Stats;
    private VerseStatistics verse2Stats;

    // === 동작별 상세 통계 ===
    private List<ActionStatistics> actionStatsList;

    // === 기존 호환용 (단순 동작별 평균 점수) ===
    private Map<String, Double> scoresByAction;

    /**
     * 전체 게임 통계
     */
    @Getter
    @Builder
    public static class OverallStatistics {
        private int totalActions;       // 전체 동작 횟수
        private int perfectCount;       // Perfect 횟수
        private int goodCount;          // Good 횟수
        private int badCount;           // Bad 횟수
        private double perfectRate;     // Perfect 비율 (%)
        private double goodRate;        // Good 비율 (%)
        private double badRate;         // Bad 비율 (%)
        private double averageScore;    // 전체 평균 점수
    }

    /**
     * 절별 통계
     */
    @Getter
    @Builder
    public static class VerseStatistics {
        private int verse;              // 절 번호 (1 또는 2)
        private int totalActions;       // 해당 절의 동작 횟수
        private int perfectCount;
        private int goodCount;
        private int badCount;
        private double perfectRate;
        private double goodRate;
        private double badRate;
        private double averageScore;    // 해당 절 평균 점수
    }

    /**
     * 동작별 상세 통계
     */
    @Getter
    @Builder
    public static class ActionStatistics {
        private int actionCode;         // 동작 코드
        private String actionName;      // 동작 이름 (예: "손 박수")
        private int totalCount;         // 해당 동작 총 횟수
        private int perfectCount;
        private int goodCount;
        private int badCount;
        private double perfectRate;
        private double goodRate;
        private double badRate;
        private double averageScore;    // 해당 동작 평균 점수
    }
}