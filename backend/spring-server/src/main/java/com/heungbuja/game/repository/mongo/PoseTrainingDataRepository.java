package com.heungbuja.game.repository.mongo;

import com.heungbuja.game.domain.PoseTrainingData;
import org.springframework.data.mongodb.repository.MongoRepository;

import java.time.LocalDateTime;
import java.util.List;

public interface PoseTrainingDataRepository extends MongoRepository<PoseTrainingData, String> {

    /**
     * 세션별 학습 데이터 조회
     */
    List<PoseTrainingData> findBySessionIdOrderByCreatedAtAsc(String sessionId);

    /**
     * 동작별 학습 데이터 조회 (모델 학습용)
     */
    List<PoseTrainingData> findByActionCode(Integer actionCode);

    /**
     * 동작명으로 학습 데이터 조회
     */
    List<PoseTrainingData> findByActionName(String actionName);

    /**
     * 검증되지 않은 데이터 조회 (라벨링 작업용)
     */
    List<PoseTrainingData> findByVerifiedFalseOrVerifiedIsNull();

    /**
     * 기간별 데이터 조회
     */
    List<PoseTrainingData> findByCreatedAtBetween(LocalDateTime startDate, LocalDateTime endDate);

    /**
     * 동작별 데이터 개수
     */
    long countByActionCode(Integer actionCode);

    /**
     * 사용자별 데이터 조회
     */
    List<PoseTrainingData> findByUserId(Long userId);
}
