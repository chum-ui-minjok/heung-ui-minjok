package com.heungbuja.game.service;

import com.heungbuja.game.dto.FrameBatchRequest;
import com.heungbuja.game.dto.FrameBatchResponse;
import com.heungbuja.game.dto.GameStartRequest;
import com.heungbuja.game.dto.GameStartResponse;
import com.heungbuja.game.state.GameState;
import com.heungbuja.song.domain.SongBeat;
import com.heungbuja.song.domain.SongChoreography;
import com.heungbuja.song.domain.SongLyrics;
import com.heungbuja.song.entity.Song;
import com.heungbuja.song.repository.SongBeatRepository;
import com.heungbuja.song.repository.SongChoreographyRepository;
import com.heungbuja.song.repository.SongLyricsRepository;
import com.heungbuja.song.repository.SongRepository;
import com.heungbuja.user.entity.User;
import com.heungbuja.user.repository.UserRepository;
import com.heungbuja.game.repository.ActionRepository;
import com.heungbuja.game.entity.GameResult;
import com.heungbuja.game.repository.GameResultRepository;
import com.heungbuja.game.factory.ScoringStrategyFactory;
import com.heungbuja.game.strategy.ScoringStrategy;
import com.heungbuja.game.enums.GameStatus;
import com.heungbuja.common.exception.CustomException;
import com.heungbuja.common.exception.ErrorCode;

import lombok.Getter;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;

import java.time.Duration;
import java.util.List;
import java.util.UUID;
import java.util.Map;

@Slf4j
@Service
@RequiredArgsConstructor
public class GameService {

    // --- 상수 정의 ---
    /** 1절을 구성하는 총 세그먼트(묶음)의 수 */
    private static final int VERSE_1_TOTAL_SEGMENTS = 6;
    /** 게임 전체를 구성하는 총 세그먼트의 수 (1절 + 2절) */
    private static final int GAME_TOTAL_SEGMENTS = 12;
    /** 1절 마지막 요청 시, AI 응답을 기다리는 최대 시간 (초) */
    private static final int MAX_WAIT_SECONDS_FOR_LEVEL_DECISION = 5;
    /** Redis 세션 만료 시간 (분) */
    private static final int SESSION_TIMEOUT_MINUTES = 30;

    // --- 의존성 주입 ---
    private final ScoringStrategyFactory scoringStrategyFactory;
    private final UserRepository userRepository;
    private final SongRepository songRepository;
    private final SongBeatRepository songBeatRepository;
    private final SongLyricsRepository songLyricsRepository;
    private final SongChoreographyRepository songChoreographyRepository;
    private final RedisTemplate<String, GameState> gameStateRedisTemplate;
    private final WebClient webClient;
    private final GameResultRepository gameResultRepository;
    private final com.heungbuja.s3.service.MediaUrlService mediaUrlService;
    // (추가) private final GameResultRepository gameResultRepository; // MySQL 결과 저장을 위한 Repository

    /**
     * 1. 게임 시작 로직
     */
    @Transactional(readOnly = true)
    public GameStartResponse startGame(GameStartRequest request) {
        // 1-1. User 정보 검증
        User user = userRepository.findById(request.getUserId())
                .orElseThrow(() -> new CustomException(ErrorCode.USER_NOT_FOUND));

        if (!user.getIsActive()) {
            throw new CustomException(ErrorCode.USER_NOT_ACTIVE);
        }

        // 1-2. MySQL에서 노래 기본 정보 조회
        Song song = songRepository.findById(request.getSongId())
                .orElseThrow(() -> new CustomException(ErrorCode.SONG_NOT_FOUND));
        String songTitle = song.getTitle();

        SongBeat beatInfo;
        SongLyrics lyricsInfo;
        SongChoreography choreographyInfo;

        // 1-3. MongoDB에서 노래 상세 메타데이터 조회 (병렬 처리로 성능 향상 가능)
        try {
            // 3개의 조회 작업을 각각 Mono로 감싸서 비동기 실행 준비
            Mono<SongBeat> beatInfoMono = Mono.fromCallable(() ->
                    songBeatRepository.findByAudioTitle(songTitle)
                            .orElseThrow(() -> new RuntimeException("Beat info not found"))
            );

            Mono<SongLyrics> lyricsInfoMono = Mono.fromCallable(() ->
                    songLyricsRepository.findByTitle(songTitle)
                            .orElseThrow(() -> new RuntimeException("Lyrics info not found"))
            );

            Mono<SongChoreography> choreographyInfoMono = Mono.fromCallable(() ->
                    songChoreographyRepository.findBySong(songTitle)
                            .orElseThrow(() -> new RuntimeException("Choreography info not found"))
            );

            // Mono.zip: 3개의 작업이 모두 끝날 때까지 기다렸다가, 결과를 튜플(Tuple) 형태로 한번에 받음
            var resultTuple = Mono.zip(beatInfoMono, lyricsInfoMono, choreographyInfoMono).block();

            // 튜플에서 각 결과를 꺼내어 변수에 할당
            beatInfo = resultTuple.getT1();
            lyricsInfo = resultTuple.getT2();
            choreographyInfo = resultTuple.getT3();

        } catch (Exception e) {
            // Mono.zip 실행 중 orElseThrow가 호출되어 RuntimeException이 발생하면 catch 블록으로 들어옴
            log.error("게임 시작에 필요한 메타데이터 조회 실패. songTitle={}", songTitle, e);
            // MongoDB에서 하나라도 데이터를 못 찾으면 GAME_METADATA_NOT_FOUND 에러 발생
            throw new CustomException(ErrorCode.GAME_METADATA_NOT_FOUND);
        }

        // 1-4. 게임 세션 ID 생성 및 Redis에 초기 상태 저장
        String sessionId = UUID.randomUUID().toString();
        GameState initialGameState = GameState.initial(sessionId, user.getId(), song.getId());
        gameStateRedisTemplate.opsForValue().set(sessionId, initialGameState, Duration.ofMinutes(30)); // 30분 후 세션 만료
        log.info("새로운 게임 세션 시작: userId={}, sessionId={}", user.getId(), sessionId);

        // 1-5. presigned URL 생성
        String presignedUrl = mediaUrlService.issueUrlById(song.getMedia().getId());

        // 1-6. 프론트엔드에 필요한 모든 정보를 담아 응답
        return GameStartResponse.builder()
                .sessionId(sessionId)
                .audioUrl(presignedUrl)
                .beatInfo(beatInfo)
                .lyricsInfo(lyricsInfo)
                .choreographyInfo(choreographyInfo)
                .build();
    }

    /**
     * 2. 프레임 묶음 분석 로직
     */
    public FrameBatchResponse analyzeFrameBatch(FrameBatchRequest request) {
        String sessionId = request.getSessionId();
        GameState currentGameState = getGameState(sessionId);

        // 요청 데이터 논리 검사 (오류 상태 처리)
        if (!currentGameState.getSongId().equals(request.getSongId())) {
            log.warn("세션ID와 노래ID가 일치하지 않습니다! sessionId={}, gameState.songId={}, request.songId={}",
                    sessionId, currentGameState.getSongId(), request.getSongId());
            throw new CustomException(ErrorCode.GAME_SESSION_INVALID);
        }

        // Python AI 서버로 분석 요청 (비동기)
        callAiServerAndProcessResult(currentGameState, request);

        // 1절 종료 시 레벨 결정
        if (request.getVerse() == 1 && request.isLastSegmentOfVerse1()) {
            // 모든 AI 분석이 완료될 때까지 최대 MAX_WAIT_SECONDS_FOR_LEVEL_DECISION (5)초간 기다림 (Polling 방식)
            for (int i = 0; i < MAX_WAIT_SECONDS_FOR_LEVEL_DECISION; i++) {
                GameState latestState = getGameState(sessionId);
                if (latestState.getBatchCompletedCount() >= VERSE_1_TOTAL_SEGMENTS) {
                    // 모든 결과가 도착했으면 레벨 결정
                    return decideLevelAndRespond(sessionId, latestState);
                }
                try {
                    Thread.sleep(1000); // 1초 대기 후 다시 확인
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    throw new CustomException(ErrorCode.INTERNAL_SERVER_ERROR, "레벨 결정 대기 중 오류 발생");
                }
            }
            // 설정된 시간을 기다려도 모든 결과가 오지 않은 경우
            log.warn("세션 {}의 1절 분석이 {}초 내에 완료되지 않았습니다. 현재까지의 점수로 레벨을 결정합니다.",
                    sessionId, MAX_WAIT_SECONDS_FOR_LEVEL_DECISION);
            return decideLevelAndRespond(sessionId, getGameState(sessionId));
        }


        // 4. 2절 종료 시 게임 종료 알림 (새로운 로직)
        if (request.isLastSegmentOfGame()) {
            log.info("세션 {}의 마지막 묶음 요청 접수. 게임 종료 처리 예정.", sessionId);
            // 이 요청 또한 AI 분석이 끝나야 GAME_OVER가 확정되지만, 우선은 즉시 응답
            return new FrameBatchResponse(GameStatus.GAME_OVER, null);
        }

        // 5. 일반적인 경우
        return new FrameBatchResponse(GameStatus.ACCEPTED, null);
    }

    /**
     * 레벨을 결정하고 응답을 생성하는 헬퍼 메소드
     */
    private FrameBatchResponse decideLevelAndRespond(String sessionId, GameState gameState) {
        double averageScore = calculateAverageScore(gameState.getVerse1Scores());
        int nextLevel = determineLevel(averageScore);

        gameState.setNextLevel(nextLevel);
        saveGameState(sessionId, gameState);
        log.info("세션 {}의 1절 종료. 평균 점수: {}, 다음 레벨: {}", sessionId, averageScore, nextLevel);

        return new FrameBatchResponse(GameStatus.LEVEL_DECIDED, nextLevel);
    }


    /**
     * 3. Python AI 서버 호출 및 결과 처리
     */
    private void callAiServerAndProcessResult(GameState gameState, FrameBatchRequest request) {
        String sessionId = gameState.getSessionId();

        // webClient.post()
        //         .uri("http://motion-server:8000/analyze")
        //         .bodyValue(request)
        //         .retrieve()
        //         .bodyToMono(AiResponse.class)
        //         .subscribe(
        //             // --- 1. 성공 시 콜백 ---
        //             aiResponse -> {
        //                 log.info("세션 {}의 분석 결과 도착: {}", sessionId, aiResponse.getActionCodes());
        //
        //                 ScoringStrategy strategy = scoringStrategyFactory.createStrategy(request.getVerse(), request.getDifficulty());
        //                 List<Integer> correctActionCodes = findCorrectActionsFromDB(gameState.getSongId(), request.getVerse(), request.getSegmentIndex());
        //                 double score = strategy.grade(aiResponse.getActionCodes(), correctActionCodes);
        //
        //                 // 성공했으므로, 계산된 점수로 상태 업데이트
        //                 updateGameState(sessionId, request.getVerse(), score);
        //             },
        //
        //             // --- ▼ 2. 실패 시 콜백 (핵심 수정) ▼ ---
        //             error -> {
        //                 log.error("AI 서버 호출 또는 분석 실패 (세션 {}): {}", sessionId, error.getMessage());
        //
        //                 // AI 분석에 실패했으므로, 해당 묶음의 점수를 0점으로 처리
        //                 updateGameState(sessionId, request.getVerse(), 0.0);
        //             }
        //         );

        log.info("세션 {}의 {}절-{}번째 이미지 묶음을 AI 서버로 전송했습니다.",
                sessionId, request.getVerse(), request.getSegmentIndex());
    }

    /**
     * Redis의 GameState를 업데이트하는 헬퍼 메소드
     * @param sessionId 현재 세션 ID
     * @param verse 현재 절
     * @param score 이번 묶음의 점수 (성공 시 계산된 점수, 실패 시 0.0)
     */
    private void updateGameState(String sessionId, int verse, double score) {
        // (주의) 이 부분은 여전히 동시성 문제에 취약합니다.
        // 나중에 Redis의 원자적 연산을 사용하여 개선해야 합니다.
        GameState currentState = getGameState(sessionId);

        if (verse == 1) {
            currentState.getVerse1Scores().add(score);
        } else {
            currentState.getVerse2Scores().add(score);
        }

        // 성공하든 실패하든, "하나의 묶음에 대한 처리가 끝났으므로" 카운터를 1 증가시킴
        currentState.setBatchCompletedCount(currentState.getBatchCompletedCount() + 1);

        saveGameState(sessionId, currentState);
        log.info("세션 {}의 게임 상태 업데이트 완료. 점수: {}, 완료된 묶음 수: {}",
                sessionId, score, currentState.getBatchCompletedCount());
    }

    /**
     * 정답지 조회 메소드 (뼈대)
     * TODO: 이 메소드의 내용을 실제 MongoDB 조회 로직으로 채워야 합니다.
     */
    private List<Integer> findCorrectActionsFromDB(Long songId, int verse, int segmentIndex) {
        // 1. songId로 SongChoreography 정보를 찾습니다.
        // 2. verse와 segmentIndex에 해당하는 안무 블록(block)을 찾습니다.
        // 3. 해당 블록에 미리 정의된 '정답 동작 코드 리스트'를 반환합니다.

        // 아래는 임시 하드코딩된 정답지입니다.
        log.warn("정답지 조회 로직이 구현되지 않았습니다. 임시 정답지를 사용합니다.");
        return List.of(0, 0, 1, 1, 0, 0);
    }

    /**
     * 게임 종료 및 결과 저장
     */
    @Transactional // DB에 데이터를 쓰는 작업이므로 @Transactional 어노테이션 추가
    public void endGame(String sessionId) {
        // Redis에서 최종 게임 상태를 가져옵니다.
        GameState finalGameState = getGameState(sessionId);
        if (finalGameState == null) {
            log.warn("이미 처리되었거나 존재하지 않는 세션 ID로 종료 요청: {}", sessionId);
            return; // 이미 삭제되었으면 아무것도 하지 않음
        }

        // 1. 결과 계산
        double verse1Avg = calculateAverageScore(finalGameState.getVerse1Scores());
        double verse2Avg = calculateAverageScore(finalGameState.getVerse2Scores());

        // 2. GameResult 엔티티 생성을 위해 User와 Song 엔티티를 DB에서 조회
        User user = userRepository.findById(finalGameState.getUserId())
                .orElseThrow(() -> new CustomException(ErrorCode.USER_NOT_FOUND, "게임 결과 저장 중 사용자를 찾을 수 없습니다."));
        Song song = songRepository.findById(finalGameState.getSongId())
                .orElseThrow(() -> new CustomException(ErrorCode.SONG_NOT_FOUND, "게임 결과 저장 중 노래를 찾을 수 없습니다."));

        // 3. GameResult 엔티티 객체 생성
        GameResult gameResult = GameResult.builder()
                .user(user)
                .song(song)
                .verse1AvgScore(verse1Avg)
                .verse2AvgScore(verse2Avg)
                .finalLevel(finalGameState.getNextLevel())
                .build();

        // 4. MySQL에 최종 결과 저장
        gameResultRepository.save(gameResult);
        log.info("세션 {}의 게임 결과 저장 완료. User: {}, Song: {}", sessionId, user.getId(), song.getId());

        // 5. Redis에 저장된 임시 데이터 삭제
        gameStateRedisTemplate.delete(sessionId);
        log.info("세션 {}의 Redis 데이터 삭제 완료.", sessionId);
    }


    // --- Helper 메소드들 ---
    private GameState getGameState(String sessionId) {
        // 1. Redis 조회 결과를 'gameState'라는 변수에 먼저 저장합니다.
        GameState gameState = gameStateRedisTemplate.opsForValue().get(sessionId);

        // 2. 변수가 null인지 확인합니다.
        if (gameState == null) {
            // 3. null이면 예외를 발생시키고 메소드를 종료합니다.
            throw new CustomException(ErrorCode.GAME_SESSION_NOT_FOUND);
        }

        // 4. null이 아니면, 변수에 담긴 값을 반환합니다.
        return gameState;
    }

    private void saveGameState(String sessionId, GameState gameState) {
        gameStateRedisTemplate.opsForValue().set(sessionId, gameState, Duration.ofMinutes(30));
    }

    private double calculateAverageScore(List<Double> scores) {
        if (scores == null || scores.isEmpty()) {
            return 0.0;
        }
        return scores.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
    }

    // 점수 기반 레벨 결정 로직 분리
    private int determineLevel(double averageScore) {
        if (averageScore >= 80) {
            return 3;
        } else if (averageScore >= 60) {
            return 2;
        }
        return 1;
    }

    // AI 서버의 응답을 받을 내부 DTO 클래스
    @Getter
    private static class AiResponse {
        private List<Integer> actionCodes;
    }
}