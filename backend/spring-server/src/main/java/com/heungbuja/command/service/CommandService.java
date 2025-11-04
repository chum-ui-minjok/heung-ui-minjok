package com.heungbuja.command.service;

import com.heungbuja.command.dto.CommandRequest;
import com.heungbuja.command.dto.CommandResponse;
import com.heungbuja.command.dto.IntentResult;
import com.heungbuja.common.exception.CustomException;
import com.heungbuja.common.exception.ErrorCode;
import com.heungbuja.emergency.dto.EmergencyRequest;
import com.heungbuja.emergency.service.EmergencyService;
import com.heungbuja.song.dto.SongInfoDto;
import com.heungbuja.song.enums.PlaybackMode;
import com.heungbuja.song.service.ListeningHistoryService;
import com.heungbuja.song.entity.Song;
import com.heungbuja.song.service.SongService;
import com.heungbuja.user.entity.User;
import com.heungbuja.user.service.UserService;
import com.heungbuja.voice.entity.VoiceCommand;
import com.heungbuja.voice.enums.Intent;
import com.heungbuja.voice.repository.VoiceCommandRepository;
import com.heungbuja.voice.service.TtsService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

/**
 * 통합 명령 처리 서비스
 * 의도 분석 → 적절한 서비스 호출 → 응답 생성
 *
 * 주의: 프론트엔드가 음악 재생을 관리하므로, 백엔드는:
 * - 노래 정보만 전달 (audioUrl)
 * - 청취 이력만 기록
 * - 상태 관리 없음
 */
@Slf4j
@Service
@RequiredArgsConstructor
@Transactional(readOnly = true)
public class CommandService {

    // 핵심 서비스 (인터페이스 기반 - 느슨한 결합)
    private final IntentClassifier intentClassifier;
    private final TtsService ttsService;

    // 도메인 서비스
    private final UserService userService;
    private final SongService songService;
    private final ListeningHistoryService listeningHistoryService;
    private final EmergencyService emergencyService;

    // 기타
    private final VoiceCommandRepository voiceCommandRepository;
    private final ResponseGenerator responseGenerator;

    /**
     * 텍스트 명령어 처리 (통합 엔드포인트)
     *
     * noRollbackFor: CustomException을 Controller에서 catch하므로
     * 트랜잭션을 롤백하지 않도록 설정
     */
    @Transactional(noRollbackFor = {CustomException.class, Exception.class})
    public CommandResponse processTextCommand(CommandRequest request) {
        User user = userService.findById(request.getUserId());
        String text = request.getText().trim();

        log.info("명령 처리 시작: userId={}, text='{}'", user.getId(), text);

        try {
            // 1. 의도 분석 (IntentClassifier - 교체 가능)
            IntentResult intentResult = intentClassifier.classify(text);
            Intent intent = intentResult.getIntent();

            // 원본 텍스트 저장 (Emergency 등에서 사용)
            intentResult = IntentResult.builder()
                    .intent(intentResult.getIntent())
                    .entities(intentResult.getEntities())
                    .confidence(intentResult.getConfidence())
                    .rawText(text)
                    .build();

            log.info("의도 분석 완료: intent={}, classifier={}", intent, intentClassifier.getClassifierType());

            // 2. 음성 명령 로그 저장
            saveVoiceCommand(user, text, intent);

            // 3. 의도에 따른 처리
            CommandResponse response = executeIntent(user, intentResult);

            log.info("명령 처리 완료: userId={}, intent={}, success={}", user.getId(), intent, response.isSuccess());

            return response;

        } catch (CustomException e) {
            // CustomException은 그대로 던져서 Controller에서 적절한 HTTP 상태로 변환
            throw e;
        } catch (Exception e) {
            // 예상치 못한 에러는 COMMAND_EXECUTION_FAILED로 변환
            log.error("명령 처리 실패: userId={}, text='{}'", user.getId(), text, e);
            throw new CustomException(ErrorCode.COMMAND_EXECUTION_FAILED, "명령 처리 중 오류가 발생했습니다");
        }
    }

    /**
     * 의도에 따른 실행
     */
    private CommandResponse executeIntent(User user, IntentResult intentResult) {
        Intent intent = intentResult.getIntent();

        return switch (intent) {
            // 노래 검색
            case SELECT_BY_ARTIST -> handleSearchByArtist(user, intentResult);
            case SELECT_BY_TITLE -> handleSearchByTitle(user, intentResult);
            case SELECT_BY_ARTIST_TITLE -> handleSearchByArtistAndTitle(user, intentResult);

            // 재생 제어 (프론트가 관리하므로 TTS 응답만)
            case MUSIC_PAUSE -> handleSimpleResponse(Intent.MUSIC_PAUSE);
            case MUSIC_RESUME -> handleSimpleResponse(Intent.MUSIC_RESUME);
            case MUSIC_NEXT -> handleSimpleResponse(Intent.MUSIC_NEXT);
            case MUSIC_STOP -> handleSimpleResponse(Intent.MUSIC_STOP);

            // 모드 관련 (프론트가 관리하므로 TTS 응답만)
            case MODE_LISTENING_START -> handleSimpleResponse(Intent.MODE_LISTENING_START);
            case MODE_EXERCISE_START -> handleSimpleResponse(Intent.MODE_EXERCISE_START);
            case MODE_SWITCH_TO_LISTENING -> handleSimpleResponse(Intent.MODE_SWITCH_TO_LISTENING);
            case MODE_SWITCH_TO_EXERCISE -> handleSimpleResponse(Intent.MODE_SWITCH_TO_EXERCISE);

            // 응급 상황
            case EMERGENCY -> handleEmergency(user, intentResult);
            case EMERGENCY_CANCEL -> handleEmergencyCancel(user);

            // 인식 불가
            case UNKNOWN -> handleUnknown();

            default -> handleError();
        };
    }

    /**
     * 가수명으로 노래 검색
     */
    private CommandResponse handleSearchByArtist(User user, IntentResult intentResult) {
        String query = intentResult.getEntity("query");
        if (query == null) query = intentResult.getEntity("artist");

        Song song = songService.searchByArtist(query);

        // 청취 이력 기록
        listeningHistoryService.recordListening(user, song, PlaybackMode.LISTENING);

        String responseText = responseGenerator.generateResponse(Intent.SELECT_BY_ARTIST, song.getArtist(), song.getTitle());
        String ttsUrl = ttsService.synthesize(responseText);

        return CommandResponse.withSong(
                Intent.SELECT_BY_ARTIST,
                responseText,
                "/commands/tts/" + ttsUrl,
                SongInfoDto.from(song, PlaybackMode.LISTENING)
        );
    }

    /**
     * 제목으로 노래 검색
     */
    private CommandResponse handleSearchByTitle(User user, IntentResult intentResult) {
        String title = intentResult.getEntity("title");
        Song song = songService.searchByTitle(title);

        // 청취 이력 기록
        listeningHistoryService.recordListening(user, song, PlaybackMode.LISTENING);

        String responseText = responseGenerator.generateResponse(Intent.SELECT_BY_TITLE, song.getArtist(), song.getTitle());
        String ttsUrl = ttsService.synthesize(responseText);

        return CommandResponse.withSong(
                Intent.SELECT_BY_TITLE,
                responseText,
                "/commands/tts/" + ttsUrl,
                SongInfoDto.from(song, PlaybackMode.LISTENING)
        );
    }

    /**
     * 가수+제목으로 노래 검색
     */
    private CommandResponse handleSearchByArtistAndTitle(User user, IntentResult intentResult) {
        String artist = intentResult.getEntity("artist");
        String title = intentResult.getEntity("title");

        Song song = songService.searchByArtistAndTitle(artist, title);

        // 청취 이력 기록
        listeningHistoryService.recordListening(user, song, PlaybackMode.LISTENING);

        String responseText = responseGenerator.generateResponse(Intent.SELECT_BY_ARTIST_TITLE, song.getArtist(), song.getTitle());
        String ttsUrl = ttsService.synthesize(responseText);

        return CommandResponse.withSong(
                Intent.SELECT_BY_ARTIST_TITLE,
                responseText,
                "/commands/tts/" + ttsUrl,
                SongInfoDto.from(song, PlaybackMode.LISTENING)
        );
    }

    /**
     * 단순 응답 (TTS만)
     * 프론트가 재생을 관리하므로 백엔드는 음성 안내만
     */
    private CommandResponse handleSimpleResponse(Intent intent) {
        String responseText = responseGenerator.generateResponse(intent);
        String ttsUrl = ttsService.synthesize(responseText);

        return CommandResponse.success(intent, responseText, "/commands/tts/" + ttsUrl);
    }

    /**
     * 응급 상황 처리
     */
    private CommandResponse handleEmergency(User user, IntentResult intentResult) {
        // 응급 신고 생성
        EmergencyRequest emergencyRequest = EmergencyRequest.builder()
                .userId(user.getId())
                .triggerWord(intentResult.getEntity("keyword"))
                .fullText(intentResult.getRawText())  // 전체 발화 텍스트
                .build();

        emergencyService.detectEmergency(emergencyRequest);

        String responseText = responseGenerator.generateResponse(Intent.EMERGENCY);
        String ttsUrl = ttsService.synthesize(responseText, "urgent"); // 긴급 음성 타입

        return CommandResponse.success(Intent.EMERGENCY, responseText, "/commands/tts/" + ttsUrl);
    }

    /**
     * 응급 상황 취소 처리
     */
    private CommandResponse handleEmergencyCancel(User user) {
        emergencyService.cancelRecentReport(user.getId());

        String responseText = responseGenerator.generateResponse(Intent.EMERGENCY_CANCEL);
        String ttsUrl = ttsService.synthesize(responseText);

        return CommandResponse.success(Intent.EMERGENCY_CANCEL, responseText, "/commands/tts/" + ttsUrl);
    }

    /**
     * 인식 불가
     */
    private CommandResponse handleUnknown() {
        String responseText = responseGenerator.generateResponse(Intent.UNKNOWN);
        String ttsUrl = ttsService.synthesize(responseText);

        return CommandResponse.failure(Intent.UNKNOWN, responseText, "/commands/tts/" + ttsUrl);
    }

    /**
     * 에러 처리
     */
    private CommandResponse handleError() {
        String errorMsg = responseGenerator.errorMessage();
        String ttsUrl = ttsService.synthesize(errorMsg);

        return CommandResponse.failure(Intent.UNKNOWN, errorMsg, "/commands/tts/" + ttsUrl);
    }

    /**
     * 음성 명령 로그 저장
     */
    private void saveVoiceCommand(User user, String text, Intent intent) {
        VoiceCommand command = VoiceCommand.builder()
                .user(user)
                .rawText(text)
                .intent(intent.name())
                .build();

        voiceCommandRepository.save(command);
    }
}
