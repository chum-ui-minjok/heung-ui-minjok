package com.heungbuja.command.controller;

import com.heungbuja.command.dto.CommandRequest;
import com.heungbuja.command.dto.CommandResponse;
import com.heungbuja.command.service.CommandService;
import com.heungbuja.voice.service.SttService;
import com.heungbuja.voice.service.TtsService;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

/**
 * 통합 음성 명령 컨트롤러
 */
@Slf4j
@RestController
@RequestMapping("/commands")
@RequiredArgsConstructor
public class CommandController {

    private final CommandService commandService;
    private final SttService sttService;
    private final TtsService ttsService;

    /**
     * 음성 파일로 명령 처리 (통합 엔드포인트)
     * 음성 업로드 → STT → Intent 분석 → 실행 → TTS 응답
     */
    @PostMapping(value = "/process", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<CommandResponse> processVoiceCommand(
            @RequestParam("userId") Long userId,
            @RequestParam("audioFile") MultipartFile audioFile) {

        log.info("음성 명령 처리 요청: userId={}, 파일크기={} bytes", userId, audioFile.getSize());

        try {
            // 1. STT: 음성 → 텍스트
            String transcribedText = sttService.transcribe(audioFile);
            log.info("STT 변환 완료: text='{}'", transcribedText);

            // 2. 텍스트 명령 처리
            CommandRequest request = CommandRequest.builder()
                    .userId(userId)
                    .text(transcribedText)
                    .build();

            CommandResponse response = commandService.processTextCommand(request);

            return ResponseEntity.ok(response);

        } catch (Exception e) {
            log.error("음성 명령 처리 실패", e);
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body(CommandResponse.failure(null, "처리 중 오류가 발생했습니다", null));
        }
    }

    /**
     * 텍스트 명령 직접 처리 (디버깅용)
     */
    @PostMapping("/text")
    public ResponseEntity<CommandResponse> processTextCommand(
            @Valid @RequestBody CommandRequest request) {

        log.info("텍스트 명령 처리 요청: userId={}, text='{}'", request.getUserId(), request.getText());

        CommandResponse response = commandService.processTextCommand(request);
        return ResponseEntity.ok(response);
    }

    /**
     * TTS 음성 파일 다운로드
     */
    @GetMapping("/tts/{fileId}")
    public ResponseEntity<byte[]> getTtsAudio(@PathVariable String fileId) {
        try {
            byte[] audioData = ttsService.getAudioFile(fileId);

            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.parseMediaType("audio/mpeg"));
            headers.setContentDispositionFormData("attachment", fileId + ".mp3");

            return new ResponseEntity<>(audioData, headers, HttpStatus.OK);

        } catch (Exception e) {
            log.error("TTS 파일 다운로드 실패: fileId={}", fileId, e);
            return ResponseEntity.status(HttpStatus.NOT_FOUND).build();
        }
    }
}
