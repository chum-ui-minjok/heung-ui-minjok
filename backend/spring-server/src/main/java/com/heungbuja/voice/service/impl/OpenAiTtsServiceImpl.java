package com.heungbuja.voice.service.impl;

import com.heungbuja.common.exception.CustomException;
import com.heungbuja.common.exception.ErrorCode;
import com.heungbuja.voice.service.TtsService;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Primary;
import org.springframework.context.annotation.Profile;
import org.springframework.http.*;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;
import java.util.UUID;

/**
 * OpenAI TTS API를 사용한 음성 합성 서비스
 * GMS SSAFY 프록시를 통해 호출
 */
@Slf4j
@Service
@Primary // 이 구현체를 우선 사용
@Profile({"prod", "!local"}) // local 제외한 모든 프로파일
public class OpenAiTtsServiceImpl implements TtsService {

    @Value("${openai.gms.api-key}")
    private String gmsApiKey;

    @Value("${openai.gms.tts.url:https://gms.ssafy.io/gmsapi/api.openai.com/v1/audio/speech}")
    private String ttsApiUrl;

    @Value("${tts.storage.path:./tts-files}")
    private String storagePath;

    private final RestTemplate restTemplate;

    public OpenAiTtsServiceImpl() {
        this.restTemplate = new RestTemplate();
    }

    @Override
    public String synthesize(String text, String voiceType) {
        log.info("OpenAI TTS 시작: text='{}', voiceType='{}'", text, voiceType);

        try {
            long startTime = System.currentTimeMillis();

            // 저장 디렉토리 생성
            Path storageDir = Paths.get(storagePath);
            if (!Files.exists(storageDir)) {
                Files.createDirectories(storageDir);
            }

            // 음성 타입 매핑 (urgent → alloy, default → nova 등)
            String voice = mapVoiceType(voiceType);

            // 요청 생성
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);
            headers.setBearerAuth(gmsApiKey);

            Map<String, Object> requestBody = new HashMap<>();
            requestBody.put("model", "gpt-4o-mini-tts");
            requestBody.put("input", text);
            requestBody.put("voice", voice);
            requestBody.put("response_format", "mp3");

            HttpEntity<Map<String, Object>> requestEntity = new HttpEntity<>(requestBody, headers);

            // API 호출
            ResponseEntity<byte[]> response = restTemplate.exchange(
                    ttsApiUrl,
                    HttpMethod.POST,
                    requestEntity,
                    byte[].class
            );

            long endTime = System.currentTimeMillis();
            log.info("OpenAI TTS 완료: 소요 시간={}ms", endTime - startTime);

            if (response.getStatusCode() == HttpStatus.OK && response.getBody() != null) {
                // 파일 저장
                String fileId = UUID.randomUUID().toString();
                String fileName = fileId + ".mp3";
                Path filePath = storageDir.resolve(fileName);

                Files.write(filePath, response.getBody());

                log.info("TTS 파일 저장 완료: fileId={}, 크기={} bytes",
                        fileId, response.getBody().length);

                return fileId;
            } else {
                throw new CustomException(ErrorCode.INTERNAL_SERVER_ERROR,
                        "TTS API 응답 오류");
            }

        } catch (CustomException e) {
            throw e;
        } catch (Exception e) {
            log.error("OpenAI TTS 실패", e);
            throw new CustomException(ErrorCode.INTERNAL_SERVER_ERROR,
                    "음성 합성에 실패했습니다: " + e.getMessage());
        }
    }

    @Override
    public byte[] getAudioFile(String fileId) {
        try {
            Path filePath = Paths.get(storagePath, fileId + ".mp3");

            if (!Files.exists(filePath)) {
                throw new CustomException(ErrorCode.INVALID_INPUT_VALUE,
                        "TTS 파일을 찾을 수 없습니다");
            }

            return Files.readAllBytes(filePath);

        } catch (IOException e) {
            log.error("TTS 파일 읽기 실패: fileId={}", fileId, e);
            throw new CustomException(ErrorCode.INTERNAL_SERVER_ERROR,
                    "TTS 파일을 읽을 수 없습니다");
        }
    }

    /**
     * 음성 타입 매핑
     * OpenAI TTS는 alloy, echo, fable, onyx, nova, shimmer 지원
     */
    private String mapVoiceType(String voiceType) {
        if (voiceType == null || voiceType.equals("default")) {
            return "nova"; // 기본 음성 (여성, 따뜻함)
        }

        return switch (voiceType.toLowerCase()) {
            case "urgent", "emergency" -> "alloy"; // 긴급: 중성적이고 명확한 음성
            case "calm", "gentle" -> "shimmer"; // 부드럽고 차분한 음성
            case "energetic" -> "echo"; // 활기찬 음성
            case "male" -> "onyx"; // 남성 음성
            case "female" -> "nova"; // 여성 음성
            default -> "nova";
        };
    }
}
