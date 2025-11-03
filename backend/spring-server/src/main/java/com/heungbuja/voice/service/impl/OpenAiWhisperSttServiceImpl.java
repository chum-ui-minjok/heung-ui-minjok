package com.heungbuja.voice.service.impl;

import com.heungbuja.common.exception.CustomException;
import com.heungbuja.common.exception.ErrorCode;
import com.heungbuja.voice.service.SttService;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Primary;
import org.springframework.context.annotation.Profile;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.*;
import org.springframework.stereotype.Service;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.multipart.MultipartFile;

import java.util.Map;

/**
 * OpenAI Whisper API를 사용한 STT 서비스 구현
 * GMS SSAFY 프록시를 통해 호출
 */
@Slf4j
@Service
@Primary // 이 구현체를 우선 사용
@Profile({"prod", "!local"}) // local 제외한 모든 프로파일
public class OpenAiWhisperSttServiceImpl implements SttService {

    @Value("${openai.gms.api-key}")
    private String gmsApiKey;

    @Value("${openai.gms.stt.url:https://gms.ssafy.io/gmsapi/api.openai.com/v1/audio/transcriptions}")
    private String sttApiUrl;

    private final RestTemplate restTemplate;

    public OpenAiWhisperSttServiceImpl() {
        this.restTemplate = new RestTemplate();
    }

    @Override
    public String transcribe(MultipartFile audioFile) {
        if (!isSupportedFormat(audioFile)) {
            throw new CustomException(ErrorCode.INVALID_INPUT_VALUE,
                    "지원하지 않는 오디오 포맷입니다");
        }

        log.info("OpenAI Whisper STT 시작: 파일명={}, 크기={} bytes",
                audioFile.getOriginalFilename(), audioFile.getSize());

        try {
            long startTime = System.currentTimeMillis();

            // Multipart 요청 생성
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.MULTIPART_FORM_DATA);
            headers.setBearerAuth(gmsApiKey);

            MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
            body.add("file", new ByteArrayResource(audioFile.getBytes()) {
                @Override
                public String getFilename() {
                    return audioFile.getOriginalFilename();
                }
            });
            body.add("model", "whisper-1");
            body.add("language", "ko"); // 한국어 우선 인식

            HttpEntity<MultiValueMap<String, Object>> requestEntity = new HttpEntity<>(body, headers);

            // API 호출
            ResponseEntity<Map> response = restTemplate.exchange(
                    sttApiUrl,
                    HttpMethod.POST,
                    requestEntity,
                    Map.class
            );

            long endTime = System.currentTimeMillis();
            log.info("OpenAI Whisper STT 완료: 소요 시간={}ms", endTime - startTime);

            if (response.getStatusCode() == HttpStatus.OK && response.getBody() != null) {
                String text = (String) response.getBody().get("text");
                log.info("STT 결과: '{}'", text);
                return text.trim();
            } else {
                throw new CustomException(ErrorCode.INTERNAL_SERVER_ERROR,
                        "STT API 응답 오류");
            }

        } catch (CustomException e) {
            throw e;
        } catch (Exception e) {
            log.error("OpenAI Whisper STT 실패", e);
            throw new CustomException(ErrorCode.INTERNAL_SERVER_ERROR,
                    "음성 인식에 실패했습니다: " + e.getMessage());
        }
    }
}
