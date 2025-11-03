package com.heungbuja.command.service.impl;

import com.heungbuja.command.dto.IntentResult;
import com.heungbuja.command.service.IntentClassifier;
import com.heungbuja.voice.enums.Intent;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.util.Arrays;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * 키워드 기반 의도 분석기
 * 간단하고 빠르지만, 복잡한 문장은 처리 어려움
 * 추후 RAG 또는 LLM 기반으로 교체 가능
 */
@Slf4j
@Component
public class KeywordBasedIntentClassifier implements IntentClassifier {

    // 응급 키워드
    private static final List<String> EMERGENCY_KEYWORDS = Arrays.asList(
            "도와줘", "도와주세요", "살려줘", "살려주세요", "아야", "아파", "쓰러졌어", "위험해"
    );

    // 재생 제어 키워드
    private static final List<String> PAUSE_KEYWORDS = Arrays.asList("잠깐", "멈춰", "정지", "일시정지");
    private static final List<String> RESUME_KEYWORDS = Arrays.asList("다시", "계속", "재생");
    private static final List<String> NEXT_KEYWORDS = Arrays.asList("다음", "건너뛰기", "스킵", "넘겨");
    private static final List<String> STOP_KEYWORDS = Arrays.asList("그만", "종료", "끝");

    // 모드 키워드
    private static final List<String> LISTENING_START_KEYWORDS = Arrays.asList(
            "노래 들려줘", "음악 틀어줘", "노래 듣고 싶어", "음악 듣고 싶어"
    );
    private static final List<String> EXERCISE_START_KEYWORDS = Arrays.asList(
            "체조하고 싶어", "운동할래", "체조할래", "같이 운동해줘"
    );
    private static final List<String> SWITCH_TO_LISTENING = Arrays.asList(
            "그냥 듣기만 할래", "감상으로 바꿔줘", "감상 모드"
    );
    private static final List<String> SWITCH_TO_EXERCISE = Arrays.asList(
            "체조로 바꿔줘", "운동 모드", "체조 모드"
    );

    @Override
    public IntentResult classify(String text) {
        String normalized = text.trim().toLowerCase();

        log.debug("의도 분석 시작: text='{}'", normalized);

        // 1. 응급 상황 (최우선)
        String matchedKeyword = findMatchedKeyword(normalized, EMERGENCY_KEYWORDS);
        if (matchedKeyword != null) {
            IntentResult result = IntentResult.builder()
                    .intent(Intent.EMERGENCY)
                    .confidence(1.0)
                    .build();
            result.addEntity("keyword", matchedKeyword);
            log.debug("응급 상황 감지: keyword='{}'", matchedKeyword);
            return result;
        }

        // 2. 재생 제어
        if (containsAny(normalized, PAUSE_KEYWORDS)) {
            return IntentResult.of(Intent.MUSIC_PAUSE);
        }
        if (containsAny(normalized, RESUME_KEYWORDS)) {
            return IntentResult.of(Intent.MUSIC_RESUME);
        }
        if (containsAny(normalized, NEXT_KEYWORDS)) {
            return IntentResult.of(Intent.MUSIC_NEXT);
        }
        if (containsAny(normalized, STOP_KEYWORDS)) {
            return IntentResult.of(Intent.MUSIC_STOP);
        }

        // 3. 모드 전환/시작
        if (containsAny(normalized, LISTENING_START_KEYWORDS)) {
            return IntentResult.of(Intent.MODE_LISTENING_START);
        }
        if (containsAny(normalized, EXERCISE_START_KEYWORDS)) {
            return IntentResult.of(Intent.MODE_EXERCISE_START);
        }
        if (containsAny(normalized, SWITCH_TO_LISTENING)) {
            return IntentResult.of(Intent.MODE_SWITCH_TO_LISTENING);
        }
        if (containsAny(normalized, SWITCH_TO_EXERCISE)) {
            return IntentResult.of(Intent.MODE_SWITCH_TO_EXERCISE);
        }

        // 4. 노래 검색 (가수 + 제목 패턴)
        IntentResult songIntent = detectSongSearchIntent(normalized);
        if (songIntent != null) {
            return songIntent;
        }

        // 5. 인식 불가
        return IntentResult.of(Intent.UNKNOWN);
    }

    /**
     * 노래 검색 의도 감지
     */
    private IntentResult detectSongSearchIntent(String text) {
        // "틀어줘", "들려줘", "재생" 등의 트리거 단어 제거
        String cleanText = text
                .replaceAll("(틀어줘|틀어|들려줘|들려|재생|노래|음악|해줘|해)", "")
                .replaceAll("\\s+", " ")
                .trim();

        if (cleanText.isEmpty()) {
            return null;
        }

        // 패턴 1: "가수의 제목" (예: "태진아의 사랑은 아무나 하나")
        Pattern pattern1 = Pattern.compile("(.+)의\\s*(.+)");
        Matcher matcher1 = pattern1.matcher(cleanText);
        if (matcher1.find()) {
            String artist = matcher1.group(1).trim();
            String title = matcher1.group(2).trim();

            IntentResult result = IntentResult.builder()
                    .intent(Intent.SELECT_BY_ARTIST_TITLE)
                    .confidence(0.9)
                    .build();
            result.addEntity("artist", artist);
            result.addEntity("title", title);

            log.debug("노래 검색 감지 (가수+제목): artist='{}', title='{}'", artist, title);
            return result;
        }

        // 패턴 2: "가수 제목" (예: "태진아 사랑은 아무나 하나")
        String[] words = cleanText.split("\\s+");
        if (words.length >= 2) {
            String artist = words[0];
            String title = String.join(" ", Arrays.copyOfRange(words, 1, words.length));

            IntentResult result = IntentResult.builder()
                    .intent(Intent.SELECT_BY_ARTIST_TITLE)
                    .confidence(0.7)
                    .build();
            result.addEntity("artist", artist);
            result.addEntity("title", title);

            log.debug("노래 검색 감지 (추정 가수+제목): artist='{}', title='{}'", artist, title);
            return result;
        }

        // 패턴 3: 단일 단어 (가수명 또는 제목)
        IntentResult result = IntentResult.builder()
                .intent(Intent.SELECT_BY_ARTIST)
                .confidence(0.5)
                .build();
        result.addEntity("query", cleanText);

        log.debug("노래 검색 감지 (일반 검색): query='{}'", cleanText);
        return result;
    }

    /**
     * 키워드 포함 여부 확인
     */
    private boolean containsAny(String text, List<String> keywords) {
        return keywords.stream().anyMatch(text::contains);
    }

    /**
     * 매칭된 키워드 찾기 (실제 매칭된 키워드 반환)
     */
    private String findMatchedKeyword(String text, List<String> keywords) {
        return keywords.stream()
                .filter(text::contains)
                .findFirst()
                .orElse(null);
    }

    @Override
    public String getClassifierType() {
        return "KEYWORD";
    }
}
