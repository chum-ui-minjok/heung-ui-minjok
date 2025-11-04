package com.heungbuja.voice.enums;

/**
 * 음성 명령 의도 (Intent)
 */
public enum Intent {
    // 음악 재생 관련
    SELECT_BY_ARTIST("가수명으로 노래 검색"),
    SELECT_BY_TITLE("제목으로 노래 검색"),
    SELECT_BY_ARTIST_TITLE("가수+제목으로 노래 검색"),

    // 재생 제어
    MUSIC_PAUSE("일시정지"),
    MUSIC_RESUME("재생 재개"),
    MUSIC_NEXT("다음 곡"),
    MUSIC_STOP("재생 종료"),

    // 모드 관련
    MODE_LISTENING_START("감상 모드 시작"),
    MODE_EXERCISE_START("체조 모드 시작"),
    MODE_SWITCH_TO_LISTENING("감상 모드로 전환"),
    MODE_SWITCH_TO_EXERCISE("체조 모드로 전환"),

    // 응급 상황
    EMERGENCY("응급 상황 감지"),
    EMERGENCY_CANCEL("응급 상황 취소"),

    // 기타
    UNKNOWN("인식 불가");

    private final String description;

    Intent(String description) {
        this.description = description;
    }

    public String getDescription() {
        return description;
    }
}
