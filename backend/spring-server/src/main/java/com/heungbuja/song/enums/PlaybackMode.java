package com.heungbuja.song.enums;

/**
 * 재생 모드
 */
public enum PlaybackMode {
    LISTENING("감상 모드"),
    EXERCISE("체조 모드");

    private final String description;

    PlaybackMode(String description) {
        this.description = description;
    }

    public String getDescription() {
        return description;
    }
}
