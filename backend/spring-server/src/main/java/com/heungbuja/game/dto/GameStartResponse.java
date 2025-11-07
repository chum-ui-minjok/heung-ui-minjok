package com.heungbuja.game.dto;

import com.heungbuja.song.domain.SongLyrics;
import lombok.Builder;
import lombok.Getter;
import java.util.List;
import java.util.Map;

@Getter
@Builder
public class GameStartResponse {
    private String sessionId;
//    private String websocketUrl;
    private String audioUrl;
    private Map<String, String> videoUrls;

    /** 노래의 BPM (Beats Per Minute) */
    private double bpm;

    /** 노래 전체 길이 (초) */
    private double duration;

    /** 노래의 주요 섹션별 시작 시간 정보 */
    private SectionInfo sectionInfo;

    /** 가사 정보 (원본 JSON 그대로 전달) */
    private SongLyrics lyricsInfo;
}