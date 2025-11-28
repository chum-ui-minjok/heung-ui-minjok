package com.heungbuja.music.service;

import com.heungbuja.common.exception.CustomException;
import com.heungbuja.common.exception.ErrorCode;
import com.heungbuja.context.service.ConversationContextService;
import com.heungbuja.music.dto.MusicListResponse;
import com.heungbuja.music.dto.MusicPlayResponse;
import com.heungbuja.s3.service.MediaUrlService;
import com.heungbuja.session.service.SessionStateService;
import com.heungbuja.session.state.ActivityState;
import com.heungbuja.song.entity.Song;
import com.heungbuja.song.enums.PlaybackMode;
import com.heungbuja.song.repository.jpa.SongRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;
import java.util.stream.Collectors;

/**
 * 음악 듣기 모드 서비스
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class MusicService {

    private final SongRepository songRepository;
    private final MediaUrlService mediaUrlService;
    private final ConversationContextService conversationContextService;
    private final SessionStateService sessionStateService;

    /**
     * 음악 목록 조회 (최대 limit개)
     */
    @Transactional(readOnly = true)
    public List<MusicListResponse> getMusicList(int limit) {
        List<Song> songs = songRepository.findAll();
        return songs.stream()
                .limit(limit)
                .map(MusicListResponse::from)
                .collect(Collectors.toList());
    }

    /**
     * 음악 재생 - presigned URL 발급 + 컨텍스트/세션 상태 업데이트
     */
    @Transactional
    public MusicPlayResponse playSong(Long userId, Long songId) {
        Song song = songRepository.findById(songId)
                .orElseThrow(() -> new CustomException(ErrorCode.SONG_NOT_FOUND,
                        "노래를 찾을 수 없습니다 (ID: " + songId + ")"));

        // 기존 활동 인터럽트 처리 (게임 중이면 중단)
        handleActivityInterrupt(userId, "음악 재생");

        // Redis: 컨텍스트 업데이트 (모드 + 현재 곡)
        conversationContextService.changeMode(userId, PlaybackMode.LISTENING);
        conversationContextService.setCurrentSong(userId, songId);

        // Redis: 활동 상태 업데이트
        sessionStateService.setCurrentActivity(userId, ActivityState.music(String.valueOf(songId)));

        // Media에서 presigned URL 발급
        String audioUrl = mediaUrlService.issueUrlById(song.getMedia().getId());

        log.info("음악 재생: userId={}, songId={}, title={}, artist={}",
                userId, songId, song.getTitle(), song.getArtist());

        return MusicPlayResponse.success(
                song.getId(),
                song.getTitle(),
                song.getArtist(),
                audioUrl
        );
    }

    /**
     * 음악 종료 + 세션 상태 정리
     */
    public void stopMusic(Long userId) {
        // Redis: 활동 상태 삭제 (IDLE로 전환)
        sessionStateService.clearActivity(userId);

        // Redis: 모드를 HOME으로 변경
        conversationContextService.changeMode(userId, PlaybackMode.HOME);

        log.info("음악 종료: userId={}", userId);
    }

    /**
     * 현재 활동 인터럽트 처리
     */
    private void handleActivityInterrupt(Long userId, String newActivity) {
        ActivityState currentActivity = sessionStateService.getCurrentActivity(userId);

        if (currentActivity.getType() == com.heungbuja.session.enums.ActivityType.IDLE) {
            return;
        }

        log.info("활동 인터럽트: userId={}, 현재={}, 새활동={}",
                userId, currentActivity.getType(), newActivity);

        switch (currentActivity.getType()) {
            case GAME:
                String sessionId = currentActivity.getSessionId();
                if (sessionStateService.trySetInterrupt(sessionId, newActivity)) {
                    log.info("게임 중단 플래그 설정: sessionId={}, reason={}", sessionId, newActivity);
                }
                break;

            case MUSIC:
                sessionStateService.clearActivity(userId);
                log.info("기존 음악 중단: userId={}", userId);
                break;

            default:
                break;
        }
    }
}
