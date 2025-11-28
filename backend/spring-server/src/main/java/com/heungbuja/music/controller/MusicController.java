package com.heungbuja.music.controller;

import com.heungbuja.common.dto.ControlResponse;
import com.heungbuja.common.exception.CustomException;
import com.heungbuja.common.exception.ErrorCode;
import com.heungbuja.music.dto.MusicListResponse;
import com.heungbuja.music.dto.MusicPlayRequest;
import com.heungbuja.music.dto.MusicPlayResponse;
import com.heungbuja.music.service.MusicService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.web.bind.annotation.*;

import java.util.List;

/**
 * 음악 듣기 모드 컨트롤러 (클릭 기반)
 */
@Slf4j
@RestController
@RequestMapping("/api/music")
@RequiredArgsConstructor
public class MusicController {

    private final MusicService musicService;

    /**
     * 음악 목록 조회 (최대 5곡)
     */
    @GetMapping("/list")
    public ResponseEntity<List<MusicListResponse>> getMusicList() {
        List<MusicListResponse> musicList = musicService.getMusicList(5);
        return ResponseEntity.ok(musicList);
    }

    /**
     * 음악 재생
     */
    @PostMapping("/play")
    public ResponseEntity<MusicPlayResponse> playMusic(@RequestBody MusicPlayRequest request) {
        if (request.getSongId() == null) {
            throw new CustomException(ErrorCode.INVALID_INPUT_VALUE, "songId는 필수입니다");
        }
        Long userId = getCurrentUserId();
        MusicPlayResponse response = musicService.playSong(userId, request.getSongId());
        return ResponseEntity.ok(response);
    }

    /**
     * 음악 종료
     */
    @PostMapping("/stop")
    public ResponseEntity<ControlResponse> stopMusic() {
        Long userId = getCurrentUserId();
        musicService.stopMusic(userId);
        return ResponseEntity.ok(ControlResponse.success("음악이 종료되었습니다"));
    }

    /**
     * SecurityContext에서 현재 인증된 사용자 ID 추출
     */
    private Long getCurrentUserId() {
        Authentication authentication = SecurityContextHolder.getContext().getAuthentication();
        if (authentication == null || !authentication.isAuthenticated()) {
            throw new CustomException(ErrorCode.UNAUTHORIZED, "인증이 필요합니다");
        }

        Object principal = authentication.getPrincipal();
        if (principal instanceof Long) {
            return (Long) principal;
        }

        throw new CustomException(ErrorCode.UNAUTHORIZED, "유효하지 않은 인증 정보입니다");
    }
}
