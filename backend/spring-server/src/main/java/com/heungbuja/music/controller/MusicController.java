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
        MusicPlayResponse response = musicService.playSong(request.getSongId());
        return ResponseEntity.ok(response);
    }

    /**
     * 음악 종료
     */
    @PostMapping("/stop")
    public ResponseEntity<ControlResponse> stopMusic() {
        musicService.stopMusic();
        return ResponseEntity.ok(ControlResponse.success("음악이 종료되었습니다"));
    }
}
