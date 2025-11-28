package com.heungbuja.music.service;

import com.heungbuja.common.exception.CustomException;
import com.heungbuja.common.exception.ErrorCode;
import com.heungbuja.music.dto.MusicListResponse;
import com.heungbuja.music.dto.MusicPlayResponse;
import com.heungbuja.s3.service.MediaUrlService;
import com.heungbuja.song.entity.Song;
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
     * 음악 재생 - presigned URL 발급
     */
    @Transactional(readOnly = true)
    public MusicPlayResponse playSong(Long songId) {
        Song song = songRepository.findById(songId)
                .orElseThrow(() -> new CustomException(ErrorCode.SONG_NOT_FOUND,
                        "노래를 찾을 수 없습니다 (ID: " + songId + ")"));

        // Media에서 presigned URL 발급
        String audioUrl = mediaUrlService.issueUrlById(song.getMedia().getId());

        log.info("음악 재생: songId={}, title={}, artist={}", songId, song.getTitle(), song.getArtist());

        return MusicPlayResponse.success(
                song.getId(),
                song.getTitle(),
                song.getArtist(),
                audioUrl
        );
    }

    /**
     * 음악 종료 (로깅용)
     */
    public void stopMusic() {
        log.info("음악 종료 요청");
        // 프론트에서 재생 관리하므로 별도 로직 없음
    }
}
