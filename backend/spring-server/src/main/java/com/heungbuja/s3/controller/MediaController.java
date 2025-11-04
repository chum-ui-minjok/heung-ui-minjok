package com.heungbuja.s3.controller;

import com.heungbuja.s3.service.MediaUrlService;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

import java.util.Map;

@RestController
@RequestMapping("/media")
@RequiredArgsConstructor
public class MediaController {

    private final MediaUrlService mediaUrlService;

    // 단일 파일용 (노래/영상)
    @GetMapping("/{id}/url")
    public Map<String, String> getUrl(@PathVariable long id) {
        String url = mediaUrlService.issueUrlById(id);
        return Map.of("url", url);
    }

    // 음악 + 영상 쌍 URL 반환 (동시 재생용)
    @GetMapping("/pair/url")
    public Map<String, String> getPairUrl(@RequestParam long musicId,
                                          @RequestParam long videoId) {
        String musicUrl = mediaUrlService.issueUrlById(musicId);
        String videoUrl = mediaUrlService.issueUrlById(videoId);
        return Map.of("musicUrl", musicUrl, "videoUrl", videoUrl);
    }

    @GetMapping("/test")
    public Map<String, String> testPresignedUrl() {
        String url = mediaUrlService.testPresignedUrl();
        return Map.of("url", url);
    }

    // 로컬 테스트: DB 없이 고정 비디오 키로 프리사인드 URL 발급
    @GetMapping("/test/video")
    public Map<String, String> testVideoPresignedUrl() {
        String url = mediaUrlService.testPresignedUrl("video/level3.mp4");
        return Map.of("url", url);
    }
}
