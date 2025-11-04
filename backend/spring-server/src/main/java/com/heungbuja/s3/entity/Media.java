package com.heungbuja.s3.entity;

import jakarta.persistence.*;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@Entity
@Table(name = "media")
@Getter
@Setter
@NoArgsConstructor
public class Media {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    private String title;             // 제목
    private String type;              // MUSIC / VIDEO
    private String s3Key;             // S3 경로 (예: music/song1.mp3)
    private String bucket;            // 버킷명
    private String mimeType;          // 파일 형식
    private Long sizeBytes;           // 파일 크기
    private Long uploaderId;          // 업로더 ID

    @Column(updatable = false)
    private java.time.LocalDateTime createdAt = java.time.LocalDateTime.now();
}