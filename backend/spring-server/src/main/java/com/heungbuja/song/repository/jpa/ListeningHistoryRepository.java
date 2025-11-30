package com.heungbuja.song.repository.jpa;

import com.heungbuja.song.entity.ListeningHistory;
import com.heungbuja.song.enums.PlaybackMode;
import com.heungbuja.user.entity.User;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface ListeningHistoryRepository extends JpaRepository<ListeningHistory, Long> {

    /**
     * 사용자의 최근 청취 이력 조회
     */
    List<ListeningHistory> findByUserOrderByPlayedAtDesc(User user);

    /**
     * 사용자의 최근 N개 청취 이력 조회
     */
    List<ListeningHistory> findTop10ByUserOrderByPlayedAtDesc(User user);

    /**
     * 곡별 재생 횟수 집계 (전체)
     * @return [songId, playCount] 배열 리스트
     */
    @Query("SELECT h.song.id, COUNT(h) FROM ListeningHistory h GROUP BY h.song.id")
    List<Object[]> countBySong();

    /**
     * 모드별 곡 재생 횟수 집계 (LISTENING / GAME)
     * @return [songId, playCount] 배열 리스트
     */
    @Query("SELECT h.song.id, COUNT(h) FROM ListeningHistory h WHERE h.mode = :mode GROUP BY h.song.id")
    List<Object[]> countBySongAndMode(@Param("mode") PlaybackMode mode);
}
