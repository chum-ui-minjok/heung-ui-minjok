package com.heungbuja.common.websocket;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;

import java.time.LocalDateTime;

@Getter
@Builder
@AllArgsConstructor
public class EmergencyAlertMessage {

    private String type;  // "EMERGENCY_REPORT"
    private Long reportId;
    private Long userId;
    private String userName;
    private String triggerWord;
    private LocalDateTime reportedAt;
    private String priority;  // "CRITICAL"

    public static EmergencyAlertMessage from(Long reportId, Long userId, String userName,
                                             String triggerWord, LocalDateTime reportedAt) {
        return EmergencyAlertMessage.builder()
                .type("EMERGENCY_REPORT")
                .reportId(reportId)
                .userId(userId)
                .userName(userName)
                .triggerWord(triggerWord)
                .reportedAt(reportedAt)
                .priority("CRITICAL")
                .build();
    }
}
