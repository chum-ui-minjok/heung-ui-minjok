// ===== MediaPipe Pose 관련 타입 =====

/**
 * 프레임 데이터 (WebSocket 전송용)
 * 기존 frameData(Base64 이미지) 대신 poseData(좌표 배열) 사용
 */
export interface PoseFrameData {
  sessionId: string;
  currentPlayTime: number;
  poseData: number[][];  // [[x, y], [x, y], ...] 33개 랜드마크
}

/**
 * MediaPipe 랜드마크 (원본)
 */
export interface PoseLandmark {
  x: number;        // 0~1 (이미지 너비 기준 정규화)
  y: number;        // 0~1 (이미지 높이 기준 정규화)
  z: number;        // 깊이 (사용 안 함)
  visibility: number;  // 가시성 (사용 안 함)
}

/**
 * MediaPipe Pose 설정 옵션
 */
export interface PoseOptions {
  modelComplexity: 0 | 1 | 2;      // 0: 빠름, 1: 균형, 2: 정확
  smoothLandmarks: boolean;         // 좌표 떨림 방지
  enableSegmentation: boolean;      // 배경 분리
  minDetectionConfidence: number;   // 사람 감지 최소 신뢰도 (0~1)
  minTrackingConfidence: number;    // 추적 최소 신뢰도 (0~1)
}

/**
 * 랜드마크 인덱스 상수
 * 서버에서는 11~32번만 사용하지만, 전체 33개를 전송
 */
export const POSE_LANDMARKS = {
  // 얼굴 (0-10) - 서버에서 미사용
  NOSE: 0,
  LEFT_EYE_INNER: 1,
  LEFT_EYE: 2,
  LEFT_EYE_OUTER: 3,
  RIGHT_EYE_INNER: 4,
  RIGHT_EYE: 5,
  RIGHT_EYE_OUTER: 6,
  LEFT_EAR: 7,
  RIGHT_EAR: 8,
  MOUTH_LEFT: 9,
  MOUTH_RIGHT: 10,

  // 상체 (11-22) - 서버에서 사용
  LEFT_SHOULDER: 11,
  RIGHT_SHOULDER: 12,
  LEFT_ELBOW: 13,
  RIGHT_ELBOW: 14,
  LEFT_WRIST: 15,
  RIGHT_WRIST: 16,
  LEFT_PINKY: 17,
  RIGHT_PINKY: 18,
  LEFT_INDEX: 19,
  RIGHT_INDEX: 20,
  LEFT_THUMB: 21,
  RIGHT_THUMB: 22,

  // 하체 (23-32) - 서버에서 사용
  LEFT_HIP: 23,
  RIGHT_HIP: 24,
  LEFT_KNEE: 25,
  RIGHT_KNEE: 26,
  LEFT_ANKLE: 27,
  RIGHT_ANKLE: 28,
  LEFT_HEEL: 29,
  RIGHT_HEEL: 30,
  LEFT_FOOT_INDEX: 31,
  RIGHT_FOOT_INDEX: 32,
} as const;

export const TOTAL_LANDMARKS = 33;
