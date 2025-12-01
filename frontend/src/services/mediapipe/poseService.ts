import { Pose, type Results, type NormalizedLandmark } from '@mediapipe/pose';
import { Camera } from '@mediapipe/camera_utils';
import { TOTAL_LANDMARKS } from '@/types';
import { GAME_CONFIG } from '@/utils/constants';

type PoseCallback = (landmarks: number[][] | null) => void;

let poseInstance: Pose | null = null;
let cameraInstance: Camera | null = null;
let onResultsCallback: PoseCallback | null = null;
let isDetectionActive = false; // Pose 감지 활성화 여부
let isPoseReady = false; // Pose 인스턴스가 사용 가능한 상태인지
let lastSendTime = 0; // FPS 제한용

/**
 * MediaPipe Pose 초기화
 */
export const initializePose = async (): Promise<Pose> => {
  if (poseInstance) {
    isPoseReady = true;  // 이미 초기화된 경우에도 플래그 복원
    return poseInstance;
  }

  poseInstance = new Pose({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`,
  });

  poseInstance.setOptions({
    modelComplexity: 1,
    smoothLandmarks: true,
    enableSegmentation: false,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });

  poseInstance.onResults(handleResults);

  await poseInstance.initialize();
  isPoseReady = true;
  console.log('✅ MediaPipe Pose 초기화 완료');

  return poseInstance;
};

/**
 * MediaPipe 결과 처리
 */
const handleResults = (results: Results): void => {
  // 감지가 비활성화되어 있으면 콜백 호출 안 함
  if (!isDetectionActive || !onResultsCallback) return;

  // FPS 제한 (GAME_CONFIG.FRAME_MS 간격으로만 전송)
  const now = performance.now();
  if (now - lastSendTime < GAME_CONFIG.FRAME_MS) return;
  lastSendTime = now;

  if (!results.poseLandmarks) {
    onResultsCallback(null);
    return;
  }

  // 33개 랜드마크에서 [x, y]만 추출
  const landmarks: number[][] = results.poseLandmarks.map(
    (lm: NormalizedLandmark) => [lm.x, lm.y]
  );

  if (landmarks.length !== TOTAL_LANDMARKS) {
    console.warn(`⚠️ 랜드마크 수 불일치: ${landmarks.length} (expected: ${TOTAL_LANDMARKS})`);
    onResultsCallback(null);
    return;
  }

  onResultsCallback(landmarks);
};

/**
 * 카메라만 시작 (Pose 감지는 비활성화 상태)
 */
export const startCamera = async (
  videoElement: HTMLVideoElement,
  callback: PoseCallback
): Promise<void> => {
  onResultsCallback = callback;
  isDetectionActive = false; // 처음엔 감지 비활성화

  if (!poseInstance) {
    await initializePose();
  }

  if (cameraInstance) {
    console.log('📹 카메라 이미 실행 중');
    return;
  }

  cameraInstance = new Camera(videoElement, {
    onFrame: async () => {
      // isPoseReady 체크: close() 이후에는 프레임을 보내지 않음
      if (isPoseReady && poseInstance && videoElement.readyState >= 2) {
        await poseInstance.send({ image: videoElement });
      }
    },
    width: 640,
    height: 480,
  });

  await cameraInstance.start();
  console.log('📹 카메라 시작 (Pose 감지 대기 중)');
};

/**
 * Pose 감지 활성화 (카메라는 이미 실행 중이어야 함)
 */
export const startPoseDetection = (): void => {
  isDetectionActive = true;
  console.log('✅ Pose 감지 활성화');
};

/**
 * Pose 감지 비활성화 (카메라는 계속 실행)
 */
export const stopPoseDetection = (): void => {
  isDetectionActive = false;
  console.log('⏹ Pose 감지 비활성화');
};

/**
 * 카메라 중지
 */
export const stopCamera = (): void => {
  isDetectionActive = false;
  if (cameraInstance) {
    cameraInstance.stop();
    cameraInstance = null;
  }
  onResultsCallback = null;
  console.log('📹 카메라 중지');
};

/**
 * 리소스 정리
 * 순서: isPoseReady off → 카메라 중지
 * 참고: poseInstance.close()는 호출하지 않음 (WASM 모듈 재초기화 문제 방지)
 *       페이지 완전 이탈 시 브라우저가 자동 정리함
 */
export const cleanupPose = (): void => {
  // 1. 플래그를 먼저 끔 → onFrame에서 send() 호출 방지
  isPoseReady = false;

  // 2. 카메라 중지 (프레임 전송 완전 중단)
  stopCamera();

  // poseInstance는 유지 (close() 호출 시 WASM 재초기화 불가)
  console.log('🧹 MediaPipe Pose 리소스 정리 완료');
};
