import { Pose, type Results, type NormalizedLandmark } from '@mediapipe/pose';
import { Camera } from '@mediapipe/camera_utils';
import { TOTAL_LANDMARKS } from '@/types';

type PoseCallback = (landmarks: number[][] | null) => void;

let poseInstance: Pose | null = null;
let cameraInstance: Camera | null = null;
let onResultsCallback: PoseCallback | null = null;

/**
 * MediaPipe Pose 초기화
 */
export const initializePose = async (): Promise<Pose> => {
  if (poseInstance) return poseInstance;

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
  console.log('✅ MediaPipe Pose 초기화 완료');

  return poseInstance;
};

/**
 * MediaPipe 결과 처리
 */
const handleResults = (results: Results): void => {
  if (!onResultsCallback) return;

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
 * 카메라 시작 및 Pose 연결
 */
export const startPoseDetection = async (
  videoElement: HTMLVideoElement,
  callback: PoseCallback
): Promise<void> => {
  onResultsCallback = callback;

  if (!poseInstance) {
    await initializePose();
  }

  cameraInstance = new Camera(videoElement, {
    onFrame: async () => {
      if (poseInstance && videoElement.readyState >= 2) {
        await poseInstance.send({ image: videoElement });
      }
    },
    width: 640,
    height: 480,
  });

  await cameraInstance.start();
  console.log('✅ Pose 감지 시작');
};

/**
 * Pose 감지 중지
 */
export const stopPoseDetection = (): void => {
  if (cameraInstance) {
    cameraInstance.stop();
    cameraInstance = null;
  }
  onResultsCallback = null;
  console.log('⏹ Pose 감지 중지');
};

/**
 * 리소스 정리
 */
export const cleanupPose = (): void => {
  stopPoseDetection();
  if (poseInstance) {
    poseInstance.close();
    poseInstance = null;
  }
  console.log('🧹 MediaPipe Pose 리소스 정리 완료');
};
