import { useRef, useState, useCallback, useEffect } from 'react';
import {
  initializePose,
  startCamera,
  stopCamera,
  startPoseDetection,
  stopPoseDetection,
  cleanupPose,
} from '@/services/mediapipe/poseService';

interface UsePoseDetectionReturn {
  videoRef: React.RefObject<HTMLVideoElement | null>;
  isReady: boolean;
  isCameraOn: boolean;
  isDetecting: boolean;
  error: string | null;
  currentLandmarks: number[][] | null;
  startCamera: () => Promise<void>;
  stopCamera: () => void;
  startDetection: () => void;
  stopDetection: () => void;
}

export const usePoseDetection = (): UsePoseDetectionReturn => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [isReady, setIsReady] = useState(false);
  const [isCameraOn, setIsCameraOn] = useState(false);
  const [isDetecting, setIsDetecting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentLandmarks, setCurrentLandmarks] = useState<number[][] | null>(null);

  /**
   * MediaPipe 초기화
   */
  useEffect(() => {
    const init = async () => {
      try {
        await initializePose();
        setIsReady(true);
      } catch (err) {
        const message = err instanceof Error ? err.message : 'MediaPipe 초기화 실패';
        setError(message);
        console.error('❌ MediaPipe 초기화 실패:', err);
      }
    };

    init();

    return () => {
      cleanupPose();
    };
  }, []);

  /**
   * 카메라 시작
   */
  const handleStartCamera = useCallback(async () => {
    if (!videoRef.current) {
      setError('비디오 엘리먼트가 없습니다');
      return;
    }

    try {
      setError(null);
      await startCamera(videoRef.current, (landmarks) => {
        setCurrentLandmarks(landmarks);
      });
      setIsCameraOn(true);
    } catch (err) {
      const message = err instanceof Error ? err.message : '카메라 시작 실패';
      setError(message);
      console.error('❌ 카메라 시작 실패:', err);
    }
  }, []);

  /**
   * 카메라 중지
   */
  const handleStopCamera = useCallback(() => {
    stopCamera();
    setIsCameraOn(false);
    setIsDetecting(false);
    setCurrentLandmarks(null);
  }, []);

  /**
   * Pose 감지 시작 (카메라는 이미 켜져있어야 함)
   */
  const handleStartDetection = useCallback(() => {
    startPoseDetection();
    setIsDetecting(true);
  }, []);

  /**
   * Pose 감지 중지 (카메라는 계속 유지)
   */
  const handleStopDetection = useCallback(() => {
    stopPoseDetection();
    setIsDetecting(false);
    setCurrentLandmarks(null);
  }, []);

  return {
    videoRef,
    isReady,
    isCameraOn,
    isDetecting,
    error,
    currentLandmarks,
    startCamera: handleStartCamera,
    stopCamera: handleStopCamera,
    startDetection: handleStartDetection,
    stopDetection: handleStopDetection,
  };
};
