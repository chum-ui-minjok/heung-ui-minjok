import { useRef, useState, useCallback, useEffect } from 'react';
import {
  initializePose,
  startPoseDetection,
  stopPoseDetection,
  cleanupPose,
} from '@/services/mediapipe/poseService';

interface UsePoseDetectionReturn {
  videoRef: React.RefObject<HTMLVideoElement | null>;
  isReady: boolean;
  isDetecting: boolean;
  error: string | null;
  currentLandmarks: number[][] | null;
  start: () => Promise<void>;
  stop: () => void;
}

export const usePoseDetection = (): UsePoseDetectionReturn => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [isReady, setIsReady] = useState(false);
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
   * Pose 감지 시작
   */
  const start = useCallback(async () => {
    if (!videoRef.current) {
      setError('비디오 엘리먼트가 없습니다');
      return;
    }

    try {
      setError(null);
      await startPoseDetection(videoRef.current, (landmarks) => {
        setCurrentLandmarks(landmarks);
      });
      setIsDetecting(true);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Pose 감지 시작 실패';
      setError(message);
      console.error('❌ Pose 감지 시작 실패:', err);
    }
  }, []);

  /**
   * Pose 감지 중지
   */
  const stop = useCallback(() => {
    stopPoseDetection();
    setIsDetecting(false);
    setCurrentLandmarks(null);
  }, []);

  return {
    videoRef,
    isReady,
    isDetecting,
    error,
    currentLandmarks,
    start,
    stop,
  };
};
