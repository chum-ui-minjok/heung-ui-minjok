import { useEffect, useRef } from "react";
import { usePorcupine } from "@picovoice/porcupine-react";

interface WakeWordDetectorProps {
  onDetection: () => void;
}

const VITE_ACCESS_KEY = import.meta.env.VITE_PICOVOICE_ACCESS_KEY;
const BASE_URL = import.meta.env.BASE_URL;

const WakeWordDetector: React.FC<WakeWordDetectorProps> = ({ onDetection }) => {
  const { keywordDetection, isLoaded, isListening, error, init, start, release } = usePorcupine();
  const detectionTimeoutRef = useRef<number | null>(null);

  useEffect(() => {
    init(VITE_ACCESS_KEY, { publicPath: `${BASE_URL}WakeWord/흥민아_ko_wasm_v3_0_0.ppn`, label: "흥민아" }, { publicPath: `${BASE_URL}WakeWord/porcupine_params_ko.pv` });
    return () => {
      release();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (isLoaded && !isListening) {
      start();
    }
  }, [isLoaded, isListening, start]);

  useEffect(() => {
    if (keywordDetection !== null) {
      // 중복 호출을 막기 위해 짧은 시간 내의 호출은 무시
      if (detectionTimeoutRef.current) return;

      console.log(`✅ [WakeWord] 3. "${keywordDetection.label}" 감지 성공!!!`);
      onDetection();

      // 1초 동안 추가 감지를 막습니다.
      detectionTimeoutRef.current = setTimeout(() => {
        detectionTimeoutRef.current = null;
      }, 1000);
    }
  }, [keywordDetection, onDetection]);

  useEffect(() => {
    if (error) {
      console.error("❌ [WakeWord] 4. 에러 발생:", error);
    }
  }, [error]);

  return null;
};

export default WakeWordDetector;
