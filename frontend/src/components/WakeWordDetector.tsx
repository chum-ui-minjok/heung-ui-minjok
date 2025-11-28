import { useEffect } from "react";
import { usePorcupine } from "@picovoice/porcupine-react";

interface WakeWordDetectorProps {
  onDetection: () => void;
}

const VITE_ACCESS_KEY = import.meta.env.VITE_PICOVOICE_ACCESS_KEY;
const BASE_URL = import.meta.env.BASE_URL;

const WakeWordDetector: React.FC<WakeWordDetectorProps> = ({ onDetection }) => {
  const { keywordDetection, isLoaded, isListening, error, init, start, release } = usePorcupine();

  useEffect(() => {
    console.log("[WakeWord] 1. Picovoice 초기화를 시도합니다.");
    console.log("[WakeWord] AccessKey:", VITE_ACCESS_KEY ? "있음" : "없음!!!");

    // ✅ publicPath 앞에 BASE_URL을 꼭 붙여주세요.
    init(
      VITE_ACCESS_KEY,
      {
        publicPath: `${BASE_URL}WakeWord/흥민아_ko_wasm_v3_0_0.ppn`,
        label: "흥민아",
      },
      {
        publicPath: `${BASE_URL}WakeWord/porcupine_params_ko.pv`,
      }
    );

    return () => {
      release();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (isLoaded && !isListening) {
      console.log("[WakeWord] 2. 초기화 완료! 감지를 시작합니다.");
      start();
    }
  }, [isLoaded, isListening, start]);

  useEffect(() => {
    if (keywordDetection !== null) {
      console.log(`✅ [WakeWord] 3. "${keywordDetection.label}" 감지 성공!!!`);
      onDetection();
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
