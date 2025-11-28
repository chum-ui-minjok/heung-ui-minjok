import { useEffect } from "react";
import { usePorcupine } from "@picovoice/porcupine-react";

interface WakeWordDetectorProps {
  onDetection: () => void;
  isVoiceActive: boolean;
}

const VITE_ACCESS_KEY = import.meta.env.VITE_PICOVOICE_ACCESS_KEY;
const BASE_URL = import.meta.env.BASE_URL;

const WakeWordDetector: React.FC<WakeWordDetectorProps> = ({ onDetection, isVoiceActive }) => {
  const { keywordDetection, isLoaded, isListening, error, init, start, stop, release } = usePorcupine();

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
    // 초기화가 완료되었고,
    if (isLoaded) {
      // 음성 시스템이 바쁘지 않고(false) + 현재 듣고 있지 않다면 -> 감지를 시작합니다.
      if (!isVoiceActive && !isListening) {
        console.log("[WakeWord] 음성 시스템 유휴 상태. 감지를 시작합니다.");
        start();
      }
      // 음성 시스템이 바쁘고(true) + 현재 듣고 있다면 -> 감지를 중지합니다. (무한 루프 방지)
      else if (isVoiceActive && isListening) {
        console.log("[WakeWord] 음성 시스템 활성 상태. 감지를 중지합니다.");
        stop();
      }
    }
  }, [isLoaded, isListening, isVoiceActive, start, stop]);

  useEffect(() => {
    if (keywordDetection !== null) {
      console.log(`✅ [WakeWord] 3. "${keywordDetection.label}" 감지 성공!!!`);

      // 👇 핵심 수정! 감지하자마자 스스로 멈춥니다.
      if (isListening) {
        console.log("[WakeWord] 감지 성공! 즉시 감지를 중지합니다.");
        stop();
      }

      onDetection(); // 그 다음에 부모에게 알립니다.
    }
  }, [keywordDetection, onDetection, isListening, stop]);

  useEffect(() => {
    if (error) {
      console.error("❌ [WakeWord] 4. 에러 발생:", error);
    }
  }, [error]);

  return null;
};

export default WakeWordDetector;
