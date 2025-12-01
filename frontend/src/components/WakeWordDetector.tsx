import { useEffect } from "react";
import { usePorcupine } from "@picovoice/porcupine-react";

// 인터페이스 변경: 이제 isVoiceActive와 porcupineHook prop만 받습니다.
interface WakeWordDetectorProps {
  porcupineHook: ReturnType<typeof usePorcupine>; // 부모의 훅 인스턴스를 직접 받음
  isVoiceActive: boolean;
}

const WakeWordDetector: React.FC<WakeWordDetectorProps> = ({ porcupineHook, isVoiceActive }) => {
  // 부모로부터 받은 훅의 상태와 함수를 사용합니다.
  const { isLoaded, isListening, start, stop } = porcupineHook;

  useEffect(() => {
    if (isLoaded) {
      // 음성 시스템이 바쁘고(true), 현재 듣고 있다면 -> 감지를 중지합니다.
      if (isVoiceActive && isListening) {
        console.log("[WakeWord] Voice system is busy. Stopping listener.");
        stop();
      }
      // 음성 시스템이 한가하고(false), 현재 듣고 있지 않다면 -> 감지를 시작합니다.
      else if (!isVoiceActive && !isListening) {
        console.log("[WakeWord] Voice system is idle. Starting listener.");
        start();
      }
    }
  }, [isLoaded, isListening, isVoiceActive, start, stop]);

  // 이 컴포넌트는 UI나 다른 로직 없이 오직 start/stop 제어만 담당합니다.
  return null;
};

export default WakeWordDetector;
