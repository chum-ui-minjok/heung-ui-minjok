import { useEffect, useRef } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { usePorcupine } from "@picovoice/porcupine-react";

// Components
import VoiceButton from "./VoiceButton";
import WakeWordDetector from "./WakeWordDetector";
import VoiceOverlay from "./VoiceOverlay";

// Hooks
import { useVoiceRecorder } from "@/hooks/useVoiceRecorder";
import { useVoiceCommand } from "@/hooks/useVoiceCommand";

// Stores
import { useAudioStore } from "@/store/audioStore";
import { useGameStore } from "@/store/gameStore";

const VITE_ACCESS_KEY = import.meta.env.VITE_PICOVOICE_ACCESS_KEY;
const BASE_URL = import.meta.env.BASE_URL;

const GlobalVoiceController: React.FC = () => {
  const location = useLocation();
  const navigate = useNavigate();

  // 1. usePorcupine 훅을 GlobalVoiceController에서 직접 호출합니다.
  const porcupineHook = usePorcupine();
  const { keywordDetection, release: releasePorcupine } = porcupineHook;

  // 2. 나머지 모든 음성 관련 훅과 로직은 그대로 유지합니다.
  const { isRecording, countdown, audioBlob, startRecording, clearAudioBlob } = useVoiceRecorder();
  // ... (다른 훅과 ref들)
  const { isUploading, isPlaying, responseText, response, sendCommand } = useVoiceCommand({});
  const { pause } = useAudioStore();
  const requestGameStop = useGameStore((s) => s.requestStop);

  const isVoiceBusy = isRecording || isUploading || isPlaying;

  // 3. Picovoice 초기화 로직
  useEffect(() => {
    porcupineHook.init(VITE_ACCESS_KEY, { publicPath: `${BASE_URL}WakeWord/흥민아_ko_wasm_v3_0_0.ppn`, label: "흥민아" }, { publicPath: `${BASE_URL}WakeWord/porcupine_params_ko.pv` });

    // 컴포넌트 언마운트 시 자원 해제
    return () => {
      releasePorcupine();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // 4. Wake Word 감지 시 실행될 로직 (단일 진입점)
  useEffect(() => {
    if (keywordDetection !== null) {
      console.log(`✅ Wake Word "${keywordDetection.label}" 감지됨!`);
      // 시스템이 바쁘지 않을 때만 녹음을 시작합니다. (이중 안전장치)
      if (!isVoiceBusy) {
        handleStartVoiceCommand(false);
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [keywordDetection]); // keywordDetection이 바뀔 때만 실행됩니다.

  // 5. 음성 명령 시작 함수 (기존 로직과 거의 동일)
  const handleStartVoiceCommand = (isManual = false) => {
    // 이 가드 로직이 모든 중복 호출을 막아줍니다.
    if (isRecording || isUploading || isPlaying) {
      console.log("⚠️ 이미 다른 음성 작업이 진행 중입니다.");
      return;
    }

    console.log(`🎤 음성 명령 시작 (수동: ${isManual})`);
    requestGameStop();
    pause();
    startRecording();
  };

  // 녹음 완료 시 자동 전송 로직 (기존과 동일)
  useEffect(() => {
    if (!audioBlob) return;
    sendCommand(audioBlob);
    clearAudioBlob();
  }, [audioBlob, sendCommand, clearAudioBlob]);

  const showVoiceUI = location.pathname !== "/";
  if (!showVoiceUI) return null;

  return (
    <>
      {/* WakeWordDetector는 이제 porcupine 훅 인스턴스와 isVoiceBusy 상태만 받습니다. */}
      <WakeWordDetector porcupineHook={porcupineHook} isVoiceActive={isVoiceBusy} />
      <VoiceButton onClick={() => handleStartVoiceCommand(true)} disabled={isVoiceBusy} />
      <VoiceOverlay
        isVisible={isVoiceBusy}
        countdown={countdown}
        isRecording={isRecording}
        isUploading={isUploading}
        isPlaying={isPlaying}
        responseText={responseText}
        isEmergency={response?.intent === "EMERGENCY"}
      />
    </>
  );
};

export default GlobalVoiceController;
