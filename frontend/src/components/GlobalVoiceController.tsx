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

  const porcupineHook = usePorcupine();
  const { keywordDetection, release: releasePorcupine } = porcupineHook;

  const { isRecording, countdown, audioBlob, startRecording, clearAudioBlob } = useVoiceRecorder();

  // ✨ 1. 기존 VoiceButton의 ref들을 모두 가져옵니다.
  const autoRetryFlagRef = useRef(false);
  const prevIsPlayingRef = useRef(false);
  const isManualRecordingRef = useRef(false);
  const emergencyRetryCountRef = useRef(0);

  const { isUploading, isPlaying, responseText, response, sendCommand } = useVoiceCommand({
    onRetry: () => {
      if (!autoRetryFlagRef.current) return;
      autoRetryFlagRef.current = false;
      startRecording();
    },
  });

  const { pause } = useAudioStore();
  const requestGameStop = useGameStore((s) => s.requestStop);

  const isVoiceBusy = isRecording || isUploading || isPlaying;
  const isEmergency = response?.intent === "EMERGENCY";

  // Picovoice 초기화 로직
  useEffect(() => {
    porcupineHook.init(VITE_ACCESS_KEY, { publicPath: `${BASE_URL}WakeWord/흥민아_ko_wasm_v3_0_0.ppn`, label: "흥민아" }, { publicPath: `${BASE_URL}WakeWord/porcupine_params_ko.pv` });
    return () => {
      releasePorcupine();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Wake Word 감지 로직
  useEffect(() => {
    if (keywordDetection !== null) {
      if (!isVoiceBusy) {
        handleStartVoiceCommand(false);
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [keywordDetection]);

  // ✨ 2. Emergency 재녹음 로직을 그대로 가져옵니다.
  useEffect(() => {
    const ttsJustFinished = prevIsPlayingRef.current === true && !isPlaying && !isRecording && !isUploading;

    if (isManualRecordingRef.current && isEmergency && ttsJustFinished) {
      if (emergencyRetryCountRef.current === 0) {
        console.log("🚨 응급 상황 인식 → 재녹음 1회 실행");
        emergencyRetryCountRef.current = 1;
        startRecording();
      } else {
        console.log("🚨 두 번째 응급 인식 → 홈으로 이동");
        isManualRecordingRef.current = false;
        emergencyRetryCountRef.current = 0;
        navigate("/home");
      }
    }
    prevIsPlayingRef.current = isPlaying;
  }, [isEmergency, isPlaying, isRecording, isUploading, navigate, startRecording]);

  // ✨ 3. handleStartVoiceCommand 함수를 수정하여 ref 초기화 로직을 추가합니다.
  const handleStartVoiceCommand = (isManual = false) => {
    if (isRecording || isUploading || isPlaying) {
      console.log("⚠️ 이미 다른 음성 작업이 진행 중입니다.");
      return;
    }

    console.log(`🎤 음성 명령 시작 (수동: ${isManual})`);
    autoRetryFlagRef.current = true;
    requestGameStop();
    pause();

    //Wake Word로 호출되어도, 응급 상황을 대비해 항상 플래그를 true로 설정하고 카운트를 리셋합니다.
    isManualRecordingRef.current = true;
    emergencyRetryCountRef.current = 0;

    startRecording();
  };

  // 녹음 완료 시 자동 전송 로직
  useEffect(() => {
    if (!audioBlob) return;
    sendCommand(audioBlob);
    clearAudioBlob();
  }, [audioBlob, sendCommand, clearAudioBlob]);

  const showVoiceUI = location.pathname !== "/";
  if (!showVoiceUI) return null;

  return (
    <>
      <WakeWordDetector porcupineHook={porcupineHook} isVoiceActive={isVoiceBusy} />
      <VoiceButton onClick={() => handleStartVoiceCommand(true)} disabled={isVoiceBusy} />
      <VoiceOverlay isVisible={isVoiceBusy} countdown={countdown} isRecording={isRecording} isUploading={isUploading} isPlaying={isPlaying} responseText={responseText} isEmergency={isEmergency} />
    </>
  );
};

export default GlobalVoiceController;
