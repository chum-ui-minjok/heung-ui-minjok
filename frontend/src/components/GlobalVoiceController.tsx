import { useEffect, useRef } from "react";
import { useLocation, useNavigate } from "react-router-dom";

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

const GlobalVoiceController: React.FC = () => {
  const location = useLocation();
  const navigate = useNavigate();

  // 기존 AppContent에 있던 모든 음성 관련 로직을 그대로 가져옵니다.
  const { isRecording, countdown, audioBlob, startRecording, clearAudioBlob } = useVoiceRecorder();
  const autoRetryFlagRef = useRef(false);

  const { isUploading, isPlaying, responseText, response, sendCommand } = useVoiceCommand({
    onRetry: () => {
      if (!autoRetryFlagRef.current) return;
      autoRetryFlagRef.current = false;
      startRecording();
    },
  });

  const { pause } = useAudioStore();
  const requestGameStop = useGameStore((s) => s.requestStop);
  const isEmergency = response?.intent === "EMERGENCY";
  const prevIsPlayingRef = useRef(false);
  const isManualRecordingRef = useRef(false);
  const emergencyRetryCountRef = useRef(0);

  useEffect(() => {
    const ttsJustFinished = prevIsPlayingRef.current === true && !isPlaying && !isRecording && !isUploading;
    if (isManualRecordingRef.current && isEmergency && ttsJustFinished) {
      if (emergencyRetryCountRef.current === 0) {
        emergencyRetryCountRef.current = 1;
        startRecording();
      } else {
        isManualRecordingRef.current = false;
        emergencyRetryCountRef.current = 0;
        navigate("/home");
      }
    }
    prevIsPlayingRef.current = isPlaying;
  }, [isEmergency, isPlaying, isRecording, isUploading, navigate, startRecording]);

  const handleStartVoiceCommand = (isManual = false) => {
    if (!isRecording && !isUploading && !isPlaying) {
      autoRetryFlagRef.current = true;
      requestGameStop();
      pause();
      if (isManual) {
        isManualRecordingRef.current = true;
        emergencyRetryCountRef.current = 0;
      }
      startRecording();
    }
  };

  useEffect(() => {
    if (!audioBlob) return;
    sendCommand(audioBlob);
    clearAudioBlob();
  }, [audioBlob, sendCommand, clearAudioBlob]);

  const isVoiceBusy = isRecording || isUploading || isPlaying;
  const showVoiceUI = location.pathname !== "/";

  // 로그인 페이지가 아니면 음성 관련 UI를 렌더링합니다.
  if (!showVoiceUI) {
    return null;
  }

  return (
    <>
      <WakeWordDetector onDetection={() => handleStartVoiceCommand(false)} />
      <VoiceButton onClick={() => handleStartVoiceCommand(true)} disabled={isVoiceBusy} />
      <VoiceOverlay isVisible={isVoiceBusy} countdown={countdown} isRecording={isRecording} isUploading={isUploading} isPlaying={isPlaying} responseText={responseText} isEmergency={isEmergency} />
    </>
  );
};

export default GlobalVoiceController;
