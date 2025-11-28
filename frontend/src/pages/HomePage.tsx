import React, { useEffect } from "react";
import { useNavigate } from "react-router-dom";
import "./HomePage.css";

// 음성 관련 훅과 컴포넌트 import
import { useVoiceRecorder } from "@/hooks/useVoiceRecorder";
import { useVoiceCommand } from "@/hooks/useVoiceCommand";
import VoiceOverlay from "@/components/VoiceOverlay";
import VoiceButton from "@/components/VoiceButton";
import WakeWordDetector from "@/components/WakeWordDetector"; // 새로 만든 컴포넌트

// 상태 관리 스토어 import
import { useAudioStore } from "@/store/audioStore";
import { useGameStore } from "@/store/gameStore";

const BASE_URL = import.meta.env.BASE_URL;

const HomePage: React.FC = () => {
  const navigate = useNavigate();

  // 1. 음성 관련 훅들을 HomePage에서 직접 호출하여 상태를 중앙 관리합니다.
  const { isRecording, countdown, audioBlob, startRecording } = useVoiceRecorder();
  const { isUploading, isPlaying, responseText, response, sendCommand } = useVoiceCommand({});
  const { pause } = useAudioStore();
  const requestGameStop = useGameStore((s) => s.requestStop);

  // 녹음 완료 시 자동 전송 로직
  useEffect(() => {
    if (audioBlob) {
      console.log("녹음 완료! 서버로 전송 중...");
      sendCommand(audioBlob);
    }
  }, [audioBlob, sendCommand]);

  // 2. 녹음을 시작하는 공통 함수를 만듭니다.
  const handleStartVoiceCommand = () => {
    // 이미 다른 작업 중이면 실행하지 않음
    if (isRecording || isUploading || isPlaying) {
      console.log("⚠️ 다른 음성 작업이 진행 중이라 시작할 수 없습니다.");
      return;
    }
    console.log("⏸️ 노래 & 게임 일시정지");
    requestGameStop();
    pause();
    console.log("🎙️ 녹음 시작");
    startRecording();
  };

  const handleMusicClick = () => navigate("/listening");
  const handleExerciseClick = () => navigate("/tutorial");

  const isVoiceBusy = isRecording || isUploading || isPlaying;
  const isEmergency = response?.intent === "EMERGENCY";

  return (
    <div className="home-page">
      {/* WakeWordDetector는 UI가 없으므로 아무 곳에나 두어도 됩니다. */}
      {/* onDetection prop으로 녹음 시작 함수를 넘겨줍니다. */}
      <WakeWordDetector onDetection={handleStartVoiceCommand} />

      <div className="home-container">
        <div className="button-container">
          <button className="home-button music-button" onClick={handleMusicClick}>
            {/* ... 버튼 내용 ... */}
            <div className="button-content">
              <img src={`${BASE_URL}character-music.svg`} alt="노래 감상" className="character-image" />
              <span className="button-text">노래 감상</span>
            </div>
          </button>

          <button className="home-button exercise-button" onClick={handleExerciseClick}>
            {/* ... 버튼 내용 ... */}
            <div className="button-content">
              <img src={`${BASE_URL}character-exercise.svg`} alt="음악 체조" className="character-image" />
              <span className="button-text">음악 체조</span>
            </div>
          </button>
        </div>
      </div>

      {/* VoiceButton에는 클릭 시 실행할 함수와 비활성화 상태를 넘겨줍니다. */}
      <VoiceButton onClick={handleStartVoiceCommand} disabled={isVoiceBusy} />

      {/* VoiceOverlay를 HomePage에서 직접 렌더링하여 상태를 전달합니다. */}
      <VoiceOverlay isVisible={isVoiceBusy} countdown={countdown} isRecording={isRecording} isUploading={isUploading} isPlaying={isPlaying} responseText={responseText} isEmergency={isEmergency} />
    </div>
  );
};

export default HomePage;
