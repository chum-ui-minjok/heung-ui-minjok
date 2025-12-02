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
import "./HomePage.css";
import { useModeStore } from "@/store/modeStore";
import "./HomePage.css";

const BASE_URL = import.meta.env.BASE_URL;

const HomePage: React.FC = () => {
  const navigate = useNavigate();
  const { setMode } = useModeStore();

  const handleMusicClick = () => {
    setMode("LISTENING");
    navigate("/list");
  };

  const handleExerciseClick = () => {
    setMode("EXERCISE");
    navigate("/list");
  };

  return (
    <div className="home-page">
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
    </div>
  );
};

export default HomePage;
