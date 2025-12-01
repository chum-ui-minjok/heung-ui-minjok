import { BrowserRouter, Routes, Route, useLocation, useNavigate } from "react-router-dom";
import { useEffect, useRef } from "react";
import HomePage from "./pages/HomePage";
import GamePage from "./pages/GamePage";
import TutorialPage from "./pages/TutorialPage";
import ResultPage from "./pages/ResultPage";
import SongPage from "./pages/SongPage";
// import RaspberryLoginPage from './pages/RaspberryLoginPage';
import WebLoginPage from "./pages/WebLoginPage";
import SongListPage from "./pages/SongListPage";
import ProtectedRoute from "./components/ProtectedRoute";
import VoiceButton from "./components/VoiceButton";
import WakeWordDetector from "./components/WakeWordDetector";
import VoiceOverlay from "./components/VoiceOverlay";
import GlobalVoiceController from "./components/GlobalVoiceController";

// Hooks
import { useVoiceRecorder } from "./hooks/useVoiceRecorder";
import { useVoiceCommand } from "./hooks/useVoiceCommand";

// Stores
import { useAudioStore } from "./store/audioStore";
import { useGameStore } from "./store/gameStore";

// import { checkIfRaspberryPi } from './utils/deviceDetector';
// import { useEnvironmentStore } from './store/environmentStore';
import "./index.css";
import "./App.css";

function App() {
  return (
    <BrowserRouter basename="/user">
      <div className="app">
        <Routes>
          <Route path="/" element={<WebLoginPage />} />
          <Route
            path="/home"
            element={
              <ProtectedRoute>
                <HomePage />
              </ProtectedRoute>
            }
          />
          <Route
            path="/list"
            element={
              <ProtectedRoute>
                <SongListPage />
              </ProtectedRoute>
            }
          />
          <Route
            path="/listening"
            element={
              <ProtectedRoute>
                <SongPage />
              </ProtectedRoute>
            }
          />
          <Route
            path="/tutorial"
            element={
              <ProtectedRoute>
                <TutorialPage />
              </ProtectedRoute>
            }
          />
          <Route
            path="/game/:songId"
            element={
              <ProtectedRoute>
                <GamePage />
              </ProtectedRoute>
            }
          />
          <Route
            path="/result"
            element={
              <ProtectedRoute>
                <ResultPage />
              </ProtectedRoute>
            }
          />
        </Routes>

        {/* 음성 기능 전체를 이 컴포넌트 하나가 책임집니다. */}
        <GlobalVoiceController />
      </div>
    </BrowserRouter>
  );
}

export default App;
