import { BrowserRouter, Routes, Route, useLocation } from "react-router-dom";
import { useEffect } from "react";
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

function AppContent() {
  // const [isChecking, setIsChecking] = useState<boolean>(true);
  // const { isRaspberryPi, deviceId, setEnvironment } = useEnvironmentStore();

  // useEffect(() => {
  //     detectEnvironment();
  // }, []);

  // const detectEnvironment = async () => {
  //     try {
  //         const result = await checkIfRaspberryPi();
  //         setEnvironment(result.isRaspberryPi, result.deviceId);
  //     } catch (error) {
  //         console.error('Environment detection error:', error);
  //         setEnvironment(false);
  //     } finally {
  //         setIsChecking(false);
  //     }
  // };

  // // 환경 체크 중
  // if (isChecking) {
  //     return (
  //         <div className="app-loading">
  //             <div className="loading-container">
  //                 <div className="spinner"></div>
  //                 <h2>환경 확인 중...</h2>
  //                 <p className="loading-text">잠시만 기다려주세요</p>
  //             </div>
  //         </div>
  //     );
  // }
  const location = useLocation();

  const { isRecording, countdown, audioBlob, startRecording } = useVoiceRecorder();
  const { isUploading, isPlaying, responseText, response, sendCommand } = useVoiceCommand({});
  const { pause } = useAudioStore();
  const requestGameStop = useGameStore((s) => s.requestStop);

  useEffect(() => {
    if (audioBlob) {
      console.log("App: 녹음 완료! 서버로 전송 중...");
      sendCommand(audioBlob);
    }
  }, [audioBlob, sendCommand]);

  const handleStartVoiceCommand = () => {
    if (isRecording || isUploading || isPlaying) return;
    requestGameStop();
    pause();
    startRecording();
  };

  const isVoiceBusy = isRecording || isUploading || isPlaying;
  const isEmergency = response?.intent === "EMERGENCY";
  const showVoiceUI = location.pathname !== "/";

  return (
    <div className="app">
      <Routes>
        {/* ... 라우팅 경로는 동일 ... */}
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

      {showVoiceUI && (
        <>
          {/* 👇 isVoiceBusy 상태를 isVoiceActive prop으로 전달합니다. */}
          <WakeWordDetector onDetection={handleStartVoiceCommand} isVoiceActive={isVoiceBusy} />
          <VoiceButton onClick={handleStartVoiceCommand} disabled={isVoiceBusy} />
          <VoiceOverlay isVisible={isVoiceBusy} countdown={countdown} isRecording={isRecording} isUploading={isUploading} isPlaying={isPlaying} responseText={responseText} isEmergency={isEmergency} />
        </>
      )}
    </div>
  );
}

// 최종 App 컴포넌트는 BrowserRouter로 AppContent를 감싸줍니다.
function App() {
  return (
    <BrowserRouter basename="/user">
      <AppContent />
    </BrowserRouter>
  );
}

export default App;
