import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import LoginPage from './pages/LoginPage';
import DashboardPage from './pages/DashboardPage';
import SongVisualizationPage from './pages/SongVisualizationPage';
import DeveloperDashboardPage from './pages/DeveloperDashboardPage';

function App() {
  return (
    <BrowserRouter basename="/admin">
      <Routes>
        <Route path="/login" element={<LoginPage />} />
        <Route path="/dashboard" element={<Navigate to="/dashboard/admin" replace />} />
        <Route path="/dashboard/admin" element={<DashboardPage />} />
        <Route path="/dashboard/developer" element={<DeveloperDashboardPage />} />
        <Route path="/visualization" element={<SongVisualizationPage />} />
        <Route path="/" element={<Navigate to="/login" replace />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;