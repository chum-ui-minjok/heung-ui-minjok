import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import LoginPage from "./pages/LoginPage";
import EmergencyReportPage from "./pages/EmergencyReportPage";
import UserManagementPage from "./pages/UserManagementPage";
import SongVisualizationPage from "./pages/SongVisualizationPage";
import DeveloperDashboardPage from "./pages/DeveloperDashboardPage";
import SimpleSongUploadPage from "./pages/SimpleSongUploadPage";
import DeviceRegisterPage from "./pages/DeviceRegisterPage";
import UserRegisterPage from "./pages/UserRegisterPage";
import AdminManagementPage from "./pages/AdminManagementPage";
import DashboardRedirect from "./components/DashboardRedirect";

function App() {
  return (
    <BrowserRouter basename="/admin">
      <Routes>
        <Route path="/login" element={<LoginPage />} />
        <Route path="/dashboard" element={<DashboardRedirect />} />
        <Route
          path="/dashboard/admin"
          element={<Navigate to="/dashboard/admin/emergencies" replace />}
        />
        <Route
          path="/dashboard/admin/emergencies"
          element={<EmergencyReportPage />}
        />
        <Route path="/dashboard/admin/users" element={<UserManagementPage />} />
        <Route
          path="/dashboard/admin/device-register"
          element={<DeviceRegisterPage />}
        />
        <Route
          path="/dashboard/admin/user-register"
          element={<UserRegisterPage />}
        />
        <Route
          path="/dashboard/developer"
          element={<DeveloperDashboardPage />}
        />
        <Route
          path="/dashboard/developer/song-upload"
          element={<SimpleSongUploadPage />}
        />
        <Route
          path="/dashboard/developer/visualization"
          element={<SongVisualizationPage />}
        />
        <Route
          path="/dashboard/developer/admin-management"
          element={<AdminManagementPage />}
        />
        <Route path="/" element={<Navigate to="/login" replace />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
