import { useCallback, useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  CBadge,
  CCard,
  CCardBody,
  CCardHeader,
  CCol,
  CContainer,
  CRow,
} from "@coreui/react";
import { cilMobile, cilUserPlus } from "@coreui/icons";
import {
  Button,
  EmergencyList,
  WebSocketStatus,
  DeviceRegisterModal,
  UserRegisterModal,
  EmergencyAlertModal,
  DeviceUserGrid,
} from "../components";
import DashboardHeader from "../components/DashboardHeader";
import AdminLayout from "../layouts/AdminLayout";
import { adminBaseNavItems } from "../config/navigation";
import { useWebSocket } from "../hooks/useWebSocket";
import {
  useEmergencyStore,
  useUserStore,
  useNotificationStore,
  useDeviceStore,
} from "../stores";
import { getEmergencyReports, resolveEmergency } from "../api/emergency";
import { getUsers } from "../api/user";
import { type EmergencyReport } from "../types/emergency";
import {
  mockEmergencyReports,
  mockUsers,
  mockDevices,
} from "../mocks/mockData";
import "../styles/dashboard.css";
import "../styles/device-user.css";

const useMockData = import.meta.env.VITE_USE_MOCK === "true";

const DashboardPage = () => {
  const navigate = useNavigate();

  // 모달 상태
  const [isDeviceModalOpen, setIsDeviceModalOpen] = useState(false);
  const [isUserModalOpen, setIsUserModalOpen] = useState(false);
  const [isEmergencyAlertOpen, setIsEmergencyAlertOpen] = useState(false);
  const [currentEmergencyAlert, setCurrentEmergencyAlert] =
    useState<EmergencyReport | null>(null);

  // 응급 신고 더보기 상태
  const [showAllEmergencies, setShowAllEmergencies] = useState(false);

  // 스토어
  const reports = useEmergencyStore((state) => state.reports);
  const setReports = useEmergencyStore((state) => state.setReports);
  const updateReport = useEmergencyStore((state) => state.updateReport);
  const isLoadingReports = useEmergencyStore((state) => state.isLoading);
  const setLoadingReports = useEmergencyStore((state) => state.setLoading);

  const setUsers = useUserStore((state) => state.setUsers);
  const setLoadingUsers = useUserStore((state) => state.setLoading);

  const setDevices = useDeviceStore((state) => state.setDevices);

  const clearUnread = useNotificationStore((state) => state.clearUnread);

  // WebSocket 연결
  const { isConnected, isConnecting, connect } = useWebSocket({
    onConnect: () => {
      console.log("✅ Dashboard: WebSocket connected");
    },
    onDisconnect: () => {
      console.log("❌ Dashboard: WebSocket disconnected");
    },
  });

  // 초기 데이터 로드
  // 신고 목록 로드
  const loadEmergencyReports = useCallback(async () => {
    setLoadingReports(true);
    try {
      if (useMockData) {
        // Mock 데이터 사용
        await new Promise((resolve) => setTimeout(resolve, 500)); // 로딩 시뮬레이션
        setReports(mockEmergencyReports);
      } else {
        const data = await getEmergencyReports();
        setReports(data);
      }
    } catch (error) {
      console.error("신고 목록 로드 실패:", error);
    } finally {
      setLoadingReports(false);
    }
  }, [setLoadingReports, setReports]);

  // 어르신 목록 로드
  const loadUsers = useCallback(async () => {
    setLoadingUsers(true);
    try {
      if (useMockData) {
        // Mock 데이터 사용
        await new Promise((resolve) => setTimeout(resolve, 500)); // 로딩 시뮬레이션
        setUsers(mockUsers);
        setDevices(mockDevices);
      } else {
        const data = await getUsers();
        setUsers(data);
      }
    } catch (error) {
      console.error("어르신 목록 로드 실패:", error);
    } finally {
      setLoadingUsers(false);
    }
  }, [setDevices, setLoadingUsers, setUsers]);

  const loadDashboardData = useCallback(async () => {
    await Promise.all([loadEmergencyReports(), loadUsers()]);
  }, [loadEmergencyReports, loadUsers]);

  useEffect(() => {
    // 토큰 확인
    const token = localStorage.getItem("accessToken");
    if (!token) {
      navigate("/login");
      return;
    }

    loadDashboardData();

    // Mock 모드가 아닐 때만 WebSocket 연결
    if (!useMockData) {
      connect();
    }
  }, [connect, loadDashboardData, navigate]);

  // 신고 처리
  const handleResolveEmergency = useCallback(
    async (reportId: number) => {
      try {
        if (useMockData) {
          // Mock 모드: 상태만 업데이트
          await new Promise((resolve) => setTimeout(resolve, 500));
          updateReport(reportId, {
            status: "RESOLVED",
          });
        } else {
          const updatedReport = await resolveEmergency(reportId);
          // 백엔드에서 받은 업데이트된 신고 정보로 상태 갱신
          updateReport(reportId, updatedReport);
        }
      } catch (error) {
        console.error("신고 처리 실패:", error);
        alert("신고 처리에 실패했습니다.");
      }
    },
    [updateReport]
  );

  // 알림 아이콘 클릭
  const handleNotificationClick = () => {
    clearUnread();
  };

  // 긴급 신고 알림 (WebSocket을 통해 새 신고가 들어오면 자동으로 처리됨)
  useEffect(() => {
    // 가장 최근 PENDING/CONFIRMED 신고가 있으면 알림 표시
    const latestEmergency = reports.find((r) => r.status === "CONFIRMED");

    if (latestEmergency && !currentEmergencyAlert) {
      setCurrentEmergencyAlert(latestEmergency);
      setIsEmergencyAlertOpen(true);
    }
  }, [currentEmergencyAlert, reports]);

  const activeEmergencyCount = useMemo(
    () => reports.filter((report) => report.status !== "RESOLVED").length,
    [reports]
  );

  const navigationItems = useMemo(
    () => [
      ...adminBaseNavItems,
      {
        label: "기기 등록",
        description: "새 기기를 등록합니다",
        icon: cilMobile,
        onClick: () => setIsDeviceModalOpen(true),
      },
      {
        label: "어르신 등록",
        description: "새로운 사용자를 등록합니다",
        icon: cilUserPlus,
        onClick: () => setIsUserModalOpen(true),
      },
    ],
    [setIsDeviceModalOpen, setIsUserModalOpen]
  );

  return (
    <AdminLayout navItems={navigationItems}>
      <CContainer fluid className="py-4">
        <DashboardHeader
          title="흥부자 관리자 대시보드"
          onNotificationClick={handleNotificationClick}
        />

        <CRow className="g-4">
          <CCol xs={12} xl={8}>
            <CCard className="h-100">
              <CCardHeader className="d-flex justify-content-between align-items-center">
                <span className="fw-semibold">
                  📊 실시간 신고 리스트
                  {!isLoadingReports && reports.length > 0 && (
                    <CBadge color="danger" className="ms-2">
                      {activeEmergencyCount} 건 진행 중
                    </CBadge>
                  )}
                </span>
                {reports.length > 4 && (
                  <Button
                    variant="secondary"
                    onClick={() => setShowAllEmergencies(!showAllEmergencies)}
                  >
                    {showAllEmergencies ? "최근 4건만 보기" : "전체 보기"}
                  </Button>
                )}
              </CCardHeader>
              <CCardBody>
                <EmergencyList
                  reports={showAllEmergencies ? reports : reports.slice(0, 4)}
                  onResolve={handleResolveEmergency}
                  isLoading={isLoadingReports}
                />
              </CCardBody>
            </CCard>
          </CCol>

          <CCol xs={12} xl={4}>
            <CCard className="h-100">
              <CCardHeader className="fw-semibold">
                🔌 실시간 연결 상태
              </CCardHeader>
              <CCardBody>
                <WebSocketStatus
                  isConnected={isConnected}
                  isConnecting={isConnecting}
                />
              </CCardBody>
            </CCard>
          </CCol>
        </CRow>

        <CRow className="g-4">
          <CCol xs={12}>
            <CCard>
              <CCardHeader className="d-flex justify-content-between align-items-center">
                <span className="fw-semibold">📱 기기 및 사용자 관리</span>
                <small className="text-body-secondary">
                  기기-사용자 관계 및 활동 현황
                </small>
              </CCardHeader>
              <CCardBody>
                <DeviceUserGrid />
              </CCardBody>
            </CCard>
          </CCol>
        </CRow>

        <DeviceRegisterModal
          isOpen={isDeviceModalOpen}
          onClose={() => setIsDeviceModalOpen(false)}
        />

        <UserRegisterModal
          isOpen={isUserModalOpen}
          onClose={() => {
            setIsUserModalOpen(false);
            loadUsers(); // 어르신 등록 후 목록 새로고침
          }}
        />

        <EmergencyAlertModal
          isOpen={isEmergencyAlertOpen}
          onClose={() => {
            setIsEmergencyAlertOpen(false);
            setCurrentEmergencyAlert(null);
          }}
          report={currentEmergencyAlert}
          onAcknowledge={(reportId) => {
            console.log("Emergency acknowledged:", reportId);
          }}
        />
      </CContainer>
    </AdminLayout>
  );
};

export default DashboardPage;
