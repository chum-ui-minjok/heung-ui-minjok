import { useEffect, useMemo } from "react";
import { useNavigate } from "react-router-dom";
import {
  CCard,
  CCardBody,
  CCardHeader,
  CCol,
  CContainer,
  CRow,
} from "@coreui/react";
import DashboardHeader from "../components/DashboardHeader";
import { useNotificationStore } from "../stores";
import "../styles/dashboard.css";
import AdminLayout from "../layouts/AdminLayout";
import {
  quickRegisterNavItem,
  developerBaseNavItems,
  adminManagementNavItem,
} from "../config/navigation";

const DeveloperDashboardPage = () => {
  const navigate = useNavigate();
  const clearUnread = useNotificationStore((state) => state.clearUnread);

  useEffect(() => {
    const token = localStorage.getItem("accessToken");
    if (!token) {
      navigate("/login");
    }
  }, [navigate]);

  const handleNotificationClick = () => {
    clearUnread();
  };

  const navigationItems = useMemo(
    () => [
      ...developerBaseNavItems,
      quickRegisterNavItem,
      adminManagementNavItem,
    ],
    []
  );

  return (
    <AdminLayout navItems={navigationItems}>
      <CContainer fluid className="py-4">
        <DashboardHeader
          title="흥부자 개발자 페이지"
          onNotificationClick={handleNotificationClick}
        />

        <CRow className="g-4">
          <CCol xs={12} lg={6}>
            <CCard className="h-100">
              <CCardHeader className="fw-semibold">
                📘 가이드 & 참고 링크
              </CCardHeader>
              <CCardBody className="text-body-secondary">
                <p className="mb-2">
                  - 곡 데이터는 등록 후 자동으로 대시보드에 반영됩니다.
                </p>
                <p className="mb-2">
                  - 시각화 페이지에서 실시간 악보/모션 데이터를 확인할 수
                  있습니다.
                </p>
                <p className="mb-0">
                  - 추가 도구가 필요하면 팀 내 개발 채널에 요청해주세요.
                </p>
              </CCardBody>
            </CCard>
          </CCol>
        </CRow>
      </CContainer>
    </AdminLayout>
  );
};

export default DeveloperDashboardPage;
