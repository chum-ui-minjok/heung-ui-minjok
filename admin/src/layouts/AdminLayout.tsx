import { type ReactNode, useEffect, useState, useMemo } from "react";
import { NavLink, useLocation } from "react-router-dom";
import { CNav, CNavItem } from "@coreui/react";
import CIcon from "@coreui/icons-react";
import type { NavigationItem } from "../config/navigation";
import {
  emergencyReportNavItem,
  userManagementNavItem,
  sharedSongNavItems,
  quickRegisterNavItem,
  adminManagementNavItem,
} from "../config/navigation";
import { useAuth } from "../hooks/useAuth";
import "../styles/admin-layout.css";

interface AdminLayoutProps {
  children: ReactNode;
  navItems: NavigationItem[];
}

const AdminLayout = ({ children, navItems }: AdminLayoutProps) => {
  const { logout } = useAuth();
  const location = useLocation();
  const [adminName, setAdminName] = useState("관리자");
  const [adminRole, setAdminRole] = useState("ADMIN");

  useEffect(() => {
    const syncProfile = () => {
      const storedName = localStorage.getItem("adminName");
      const storedRole = localStorage.getItem("adminRole");
      if (storedName) {
        setAdminName(storedName);
      }
      if (storedRole) {
        setAdminRole(storedRole);
      }
    };

    syncProfile();
    window.addEventListener("storage", syncProfile);
    return () => window.removeEventListener("storage", syncProfile);
  }, []);

  // 현재 경로에 따라 페이지 타입 결정
  const isDeveloperPage = location.pathname.startsWith("/dashboard/developer");
  const isAdminPage = location.pathname.startsWith("/dashboard/admin");
  const pageType = isDeveloperPage ? "개발자 페이지" : "관리자 페이지";

  // 관리자 페이지용 상단 네비게이션 항목
  const adminTopNavItems = [
    {
      to: "/dashboard/admin/emergencies",
      label: "대시보드",
    },
    {
      to: "/dashboard/admin/device-register",
      label: "등록/관리",
    },
  ];

  // 개발자 페이지용 상단 네비게이션 항목
  const developerTopNavItems = [
    {
      to: "/dashboard/developer/visualization",
      label: "곡 관리",
    },
    {
      to: "/dashboard/developer/admin-management",
      label: "관리자 관리",
    },
  ];

  // 사이드 네비게이션에 표시할 항목 결정
  const sidebarNavItems = useMemo(() => {
    // 관리자 페이지에서 '대시보드' 탭 관련 경로인 경우
    const isDashboardSection =
      location.pathname === "/dashboard/admin/emergencies" ||
      location.pathname === "/dashboard/admin/users";

    // '등록/관리' 섹션인 경우
    const isRegisterSection =
      location.pathname.startsWith("/dashboard/admin/device-register") ||
      location.pathname.startsWith("/dashboard/admin/user-register");

    // 개발자 페이지에서 '곡 관리' 섹션인 경우
    const isSongManagementSection =
      location.pathname === "/dashboard/developer" ||
      location.pathname === "/dashboard/developer/visualization" ||
      location.pathname === "/dashboard/developer/song-upload";

    if (isAdminPage && isDashboardSection) {
      return [emergencyReportNavItem, userManagementNavItem];
    }

    // '등록/관리' 섹션일 때는 '관리자 대시보드' 제외
    if (isAdminPage && isRegisterSection) {
      return navItems.filter(
        (item) => item.to !== "/dashboard/admin/emergencies"
      );
    }

    // 개발자 페이지에서 '곡 관리' 섹션인 경우
    if (isDeveloperPage && isSongManagementSection) {
      return [...sharedSongNavItems, quickRegisterNavItem];
    }

    // 개발자 페이지에서 '관리자 관리' 섹션인 경우
    if (
      isDeveloperPage &&
      location.pathname.startsWith("/dashboard/developer/admin-management")
    ) {
      return [adminManagementNavItem];
    }

    // 그 외의 경우는 기존 navItems 사용
    return navItems;
  }, [isAdminPage, isDeveloperPage, location.pathname, navItems]);

  return (
    <div className="admin-layout">
      <header className="admin-layout__topbar">
        <div className="admin-layout__brand">
          <img
            src={`${import.meta.env.BASE_URL}logo.svg`}
            alt="로고"
            className="admin-layout__logo"
          />
          <span>흥의민족</span>
          <span className="admin-layout__page-type">{pageType}</span>
        </div>
        <div className="admin-layout__topnav">
          {isAdminPage
            ? // 관리자 페이지: '대시보드', '등록/관리' 두 개만 표시
              adminTopNavItems.map((item) => {
                const isActive =
                  item.to === "/dashboard/admin/emergencies"
                    ? location.pathname === "/dashboard/admin/emergencies" ||
                      location.pathname === "/dashboard/admin/users"
                    : location.pathname.startsWith(
                        "/dashboard/admin/device-register"
                      ) ||
                      location.pathname.startsWith(
                        "/dashboard/admin/user-register"
                      );

                return (
                  <NavLink
                    key={item.to}
                    to={item.to}
                    className={`admin-layout__topnav-item ${
                      isActive ? "active" : ""
                    }`}
                  >
                    <span>{item.label}</span>
                  </NavLink>
                );
              })
            : // 개발자 페이지: '곡 관리', '관리자 관리' 두 개만 표시
              developerTopNavItems.map((item) => {
                const isActive =
                  item.to === "/dashboard/developer/visualization"
                    ? location.pathname === "/dashboard/developer" ||
                      location.pathname ===
                        "/dashboard/developer/visualization" ||
                      location.pathname === "/dashboard/developer/song-upload"
                    : location.pathname.startsWith(
                        "/dashboard/developer/admin-management"
                      );

                return (
                  <NavLink
                    key={item.to}
                    to={item.to}
                    className={`admin-layout__topnav-item ${
                      isActive ? "active" : ""
                    }`}
                  >
                    <span>{item.label}</span>
                  </NavLink>
                );
              })}
        </div>
        <div className="admin-layout__profile">
          <div className="admin-layout__profile-info">
            <span className="admin-layout__profile-name">{adminName}</span>
            <span className="admin-layout__profile-role">{adminRole}</span>
          </div>
          <div className="admin-layout__profile-divider" />
          <button
            type="button"
            className="admin-layout__logout"
            onClick={logout}
          >
            로그아웃
          </button>
        </div>
      </header>

      <div className="admin-layout__content">
        <aside className="admin-layout__sidebar">
          <CNav className="flex-column gap-2">
            {sidebarNavItems.map((item, index) => (
              <CNavItem key={item.to ?? `action-${index}`}>
                {item.to ? (
                  <NavLink
                    to={item.to}
                    className={({ isActive }) => {
                      // 특정 경로는 정확히 일치할 때만 활성화
                      const exactMatchPaths = ["/dashboard/developer"];
                      const isExactMatch = exactMatchPaths.includes(
                        item.to || ""
                      )
                        ? location.pathname === item.to
                        : isActive;
                      return `admin-layout__nav-link ${
                        isExactMatch ? "active" : ""
                      }`;
                    }}
                  >
                    {item.icon && (
                      <CIcon
                        icon={item.icon}
                        className="admin-layout__nav-icon"
                      />
                    )}
                    <div className="admin-layout__nav-text">
                      <span className="admin-layout__nav-title">
                        {item.label}
                      </span>
                      {item.description && (
                        <span className="admin-layout__nav-desc">
                          {item.description}
                        </span>
                      )}
                    </div>
                  </NavLink>
                ) : (
                  <button
                    type="button"
                    className="admin-layout__nav-link admin-layout__nav-link--action"
                    onClick={item.onClick}
                  >
                    {item.icon && (
                      <CIcon
                        icon={item.icon}
                        className="admin-layout__nav-icon"
                      />
                    )}
                    <div className="admin-layout__nav-text">
                      <span className="admin-layout__nav-title">
                        {item.label}
                      </span>
                      {item.description && (
                        <span className="admin-layout__nav-desc">
                          {item.description}
                        </span>
                      )}
                    </div>
                  </button>
                )}
              </CNavItem>
            ))}
          </CNav>
        </aside>

        <main className="admin-layout__main">{children}</main>
      </div>
    </div>
  );
};

export default AdminLayout;
