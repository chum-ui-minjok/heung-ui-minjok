import { type ReactNode, useEffect, useState } from "react";
import { NavLink } from "react-router-dom";
import { CNav, CNavItem } from "@coreui/react";
import CIcon from "@coreui/icons-react";
import type { NavigationItem } from "../config/navigation";
import { useAuth } from "../hooks/useAuth";
import "../styles/admin-layout.css";

interface AdminLayoutProps {
  children: ReactNode;
  navItems: NavigationItem[];
}

const AdminLayout = ({ children, navItems }: AdminLayoutProps) => {
  const { logout } = useAuth();
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

  return (
    <div className="admin-layout">
      <aside className="admin-layout__sidebar">
        <h1 className="admin-layout__brand">
          <span role="img" aria-label="logo">
            🎵
          </span>
          흥부자
        </h1>

        <CNav className="flex-column gap-2">
          {navItems.map((item, index) => (
            <CNavItem key={item.to ?? `action-${index}`}>
              {item.to ? (
                <NavLink
                  to={item.to}
                  className={({ isActive }) =>
                    `admin-layout__nav-link ${isActive ? "active" : ""}`
                  }
                >
                  {item.icon && (
                    <CIcon icon={item.icon} size="lg" className="me-3" />
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
                    <CIcon icon={item.icon} size="lg" className="me-3" />
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

      <div className="admin-layout__body">
        <header className="admin-layout__topbar">
          <div className="admin-layout__topnav">
            {navItems.map((item, index) =>
              item.to ? (
                <NavLink
                  key={`top-${item.to}`}
                  to={item.to}
                  className={({ isActive }) =>
                    `admin-layout__topnav-item ${isActive ? "active" : ""}`
                  }
                >
                  <span>{item.label}</span>
                </NavLink>
              ) : (
                <button
                  type="button"
                  key={`top-action-${index}`}
                  className="admin-layout__topnav-item"
                  onClick={item.onClick}
                >
                  <span>{item.label}</span>
                </button>
              )
            )}
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

        <main className="admin-layout__main">{children}</main>
      </div>
    </div>
  );
};

export default AdminLayout;
