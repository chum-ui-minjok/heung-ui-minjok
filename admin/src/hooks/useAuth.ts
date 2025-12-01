import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { loginApi } from "../api/auth";
import { type LoginRequest } from "../types/auth";

const useMockData = import.meta.env.VITE_USE_MOCK === "true";

export const useAuth = () => {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string>("");
  const navigate = useNavigate();

  const login = async (credentials: LoginRequest) => {
    setIsLoading(true);
    setError("");

    try {
      let role: string;
      
      if (useMockData) {
        // Mock 모드: 자동 로그인
        await new Promise((resolve) => setTimeout(resolve, 500));
        localStorage.setItem("accessToken", "mock-token-12345");
        localStorage.setItem("adminId", "1"); // Mock adminId
        localStorage.setItem("adminName", credentials.username || "관리자");
        // Mock 모드에서는 기본적으로 SUPER_ADMIN으로 설정
        role = "SUPER_ADMIN";
        localStorage.setItem("adminRole", role);
      } else {
        const response = await loginApi(credentials);

        // 토큰 저장
        localStorage.setItem("accessToken", response.accessToken);
        if (response.refreshToken) {
          localStorage.setItem("refreshToken", response.refreshToken);
        }

        // adminId 저장 (userId로 올 수도 있음)
        const adminId = response.adminId || response.userId;
        if (adminId) {
          localStorage.setItem("adminId", adminId.toString());
        }

        if (response.username || credentials.username) {
          localStorage.setItem(
            "adminName",
            response.username ?? credentials.username
          );
        }
        if (response.role) {
          role = response.role;
          localStorage.setItem("adminRole", role);
        } else {
          role = "ADMIN";
          localStorage.setItem("adminRole", role);
        }
      }

      // 역할에 따라 다른 대시보드로 이동
      const isSuperAdmin = role === "SUPER_ADMIN" || role === "superadmin";
      const dashboardPath = isSuperAdmin ? "/dashboard/developer" : "/dashboard/admin";
      navigate(dashboardPath);
    } catch (err) {
      const errorMessage =
        err instanceof Error ? err.message : "로그인 중 오류가 발생했습니다.";
      setError(errorMessage);
    } finally {
      setIsLoading(false);
    }
  };

  const logout = () => {
    localStorage.removeItem("accessToken");
    localStorage.removeItem("refreshToken");
    localStorage.removeItem("adminName");
    localStorage.removeItem("adminRole");
    localStorage.removeItem("adminId");
    navigate("/login");
  };

  const isAuthenticated = () => {
    return !!localStorage.getItem("accessToken");
  };

  return {
    login,
    logout,
    isAuthenticated,
    isLoading,
    error,
  };
};
