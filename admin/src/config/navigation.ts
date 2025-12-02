import {
  cilAudio,
  cilBell,
  cilCloudUpload,
  cilMobile,
  cilPeople,
  cilSpeedometer,
  cilUser,
  cilUserPlus,
} from "@coreui/icons";

export interface NavigationItem {
  to?: string;
  label: string;
  description?: string;
  icon?: string | string[];
  onClick?: () => void;
}

export const adminBaseNavItems: NavigationItem[] = [
  {
    to: "/dashboard/admin",
    label: "관리자 대시보드",
    description: "신고 현황과 기기 관리",
    icon: cilSpeedometer,
  },
];

export const sharedSongNavItems: NavigationItem[] = [
  {
    to: "/dashboard/developer/visualization",
    label: "곡 시각화",
    description: "악보/모션 미리보기",
    icon: cilAudio,
  },
];

export const developerBaseNavItems: NavigationItem[] = [...sharedSongNavItems];

export const quickRegisterNavItem: NavigationItem = {
  to: "/dashboard/developer/song-upload",
  label: "곡 간편 등록",
  description: "새로운 곡을 빠르게 등록합니다",
  icon: cilCloudUpload,
};

export const deviceRegisterNavItem: NavigationItem = {
  to: "/dashboard/admin/device-register",
  label: "기기 등록",
  description: "새 기기를 등록합니다",
  icon: cilMobile,
};

export const userRegisterNavItem: NavigationItem = {
  to: "/dashboard/admin/user-register",
  label: "어르신 등록",
  description: "새로운 사용자를 등록합니다",
  icon: cilUserPlus,
};

export const adminManagementNavItem: NavigationItem = {
  to: "/dashboard/developer/admin-management",
  label: "관리자 관리",
  description: "관리자 생성 및 관리",
  icon: cilPeople,
};

// 대시보드 섹션 네비게이션 항목
export const emergencyReportNavItem: NavigationItem = {
  to: "/dashboard/admin/emergencies",
  label: "신고 현황",
  description: "응급 신고 및 현황 관리",
  icon: cilBell,
};

export const userManagementNavItem: NavigationItem = {
  to: "/dashboard/admin/users",
  label: "사용자 관리",
  description: "기기 및 사용자 관리",
  icon: cilUser,
};
