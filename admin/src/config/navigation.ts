import {
  cilAudio,
  cilCloudUpload,
  cilCode,
  cilSpeedometer,
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
    to: "/visualization",
    label: "곡 시각화",
    description: "악보/모션 미리보기",
    icon: cilAudio,
  },
];

export const developerBaseNavItems: NavigationItem[] = [
  {
    to: "/dashboard/developer",
    label: "개발자 페이지",
    description: "곡 도구와 자료",
    icon: cilCode,
  },
  ...sharedSongNavItems,
];

export const createQuickRegisterNavItem = (
  onClick: () => void
): NavigationItem => ({
  label: "곡 간편 등록",
  description: "새로운 곡을 빠르게 등록합니다",
  icon: cilCloudUpload,
  onClick,
});
