import { useCallback, useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  CCol,
  CContainer,
  CRow,
  CTable,
  CTableBody,
  CTableDataCell,
  CTableHead,
  CTableHeaderCell,
  CTableRow,
} from "@coreui/react";
import { Button, Badge } from "../components";
import DashboardHeader from "../components/DashboardHeader";
import AdminLayout from "../layouts/AdminLayout";
import {
  adminBaseNavItems,
  deviceRegisterNavItem,
  userRegisterNavItem,
} from "../config/navigation";
import { useUserStore, useNotificationStore, useDeviceStore } from "../stores";
import { getUsers } from "../api/user";
import { mockUsers, mockDevices } from "../mocks/mockData";
import "../styles/dashboard.css";
import "../styles/emergency-report-table.css";

const useMockData = import.meta.env.VITE_USE_MOCK === "true";

// 사용자와 기기 정보를 결합한 타입
interface UserWithDevice {
  id: number;
  name: string;
  deviceId: number;
  deviceSerialNumber?: string;
  deviceLocation?: string;
  status: "ACTIVE" | "WARNING" | "EMERGENCY";
  lastActivity?: string;
  createdAt: string;
}

const UserManagementPage = () => {
  const navigate = useNavigate();

  const users = useUserStore((state) => state.users);
  const setUsers = useUserStore((state) => state.setUsers);
  const setLoadingUsers = useUserStore((state) => state.setLoading);
  const isLoadingUsers = useUserStore((state) => state.isLoading);

  const devices = useDeviceStore((state) => state.devices);
  const setDevices = useDeviceStore((state) => state.setDevices);

  const clearUnread = useNotificationStore((state) => state.clearUnread);

  // 필터링/검색 상태
  const [statusFilters, setStatusFilters] = useState<Set<string>>(
    new Set(["all"])
  );
  const [searchField, setSearchField] = useState<string>("all");
  const [searchQuery, setSearchQuery] = useState<string>("");

  // 페이지네이션 상태
  const [displayCount, setDisplayCount] = useState<number>(10);

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
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    // 토큰 확인
    const token = localStorage.getItem("accessToken");
    if (!token) {
      navigate("/login");
      return;
    }

    loadUsers();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // 알림 아이콘 클릭
  const handleNotificationClick = () => {
    clearUnread();
  };

  const navigationItems = useMemo(
    () => [...adminBaseNavItems, deviceRegisterNavItem, userRegisterNavItem],
    []
  );

  // 사용자와 기기 정보 결합
  const usersWithDevices = useMemo(() => {
    return users.map((user) => {
      const device = devices.find((d) => d.id === user.deviceId);
      return {
        id: user.id,
        name: user.name,
        deviceId: user.deviceId,
        deviceSerialNumber: device?.serialNumber,
        deviceLocation: device?.location || user.location,
        status: user.status,
        lastActivity: user.lastActivity,
        createdAt: user.createdAt,
      } as UserWithDevice;
    });
  }, [users, devices]);

  // 상태 필터 체크박스 핸들러
  const handleStatusFilterChange = (status: string) => {
    setStatusFilters((prev) => {
      const newFilters = new Set(prev);
      if (status === "all") {
        if (newFilters.has("all")) {
          newFilters.clear();
          newFilters.add("all");
        } else {
          newFilters.clear();
          newFilters.add("all");
        }
      } else {
        newFilters.delete("all");
        if (newFilters.has(status)) {
          newFilters.delete(status);
          if (newFilters.size === 0) {
            newFilters.add("all");
          }
        } else {
          newFilters.add(status);
        }
      }
      return newFilters;
    });
  };

  // 필터링 및 검색된 사용자 목록
  const filteredUsers = useMemo(() => {
    return usersWithDevices.filter((user) => {
      // 상태 필터링 (체크박스)
      if (!statusFilters.has("all") && !statusFilters.has(user.status)) {
        return false;
      }

      // 검색어 필터링
      if (searchQuery.trim()) {
        const query = searchQuery.toLowerCase();
        let matches = false;

        if (searchField === "all") {
          matches =
            user.name.toLowerCase().includes(query) ||
            user.deviceSerialNumber?.toLowerCase().includes(query) ||
            user.deviceLocation?.toLowerCase().includes(query) ||
            false;
        } else if (searchField === "name") {
          matches = user.name.toLowerCase().includes(query);
        } else if (searchField === "deviceSerialNumber") {
          matches =
            user.deviceSerialNumber?.toLowerCase().includes(query) || false;
        } else if (searchField === "deviceLocation") {
          matches = user.deviceLocation?.toLowerCase().includes(query) || false;
        }

        if (!matches) {
          return false;
        }
      }

      return true;
    });
  }, [usersWithDevices, statusFilters, searchField, searchQuery]);

  // 필터가 변경되면 표시 개수 초기화
  useEffect(() => {
    setDisplayCount(10);
  }, [statusFilters, searchField, searchQuery]);

  // 표시할 사용자 목록 (페이지네이션 적용)
  const displayedUsers = useMemo(() => {
    return filteredUsers.slice(0, displayCount);
  }, [filteredUsers, displayCount]);

  // 더보기 버튼 표시 여부
  const hasMore = filteredUsers.length > displayCount;

  // 더보기 버튼 핸들러
  const handleLoadMore = () => {
    setDisplayCount((prev: number) => prev + 10);
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleString("ko-KR", {
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  return (
    <AdminLayout navItems={navigationItems}>
      <CContainer fluid className="py-4">
        <DashboardHeader
          title="사용자 관리"
          onNotificationClick={handleNotificationClick}
        />

        {/* 필터링/검색 컨테이너 */}
        <CRow className="g-4 mb-3">
          <CCol xs={12}>
            <div className="emergency-filter-container">
              {/* 검색옵션 */}
              <div className="filter-section">
                <div className="filter-label">검색옵션</div>
                <div className="filter-options">
                  <label className="filter-checkbox">
                    <input
                      type="checkbox"
                      checked={statusFilters.has("all")}
                      onChange={() => handleStatusFilterChange("all")}
                    />
                    <span>전체</span>
                  </label>
                  <label className="filter-checkbox">
                    <input
                      type="checkbox"
                      checked={statusFilters.has("ACTIVE")}
                      onChange={() => handleStatusFilterChange("ACTIVE")}
                    />
                    <span>정상</span>
                  </label>
                  <label className="filter-checkbox">
                    <input
                      type="checkbox"
                      checked={statusFilters.has("WARNING")}
                      onChange={() => handleStatusFilterChange("WARNING")}
                    />
                    <span>주의</span>
                  </label>
                  <label className="filter-checkbox">
                    <input
                      type="checkbox"
                      checked={statusFilters.has("EMERGENCY")}
                      onChange={() => handleStatusFilterChange("EMERGENCY")}
                    />
                    <span>긴급</span>
                  </label>
                </div>
              </div>

              {/* 검색명 */}
              <div className="filter-section">
                <div className="filter-label">검색명</div>
                <div className="filter-search">
                  <div className="search-combined">
                    <select
                      value={searchField}
                      onChange={(e) => setSearchField(e.target.value)}
                      className="search-field-dropdown"
                    >
                      <option value="all">전체</option>
                      <option value="name">이름</option>
                      <option value="deviceSerialNumber">기기 일련번호</option>
                      <option value="deviceLocation">기기 위치</option>
                    </select>
                    <input
                      type="text"
                      placeholder="검색어를 입력해주세요."
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      className="search-input-combined"
                    />
                  </div>
                  <Button
                    variant="primary"
                    onClick={() => {}}
                    className="search-button"
                  >
                    검색
                  </Button>
                </div>
              </div>
            </div>
          </CCol>
        </CRow>

        {/* 결과 건수 표시 */}
        {!isLoadingUsers && users.length > 0 && (
          <CRow className="mb-2">
            <CCol xs={12}>
              <div className="filter-results-count">
                전체 {filteredUsers.length}건
                {filteredUsers.length > 10 && (
                  <span className="ms-2 text-muted">
                    ({displayedUsers.length}건 표시)
                  </span>
                )}
              </div>
            </CCol>
          </CRow>
        )}

        <CRow className="g-4">
          <CCol xs={12}>
            <div className="emergency-report-container">
              {isLoadingUsers ? (
                <div className="text-center py-5">
                  <div className="spinner-border" role="status">
                    <span className="visually-hidden">로딩 중...</span>
                  </div>
                  <p className="mt-2">사용자 목록 불러오는 중...</p>
                </div>
              ) : filteredUsers.length === 0 ? (
                <div className="text-center py-5">
                  <div className="mb-3" style={{ fontSize: "3rem" }}>
                    📋
                  </div>
                  <p>필터 조건에 맞는 사용자가 없습니다.</p>
                </div>
              ) : (
                <>
                  <CTable hover responsive className="emergency-report-table">
                    <CTableHead>
                      <CTableRow>
                        <CTableHeaderCell>번호</CTableHeaderCell>
                        <CTableHeaderCell>이름</CTableHeaderCell>
                        <CTableHeaderCell>기기 일련번호</CTableHeaderCell>
                        <CTableHeaderCell>기기 위치</CTableHeaderCell>
                        <CTableHeaderCell>상태</CTableHeaderCell>
                        <CTableHeaderCell>마지막 활동</CTableHeaderCell>
                        <CTableHeaderCell>등록일</CTableHeaderCell>
                      </CTableRow>
                    </CTableHead>
                    <CTableBody>
                      {displayedUsers.map((user) => {
                        return (
                          <CTableRow key={user.id}>
                            <CTableDataCell>
                              {filteredUsers.indexOf(user) + 1}
                            </CTableDataCell>
                            <CTableDataCell>{user.name}</CTableDataCell>
                            <CTableDataCell>
                              {user.deviceSerialNumber || "-"}
                            </CTableDataCell>
                            <CTableDataCell>
                              {user.deviceLocation || "-"}
                            </CTableDataCell>
                            <CTableDataCell>
                              <Badge status={user.status} />
                            </CTableDataCell>
                            <CTableDataCell>
                              {user.lastActivity
                                ? formatDate(user.lastActivity)
                                : "-"}
                            </CTableDataCell>
                            <CTableDataCell>
                              {formatDate(user.createdAt)}
                            </CTableDataCell>
                          </CTableRow>
                        );
                      })}
                    </CTableBody>
                  </CTable>
                  {hasMore && (
                    <div className="text-center mt-3">
                      <Button
                        variant="secondary"
                        onClick={handleLoadMore}
                        className="load-more-btn"
                      >
                        더 보기 ({Math.ceil(displayCount / 10)}/
                        {Math.ceil(filteredUsers.length / 10)})
                      </Button>
                    </div>
                  )}
                </>
              )}
            </div>
          </CCol>
        </CRow>
      </CContainer>
    </AdminLayout>
  );
};

export default UserManagementPage;
