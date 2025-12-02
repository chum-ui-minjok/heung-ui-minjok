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
import { Button, EmergencyAlertModal, Badge } from "../components";
import WebSocketStatus from "../components/WebSocketStatus";
import DashboardHeader from "../components/DashboardHeader";
import AdminLayout from "../layouts/AdminLayout";
import {
  adminBaseNavItems,
  deviceRegisterNavItem,
  userRegisterNavItem,
} from "../config/navigation";
import { useWebSocket } from "../hooks/useWebSocket";
import { useEmergencyStore, useNotificationStore } from "../stores";
import { getEmergencyReports, resolveEmergency } from "../api/emergency";
import { type EmergencyReport } from "../types/emergency";
import { mockEmergencyReports } from "../mocks/mockData";
import "../styles/dashboard.css";
import "../styles/emergency-report-table.css";

const useMockData = import.meta.env.VITE_USE_MOCK === "true";

const EmergencyReportPage = () => {
  const navigate = useNavigate();

  // 모달 상태
  const [isEmergencyAlertOpen, setIsEmergencyAlertOpen] = useState(false);
  const [currentEmergencyAlert, setCurrentEmergencyAlert] =
    useState<EmergencyReport | null>(null);
  // 이미 확인한 신고 ID 목록 (모달이 다시 열리지 않도록)
  const [acknowledgedReportIds, setAcknowledgedReportIds] = useState<
    Set<number>
  >(new Set());

  // 처리 중인 신고 ID
  const [resolvingId, setResolvingId] = useState<number | null>(null);

  // 필터링/검색 상태
  const [statusFilters, setStatusFilters] = useState<Set<string>>(
    new Set(["all"])
  );
  const [searchField, setSearchField] = useState<string>("all");
  const [searchQuery, setSearchQuery] = useState<string>("");

  // 페이지네이션 상태
  const [displayCount, setDisplayCount] = useState<number>(10);

  // 스토어
  const reports = useEmergencyStore((state) => state.reports);
  const setReports = useEmergencyStore((state) => state.setReports);
  const updateReport = useEmergencyStore((state) => state.updateReport);
  const isLoadingReports = useEmergencyStore((state) => state.isLoading);
  const setLoadingReports = useEmergencyStore((state) => state.setLoading);

  const clearUnread = useNotificationStore((state) => state.clearUnread);

  // WebSocket 연결
  const { isConnected, isConnecting, connect } = useWebSocket({
    onConnect: () => {
      console.log("✅ EmergencyReportPage: WebSocket connected");
    },
    onDisconnect: () => {
      console.log("❌ EmergencyReportPage: WebSocket disconnected");
    },
  });

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
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    // 토큰 확인
    const token = localStorage.getItem("accessToken");
    if (!token) {
      navigate("/login");
      return;
    }

    loadEmergencyReports();

    // Mock 모드가 아닐 때만 WebSocket 연결
    if (!useMockData) {
      connect();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // 신고 처리
  const handleResolveEmergency = useCallback(
    async (reportId: number) => {
      setResolvingId(reportId);
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
      } finally {
        setResolvingId(null);
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
    // 가장 최근 PENDING/CONFIRMED 신고 중 아직 확인하지 않은 것이 있으면 알림 표시
    const latestEmergency = reports.find(
      (r) => r.status === "CONFIRMED" && !acknowledgedReportIds.has(r.reportId)
    );

    if (latestEmergency && !currentEmergencyAlert) {
      setCurrentEmergencyAlert(latestEmergency);
      setIsEmergencyAlertOpen(true);
    }
  }, [acknowledgedReportIds, currentEmergencyAlert, reports]);

  const navigationItems = useMemo(
    () => [...adminBaseNavItems, deviceRegisterNavItem, userRegisterNavItem],
    []
  );

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

  // 필터링 및 검색된 신고 목록
  const filteredReports = useMemo(() => {
    return reports.filter((report) => {
      // 상태 필터링 (체크박스)
      if (!statusFilters.has("all") && !statusFilters.has(report.status)) {
        return false;
      }

      // 검색어 필터링
      if (searchQuery.trim()) {
        const query = searchQuery.toLowerCase();
        let matches = false;

        if (searchField === "all") {
          matches =
            report.userName.toLowerCase().includes(query) ||
            report.triggerWord?.toLowerCase().includes(query) ||
            report.message?.toLowerCase().includes(query) ||
            false;
        } else if (searchField === "userName") {
          matches = report.userName.toLowerCase().includes(query);
        } else if (searchField === "triggerWord") {
          matches = report.triggerWord?.toLowerCase().includes(query) || false;
        } else if (searchField === "message") {
          matches = report.message?.toLowerCase().includes(query) || false;
        }

        if (!matches) {
          return false;
        }
      }

      return true;
    });
  }, [reports, statusFilters, searchField, searchQuery]);

  // 필터가 변경되면 표시 개수 초기화
  useEffect(() => {
    setDisplayCount(10);
  }, [statusFilters, searchField, searchQuery]);

  // 표시할 신고 목록 (페이지네이션 적용)
  const displayedReports = useMemo(() => {
    return filteredReports.slice(0, displayCount);
  }, [filteredReports, displayCount]);

  // 더보기 버튼 표시 여부
  const hasMore = filteredReports.length > displayCount;

  // 더보기 버튼 핸들러
  const handleLoadMore = () => {
    setDisplayCount((prev) => prev + 10);
  };

  return (
    <AdminLayout navItems={navigationItems}>
      <CContainer fluid className="py-4">
        <DashboardHeader
          title="신고 현황"
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
                      checked={statusFilters.has("CONFIRMED")}
                      onChange={() => handleStatusFilterChange("CONFIRMED")}
                    />
                    <span>확인됨</span>
                  </label>
                  <label className="filter-checkbox">
                    <input
                      type="checkbox"
                      checked={statusFilters.has("RESOLVED")}
                      onChange={() => handleStatusFilterChange("RESOLVED")}
                    />
                    <span>해결됨</span>
                  </label>
                  <label className="filter-checkbox">
                    <input
                      type="checkbox"
                      checked={statusFilters.has("FALSE_ALARM")}
                      onChange={() => handleStatusFilterChange("FALSE_ALARM")}
                    />
                    <span>오신고</span>
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
                      <option value="userName">어르신</option>
                      <option value="triggerWord">트리거 단어</option>
                      <option value="message">메시지</option>
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
        {!isLoadingReports && reports.length > 0 && (
          <CRow className="mb-2">
            <CCol xs={12}>
              <div className="filter-results-count">
                전체 {filteredReports.length}건
                {filteredReports.length > 10 && (
                  <span className="ms-2 text-muted">
                    ({displayedReports.length}건 표시)
                  </span>
                )}
              </div>
            </CCol>
          </CRow>
        )}

        <CRow className="g-4">
          <CCol xs={12}>
            <div className="emergency-report-container">
              {isLoadingReports ? (
                <div className="text-center py-5">
                  <div className="spinner-border" role="status">
                    <span className="visually-hidden">로딩 중...</span>
                  </div>
                  <p className="mt-2">신고 목록 불러오는 중...</p>
                </div>
              ) : filteredReports.length === 0 ? (
                <div className="text-center py-5">
                  <div className="mb-3" style={{ fontSize: "3rem" }}>
                    📋
                  </div>
                  <p>필터 조건에 맞는 신고 내역이 없습니다.</p>
                </div>
              ) : (
                <>
                  <CTable hover responsive className="emergency-report-table">
                    <CTableHead>
                      <CTableRow>
                        <CTableHeaderCell>신고 ID</CTableHeaderCell>
                        <CTableHeaderCell>어르신</CTableHeaderCell>
                        <CTableHeaderCell>신고시간</CTableHeaderCell>
                        <CTableHeaderCell>트리거 단어</CTableHeaderCell>
                        <CTableHeaderCell>메시지</CTableHeaderCell>
                        <CTableHeaderCell>확인여부</CTableHeaderCell>
                        <CTableHeaderCell>상태</CTableHeaderCell>
                        <CTableHeaderCell>작업</CTableHeaderCell>
                      </CTableRow>
                    </CTableHead>
                    <CTableBody>
                      {displayedReports.map((report) => {
                        const isResolved = report.status === "RESOLVED";
                        const isFalseAlarm = report.status === "FALSE_ALARM";
                        const isResolving = resolvingId === report.reportId;

                        return (
                          <CTableRow key={report.reportId}>
                            <CTableDataCell>#{report.reportId}</CTableDataCell>
                            <CTableDataCell>{report.userName}</CTableDataCell>
                            <CTableDataCell>
                              {formatDate(report.reportedAt)}
                            </CTableDataCell>
                            <CTableDataCell>
                              {report.triggerWord || "-"}
                            </CTableDataCell>
                            <CTableDataCell>
                              {report.message || "-"}
                            </CTableDataCell>
                            <CTableDataCell>
                              {report.isConfirmed !== undefined ? (
                                report.isConfirmed ? (
                                  <span className="text-success">확인됨</span>
                                ) : (
                                  <span className="text-warning">미확인</span>
                                )
                              ) : (
                                "-"
                              )}
                            </CTableDataCell>
                            <CTableDataCell>
                              <Badge status={report.status} />
                            </CTableDataCell>
                            <CTableDataCell className="action-cell">
                              {!isResolved && !isFalseAlarm ? (
                                <Button
                                  variant="success"
                                  onClick={() =>
                                    handleResolveEmergency(report.reportId)
                                  }
                                  disabled={isResolving}
                                  className="table-action-btn"
                                >
                                  {isResolving ? "처리 중..." : "처리 완료"}
                                </Button>
                              ) : (
                                <span className="text-muted">✓ 처리 완료</span>
                              )}
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
                        {Math.ceil(filteredReports.length / 10)})
                      </Button>
                    </div>
                  )}
                </>
              )}
            </div>
          </CCol>
        </CRow>

        <EmergencyAlertModal
          isOpen={isEmergencyAlertOpen}
          onClose={() => {
            if (currentEmergencyAlert) {
              setAcknowledgedReportIds((prev) =>
                new Set(prev).add(currentEmergencyAlert.reportId)
              );
            }
            setIsEmergencyAlertOpen(false);
            setCurrentEmergencyAlert(null);
          }}
          report={currentEmergencyAlert}
          onAcknowledge={(reportId) => {
            console.log("Emergency acknowledged:", reportId);
            setAcknowledgedReportIds((prev) => new Set(prev).add(reportId));
          }}
        />

        <WebSocketStatus
          isConnected={isConnected}
          isConnecting={isConnecting}
        />
      </CContainer>
    </AdminLayout>
  );
};

export default EmergencyReportPage;
