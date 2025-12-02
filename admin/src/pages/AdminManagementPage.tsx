import { useState, useEffect, useCallback, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
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
  CButton,
  CModal,
  CModalHeader,
  CModalTitle,
  CModalBody,
  CModalFooter,
  CFormInput,
  CFormSelect,
  CAlert,
} from '@coreui/react';
import { Button } from '../components';
import DashboardHeader from '../components/DashboardHeader';
import { useNotificationStore } from '../stores';
import '../styles/dashboard.css';
import '../styles/emergency-report-table.css';
import AdminLayout from '../layouts/AdminLayout';
import {
  quickRegisterNavItem,
  developerBaseNavItems,
  adminManagementNavItem,
} from '../config/navigation';
import { getAdmins, createAdmin, deleteAdmin } from '../api/admin';
import { AdminRole, type AdminResponse, type AdminCreateRequest } from '../types/admin';

const AdminManagementPage = () => {
  const navigate = useNavigate();
  const clearUnread = useNotificationStore((state) => state.clearUnread);

  const [allAdmins, setAllAdmins] = useState<AdminResponse[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // 필터링/검색 상태
  const [roleFilters, setRoleFilters] = useState<Set<string>>(
    new Set(['all'])
  );
  const [searchField, setSearchField] = useState<string>('all');
  const [searchQuery, setSearchQuery] = useState<string>('');

  // 페이지네이션 상태
  const [displayCount, setDisplayCount] = useState<number>(10);

  // 모달 상태
  const [isCreateModalOpen, setIsCreateModalOpen] = useState(false);
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
  const [selectedAdmin, setSelectedAdmin] = useState<AdminResponse | null>(null);

  // 생성 폼 상태
  const [createForm, setCreateForm] = useState<AdminCreateRequest>({
    username: '',
    password: '',
    facilityName: '',
    contact: '',
    email: '',
    role: AdminRole.ADMIN,
  });
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [createError, setCreateError] = useState<string | null>(null);

  const navigationItems = [
    ...developerBaseNavItems,
    quickRegisterNavItem,
    adminManagementNavItem,
  ];

  // 모든 관리자 데이터 로드
  const loadAllAdmins = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      // 모든 페이지의 데이터를 가져오기 위해 여러 번 호출
      let allData: AdminResponse[] = [];
      let page = 0;
      let hasMore = true;

      while (hasMore) {
        const response = await getAdmins(page, 100, 'createdAt,desc');
        allData = [...allData, ...response.content];
        hasMore = page < response.totalPages - 1;
        page++;
      }

      setAllAdmins(allData);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : '관리자 목록을 불러오는데 실패했습니다.';
      setError(errorMessage);
      console.error('관리자 목록 로드 실패:', err);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    const token = localStorage.getItem('accessToken');
    if (!token) {
      navigate('/login');
      return;
    }

    loadAllAdmins();
  }, [navigate, loadAllAdmins]);

  const handleNotificationClick = () => {
    clearUnread();
  };

  const handleCreate = async () => {
    setCreateError(null);
    
    if (!createForm.username.trim() || !createForm.password.trim() || !createForm.facilityName.trim()) {
      setCreateError('사용자명, 비밀번호, 시설명은 필수 입력 항목입니다.');
      return;
    }

    setIsSubmitting(true);
    try {
      await createAdmin(createForm);
      setIsCreateModalOpen(false);
      setCreateForm({
        username: '',
        password: '',
        facilityName: '',
        contact: '',
        email: '',
        role: AdminRole.ADMIN,
      });
      await loadAllAdmins();
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : '관리자 생성에 실패했습니다.';
      setCreateError(errorMessage);
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleDelete = async () => {
    if (!selectedAdmin) return;

    try {
      await deleteAdmin(selectedAdmin.id);
      setIsDeleteModalOpen(false);
      setSelectedAdmin(null);
      await loadAllAdmins();
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : '관리자 삭제에 실패했습니다.';
      alert(errorMessage);
    }
  };

  // 역할 필터 체크박스 핸들러
  const handleRoleFilterChange = (role: string) => {
    setRoleFilters((prev) => {
      const newFilters = new Set(prev);
      if (role === 'all') {
        if (newFilters.has('all')) {
          newFilters.clear();
          newFilters.add('all');
        } else {
          newFilters.clear();
          newFilters.add('all');
        }
      } else {
        newFilters.delete('all');
        if (newFilters.has(role)) {
          newFilters.delete(role);
          if (newFilters.size === 0) {
            newFilters.add('all');
          }
        } else {
          newFilters.add(role);
        }
      }
      return newFilters;
    });
  };

  // 필터링 및 검색된 관리자 목록
  const filteredAdmins = useMemo(() => {
    return allAdmins.filter((admin) => {
      // 역할 필터링 (체크박스)
      if (!roleFilters.has('all') && !roleFilters.has(admin.role)) {
        return false;
      }

      // 검색어 필터링
      if (searchQuery.trim()) {
        const query = searchQuery.toLowerCase();
        let matches = false;

        if (searchField === 'all') {
          matches =
            admin.username.toLowerCase().includes(query) ||
            admin.facilityName?.toLowerCase().includes(query) ||
            admin.contact?.toLowerCase().includes(query) ||
            admin.email?.toLowerCase().includes(query) ||
            false;
        } else if (searchField === 'username') {
          matches = admin.username.toLowerCase().includes(query);
        } else if (searchField === 'facilityName') {
          matches = admin.facilityName?.toLowerCase().includes(query) || false;
        } else if (searchField === 'contact') {
          matches = admin.contact?.toLowerCase().includes(query) || false;
        } else if (searchField === 'email') {
          matches = admin.email?.toLowerCase().includes(query) || false;
        }

        if (!matches) {
          return false;
        }
      }

      return true;
    });
  }, [allAdmins, roleFilters, searchField, searchQuery]);

  // 필터가 변경되면 표시 개수 초기화
  useEffect(() => {
    setDisplayCount(10);
  }, [roleFilters, searchField, searchQuery]);

  // 표시할 관리자 목록 (페이지네이션 적용)
  const displayedAdmins = useMemo(() => {
    return filteredAdmins.slice(0, displayCount);
  }, [filteredAdmins, displayCount]);

  // 더보기 버튼 표시 여부
  const hasMore = filteredAdmins.length > displayCount;

  // 더보기 버튼 핸들러
  const handleLoadMore = () => {
    setDisplayCount((prev: number) => prev + 10);
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString('ko-KR', {
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  return (
    <AdminLayout navItems={navigationItems}>
      <CContainer fluid className="py-4">
        <DashboardHeader
          title="관리자 관리"
          onNotificationClick={handleNotificationClick}
        />

        {error && (
          <CAlert color="danger" className="mb-4">
            {error}
          </CAlert>
        )}

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
                      checked={roleFilters.has('all')}
                      onChange={() => handleRoleFilterChange('all')}
                    />
                    <span>전체</span>
                  </label>
                  <label className="filter-checkbox">
                    <input
                      type="checkbox"
                      checked={roleFilters.has(AdminRole.ADMIN)}
                      onChange={() => handleRoleFilterChange(AdminRole.ADMIN)}
                    />
                    <span>ADMIN</span>
                  </label>
                  <label className="filter-checkbox">
                    <input
                      type="checkbox"
                      checked={roleFilters.has(AdminRole.SUPER_ADMIN)}
                      onChange={() => handleRoleFilterChange(AdminRole.SUPER_ADMIN)}
                    />
                    <span>SUPER_ADMIN</span>
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
                      <option value="username">사용자명</option>
                      <option value="facilityName">시설명</option>
                      <option value="contact">연락처</option>
                      <option value="email">이메일</option>
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
        {!isLoading && allAdmins.length > 0 && (
          <CRow className="mb-2">
            <CCol xs={12}>
              <div className="filter-results-count">
                전체 {filteredAdmins.length}건
                {filteredAdmins.length > 10 && (
                  <span className="ms-2 text-muted">
                    ({displayedAdmins.length}건 표시)
                  </span>
                )}
              </div>
            </CCol>
          </CRow>
        )}

        {/* 새 관리자 생성 버튼 */}
        <CRow className="mb-3">
          <CCol>
            <Button
              variant="primary"
              onClick={() => setIsCreateModalOpen(true)}
            >
              + 새 관리자 생성
            </Button>
          </CCol>
        </CRow>

        <CRow className="g-4">
          <CCol xs={12}>
            <div className="emergency-report-container">
              {isLoading ? (
                <div className="text-center py-5">
                  <div className="spinner-border" role="status">
                    <span className="visually-hidden">로딩 중...</span>
                  </div>
                  <p className="mt-2">관리자 목록 불러오는 중...</p>
                </div>
              ) : filteredAdmins.length === 0 ? (
                <div className="text-center py-5">
                  <div className="mb-3" style={{ fontSize: '3rem' }}>
                    📋
                  </div>
                  <p>필터 조건에 맞는 관리자가 없습니다.</p>
                </div>
              ) : (
                <>
                  <CTable hover responsive className="emergency-report-table">
                    <CTableHead>
                      <CTableRow>
                        <CTableHeaderCell>번호</CTableHeaderCell>
                        <CTableHeaderCell>사용자명</CTableHeaderCell>
                        <CTableHeaderCell>시설명</CTableHeaderCell>
                        <CTableHeaderCell>연락처</CTableHeaderCell>
                        <CTableHeaderCell>이메일</CTableHeaderCell>
                        <CTableHeaderCell>역할</CTableHeaderCell>
                        <CTableHeaderCell>생성일</CTableHeaderCell>
                        <CTableHeaderCell>작업</CTableHeaderCell>
                      </CTableRow>
                    </CTableHead>
                    <CTableBody>
                      {displayedAdmins.map((admin, index) => (
                        <CTableRow key={admin.id}>
                          <CTableDataCell>
                            {filteredAdmins.indexOf(admin) + 1}
                          </CTableDataCell>
                          <CTableDataCell>{admin.username}</CTableDataCell>
                          <CTableDataCell>{admin.facilityName || '-'}</CTableDataCell>
                          <CTableDataCell>{admin.contact || '-'}</CTableDataCell>
                          <CTableDataCell>{admin.email || '-'}</CTableDataCell>
                          <CTableDataCell>
                            <span
                              className={`badge ${
                                admin.role === AdminRole.SUPER_ADMIN
                                  ? 'badge-danger'
                                  : 'badge-primary'
                              }`}
                            >
                              {admin.role === AdminRole.SUPER_ADMIN ? 'SUPER_ADMIN' : 'ADMIN'}
                            </span>
                          </CTableDataCell>
                          <CTableDataCell>{formatDate(admin.createdAt)}</CTableDataCell>
                          <CTableDataCell className="action-cell">
                            <Button
                              variant="danger"
                              onClick={() => {
                                setSelectedAdmin(admin);
                                setIsDeleteModalOpen(true);
                              }}
                              disabled={admin.role === AdminRole.SUPER_ADMIN}
                              className="table-action-btn"
                            >
                              삭제
                            </Button>
                          </CTableDataCell>
                        </CTableRow>
                      ))}
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
                        {Math.ceil(filteredAdmins.length / 10)})
                      </Button>
                    </div>
                  )}
                </>
              )}
            </div>
          </CCol>
        </CRow>

        {/* 생성 모달 */}
        <CModal visible={isCreateModalOpen} onClose={() => setIsCreateModalOpen(false)}>
          <CModalHeader>
            <CModalTitle>새 관리자 생성</CModalTitle>
          </CModalHeader>
          <CModalBody>
            {createError && (
              <CAlert color="danger" className="mb-3">
                {createError}
              </CAlert>
            )}
            <div className="mb-3">
              <label className="form-label">사용자명 *</label>
              <CFormInput
                value={createForm.username}
                onChange={(e) =>
                  setCreateForm({ ...createForm, username: e.target.value })
                }
                placeholder="사용자명을 입력하세요"
              />
            </div>
            <div className="mb-3">
              <label className="form-label">비밀번호 *</label>
              <CFormInput
                type="password"
                value={createForm.password}
                onChange={(e) =>
                  setCreateForm({ ...createForm, password: e.target.value })
                }
                placeholder="비밀번호를 입력하세요"
              />
            </div>
            <div className="mb-3">
              <label className="form-label">시설명 *</label>
              <CFormInput
                value={createForm.facilityName}
                onChange={(e) =>
                  setCreateForm({ ...createForm, facilityName: e.target.value })
                }
                placeholder="시설명을 입력하세요"
              />
            </div>
            <div className="mb-3">
              <label className="form-label">연락처</label>
              <CFormInput
                value={createForm.contact || ''}
                onChange={(e) =>
                  setCreateForm({ ...createForm, contact: e.target.value })
                }
                placeholder="연락처를 입력하세요"
              />
            </div>
            <div className="mb-3">
              <label className="form-label">이메일</label>
              <CFormInput
                type="email"
                value={createForm.email || ''}
                onChange={(e) =>
                  setCreateForm({ ...createForm, email: e.target.value })
                }
                placeholder="이메일을 입력하세요"
              />
            </div>
            <div className="mb-3">
              <label className="form-label">역할 *</label>
              <CFormSelect
                value={createForm.role}
                onChange={(e) =>
                  setCreateForm({ ...createForm, role: e.target.value as AdminRole })
                }
              >
                <option value={AdminRole.ADMIN}>ADMIN</option>
                <option value={AdminRole.SUPER_ADMIN}>SUPER_ADMIN</option>
              </CFormSelect>
            </div>
          </CModalBody>
          <CModalFooter>
            <CButton color="secondary" onClick={() => setIsCreateModalOpen(false)}>
              취소
            </CButton>
            <CButton
              color="primary"
              onClick={handleCreate}
              disabled={isSubmitting}
            >
              {isSubmitting ? '생성 중...' : '생성'}
            </CButton>
          </CModalFooter>
        </CModal>

        {/* 삭제 확인 모달 */}
        <CModal visible={isDeleteModalOpen} onClose={() => setIsDeleteModalOpen(false)}>
          <CModalHeader>
            <CModalTitle>관리자 삭제 확인</CModalTitle>
          </CModalHeader>
          <CModalBody>
            정말로 관리자 <strong>{selectedAdmin?.username}</strong>을(를) 삭제하시겠습니까?
            <br />
            <small className="text-body-secondary">
              관리 중인 기기나 사용자가 있으면 삭제할 수 없습니다.
            </small>
          </CModalBody>
          <CModalFooter>
            <CButton color="secondary" onClick={() => setIsDeleteModalOpen(false)}>
              취소
            </CButton>
            <CButton color="danger" onClick={handleDelete}>
              삭제
            </CButton>
          </CModalFooter>
        </CModal>
      </CContainer>
    </AdminLayout>
  );
};

export default AdminManagementPage;
