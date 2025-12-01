import { useState, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  CCard,
  CCardBody,
  CCardHeader,
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
  CPagination,
  CPaginationItem,
} from '@coreui/react';
import DashboardHeader from '../components/DashboardHeader';
import { useNotificationStore } from '../stores';
import '../styles/dashboard.css';
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

  const [admins, setAdmins] = useState<AdminResponse[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [currentPage, setCurrentPage] = useState(0);
  const [totalPages, setTotalPages] = useState(0);
  const [totalElements, setTotalElements] = useState(0);

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

  const loadAdmins = useCallback(async (page: number = 0) => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await getAdmins(page, 20, 'createdAt,desc');
      setAdmins(response.content);
      setTotalPages(response.totalPages);
      setTotalElements(response.totalElements);
      setCurrentPage(response.number);
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

    loadAdmins(0);
  }, [navigate, loadAdmins]);

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
      await loadAdmins(currentPage);
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
      await loadAdmins(currentPage);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : '관리자 삭제에 실패했습니다.';
      alert(errorMessage);
    }
  };

  const handlePageChange = (page: number) => {
    loadAdmins(page);
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

        <CRow className="mb-3">
          <CCol>
            <CButton
              color="primary"
              onClick={() => setIsCreateModalOpen(true)}
            >
              + 새 관리자 생성
            </CButton>
          </CCol>
        </CRow>

        <CRow>
          <CCol xs={12}>
            <CCard>
              <CCardHeader className="fw-semibold">
                관리자 목록 ({totalElements}명)
              </CCardHeader>
              <CCardBody>
                {isLoading ? (
                  <div className="text-center py-4">로딩 중...</div>
                ) : admins.length === 0 ? (
                  <div className="text-center py-4 text-body-secondary">
                    등록된 관리자가 없습니다.
                  </div>
                ) : (
                  <>
                    <CTable hover responsive>
                      <CTableHead>
                        <CTableRow>
                          <CTableHeaderCell>ID</CTableHeaderCell>
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
                        {admins.map((admin) => (
                          <CTableRow key={admin.id}>
                            <CTableDataCell>{admin.id}</CTableDataCell>
                            <CTableDataCell>{admin.username}</CTableDataCell>
                            <CTableDataCell>{admin.facilityName || '-'}</CTableDataCell>
                            <CTableDataCell>{admin.contact || '-'}</CTableDataCell>
                            <CTableDataCell>{admin.email || '-'}</CTableDataCell>
                            <CTableDataCell>
                              <span
                                className={`badge ${
                                  admin.role === AdminRole.SUPER_ADMIN
                                    ? 'bg-danger'
                                    : 'bg-primary'
                                }`}
                              >
                                {admin.role === AdminRole.SUPER_ADMIN ? 'SUPER_ADMIN' : 'ADMIN'}
                              </span>
                            </CTableDataCell>
                            <CTableDataCell>{formatDate(admin.createdAt)}</CTableDataCell>
                            <CTableDataCell>
                              <CButton
                                color="danger"
                                size="sm"
                                onClick={() => {
                                  setSelectedAdmin(admin);
                                  setIsDeleteModalOpen(true);
                                }}
                                disabled={admin.role === AdminRole.SUPER_ADMIN}
                              >
                                삭제
                              </CButton>
                            </CTableDataCell>
                          </CTableRow>
                        ))}
                      </CTableBody>
                    </CTable>

                    {totalPages > 1 && (
                      <CPagination className="mt-3 justify-content-center">
                        <CPaginationItem
                          disabled={currentPage === 0}
                          onClick={() => currentPage > 0 && handlePageChange(currentPage - 1)}
                        >
                          이전
                        </CPaginationItem>
                        {Array.from({ length: totalPages }, (_, i) => (
                          <CPaginationItem
                            key={i}
                            active={i === currentPage}
                            onClick={() => handlePageChange(i)}
                          >
                            {i + 1}
                          </CPaginationItem>
                        ))}
                        <CPaginationItem
                          disabled={currentPage === totalPages - 1}
                          onClick={() =>
                            currentPage < totalPages - 1 && handlePageChange(currentPage + 1)
                          }
                        >
                          다음
                        </CPaginationItem>
                      </CPagination>
                    )}
                  </>
                )}
              </CCardBody>
            </CCard>
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

