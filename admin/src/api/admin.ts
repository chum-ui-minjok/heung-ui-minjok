import type { AdminCreateRequest, AdminResponse, AdminPageResponse } from '../types/admin';

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8080/api';

/**
 * 관리자 목록 조회 (페이징)
 */
export const getAdmins = async (
  page: number = 0,
  size: number = 20,
  sort: string = 'createdAt,desc'
): Promise<AdminPageResponse> => {
  const token = localStorage.getItem('accessToken');
  
  const response = await fetch(
    `${API_BASE}/admins?page=${page}&size=${size}&sort=${sort}`,
    {
      method: 'GET',
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json',
      },
    }
  );

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.message || '관리자 목록을 불러오는데 실패했습니다.');
  }

  return response.json();
};

/**
 * 관리자 생성
 */
export const createAdmin = async (adminData: AdminCreateRequest): Promise<AdminResponse> => {
  const token = localStorage.getItem('accessToken');
  
  const response = await fetch(`${API_BASE}/admins`, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${token}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(adminData),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.message || '관리자 생성에 실패했습니다.');
  }

  return response.json();
};

/**
 * 관리자 삭제
 */
export const deleteAdmin = async (adminId: number): Promise<void> => {
  const token = localStorage.getItem('accessToken');
  
  const response = await fetch(`${API_BASE}/admins/${adminId}`, {
    method: 'DELETE',
    headers: {
      'Authorization': `Bearer ${token}`,
      'Content-Type': 'application/json',
    },
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.message || '관리자 삭제에 실패했습니다.');
  }
};

