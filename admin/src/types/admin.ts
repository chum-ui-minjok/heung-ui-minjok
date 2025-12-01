export enum AdminRole {
  SUPER_ADMIN = 'SUPER_ADMIN',
  ADMIN = 'ADMIN',
}

export interface Admin {
  id: number;
  username: string;
  facilityName: string | null;
  contact: string | null;
  email: string | null;
  role: AdminRole;
  createdAt: string;
}

export interface AdminCreateRequest {
  username: string;
  password: string;
  facilityName: string;
  contact?: string;
  email?: string;
  role: AdminRole;
}

export interface AdminResponse {
  id: number;
  username: string;
  facilityName: string | null;
  contact: string | null;
  email: string | null;
  role: AdminRole;
  createdAt: string;
}

export interface AdminPageResponse {
  content: AdminResponse[];
  totalElements: number;
  totalPages: number;
  size: number;
  number: number;
}

