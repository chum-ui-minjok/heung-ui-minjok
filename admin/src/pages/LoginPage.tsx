import { type FormEvent, useState } from "react";
import {
  CAlert,
  CCard,
  CCardBody,
  CCardHeader,
  CCol,
  CContainer,
  CForm,
  CFormInput,
  CRow,
} from "@coreui/react";
import { Button } from "../components";
import { useAuth } from "../hooks/useAuth";

const LoginPage = () => {
  const [username, setUsername] = useState("superadmin");
  const [password, setPassword] = useState("superadmin123!");
  const { login, isLoading, error } = useAuth();

  const handleSubmit = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();

    if (!username.trim() || !password.trim()) {
      return;
    }

    await login({ username, password });
  };

  return (
    <CContainer fluid className="min-vh-100 d-flex align-items-center justify-content-center bg-light">
      <CRow className="w-100 justify-content-center px-3">
        <CCol xs={12} md={8} lg={5} xl={4}>
          <CCard className="shadow-sm border-0">
            <CCardHeader className="text-center bg-white border-0 pt-4 pb-0">
              <h1 className="h3 mb-1">🎵 흥부자</h1>
              <p className="text-body-secondary mb-0">관리자 로그인</p>
            </CCardHeader>
            <CCardBody className="pt-4">
              <CForm onSubmit={handleSubmit} className="d-grid gap-3">
                <CFormInput
                  type="text"
                  placeholder="아이디"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  disabled={isLoading}
                  floatingLabel="아이디"
                  autoComplete="username"
                />
                <CFormInput
                  type="password"
                  placeholder="비밀번호"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  disabled={isLoading}
                  floatingLabel="비밀번호"
                  autoComplete="current-password"
                />
                <Button
                  type="submit"
                  variant="primary"
                  disabled={isLoading || !username.trim() || !password.trim()}
                >
                  {isLoading ? "로그인 중..." : "로그인"}
                </Button>
              </CForm>

              {error && (
                <CAlert color="danger" className="mt-4 mb-0" role="alert">
                  {error}
                </CAlert>
              )}
            </CCardBody>
          </CCard>
        </CCol>
      </CRow>
    </CContainer>
  );
};

export default LoginPage;