import { useState, type FormEvent } from 'react';
import { useNavigate } from 'react-router-dom';
import { Input, Button } from '../components';
import { registerDevice } from '../api/device';
import { useDeviceStore } from '../stores';
import AdminLayout from '../layouts/AdminLayout';
import { adminBaseNavItems, deviceRegisterNavItem, userRegisterNavItem } from '../config/navigation';
import '../styles/dashboard.css';

const DeviceRegisterPage = () => {
  const navigate = useNavigate();
  const [serialNumber, setSerialNumber] = useState('');
  const [location, setLocation] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState('');
  const [successMessage, setSuccessMessage] = useState('');

  const addDevice = useDeviceStore((state) => state.addDevice);

  const navigationItems = [
    ...adminBaseNavItems,
    deviceRegisterNavItem,
    userRegisterNavItem,
  ];

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setError('');
    setSuccessMessage('');

    if (!serialNumber.trim()) {
      setError('기기 일련번호는 필수 입력 항목입니다.');
      return;
    }

    setIsSubmitting(true);

    try {
      const response = await registerDevice({
        serialNumber: serialNumber.trim(),
        location: location.trim() || undefined,
      });

      // 스토어에 추가
      addDevice({
        id: response.id,
        serialNumber: response.serialNumber,
        location: response.location,
        isConnected: false,
        createdAt: response.createdAt,
      });

      setSuccessMessage(`기기가 성공적으로 등록되었습니다! (ID: ${response.id})`);

      // 2초 후 대시보드로 이동
      setTimeout(() => {
        navigate('/dashboard/admin');
      }, 2000);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : '기기 등록에 실패했습니다.';
      setError(errorMessage);
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleCancel = () => {
    if (window.confirm('정말 취소하시겠습니까? 입력한 내용이 모두 사라집니다.')) {
      navigate(-1);
    }
  };

  const handleReset = () => {
    setSerialNumber('');
    setLocation('');
    setError('');
    setSuccessMessage('');
  };

  return (
    <AdminLayout navItems={navigationItems}>
      <div style={{ maxWidth: '600px', margin: '0 auto', padding: '32px' }}>
        <div style={{ textAlign: 'center', marginBottom: '32px' }}>
          <div style={{ fontSize: '48px', marginBottom: '16px' }}>📱</div>
          <h1 style={{ fontSize: '28px', fontWeight: 700, marginBottom: '8px', color: '#213547' }}>
            기기 등록
          </h1>
          <p style={{ color: '#6b7280', fontSize: '14px' }}>
            새로운 기기를 시스템에 등록합니다
          </p>
        </div>

        <form onSubmit={handleSubmit} style={{ textAlign: 'left' }}>
          <Input
            label="기기 일련번호 (필수)"
            placeholder="예: DEVICE-2024-001"
            value={serialNumber}
            onChange={(e) => setSerialNumber(e.target.value)}
            disabled={isSubmitting}
            error={error && !serialNumber.trim() ? error : ''}
          />

          <Input
            label="설치 위치 (선택)"
            placeholder="예: 101호"
            value={location}
            onChange={(e) => setLocation(e.target.value)}
            disabled={isSubmitting}
          />

          {error && serialNumber.trim() && (
            <div className="error-message" style={{ marginTop: '12px' }}>{error}</div>
          )}

          {successMessage && (
            <div className="success-message" style={{ marginTop: '12px' }}>{successMessage}</div>
          )}

          <div style={{ 
            display: 'flex', 
            gap: '12px', 
            marginTop: '24px',
            justifyContent: 'flex-end'
          }}>
            <Button
              type="button"
              variant="secondary"
              onClick={handleCancel}
              disabled={isSubmitting}
            >
              취소
            </Button>
            <Button
              type="button"
              variant="secondary"
              onClick={handleReset}
              disabled={isSubmitting || (!serialNumber && !location)}
            >
              초기화
            </Button>
            <Button
              type="submit"
              variant="success"
              disabled={isSubmitting}
            >
              {isSubmitting ? '등록 중...' : '등록'}
            </Button>
          </div>
        </form>
      </div>
    </AdminLayout>
  );
};

export default DeviceRegisterPage;

