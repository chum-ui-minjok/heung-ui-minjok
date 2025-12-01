import { useState, type FormEvent, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Input, Select, Textarea, Button } from '../components';
import { registerUser } from '../api/user';
import { getDevices } from '../api/device';
import { useUserStore, useDeviceStore } from '../stores';
import { type Gender } from '../types/user';
import AdminLayout from '../layouts/AdminLayout';
import { adminBaseNavItems, deviceRegisterNavItem, userRegisterNavItem } from '../config/navigation';
import '../styles/dashboard.css';

const UserRegisterPage = () => {
  const navigate = useNavigate();
  const [name, setName] = useState('');
  const [birthDate, setBirthDate] = useState('');
  const [gender, setGender] = useState<Gender | ''>('');
  const [emergencyContact, setEmergencyContact] = useState('');
  const [deviceId, setDeviceId] = useState('');
  const [medicalNotes, setMedicalNotes] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState('');
  const [successMessage, setSuccessMessage] = useState('');
  const [isLoadingDevices, setIsLoadingDevices] = useState(true);
  const [deviceLoadError, setDeviceLoadError] = useState<string | null>(null);

  const addUser = useUserStore((state) => state.addUser);
  const devices = useDeviceStore((state) => state.devices);
  const setDevices = useDeviceStore((state) => state.setDevices);

  const navigationItems = [
    ...adminBaseNavItems,
    deviceRegisterNavItem,
    userRegisterNavItem,
  ];

  // availableDevices를 컴포넌트 내부에서 계산
  const availableDevices = devices.filter((device) => !device.connectedUserId);

  // 페이지 로드 시 기기 목록 로드
  useEffect(() => {
    loadAvailableDevices();
  }, []);

  const loadAvailableDevices = async () => {
    setIsLoadingDevices(true);
    setDeviceLoadError(null);
    try {
      const deviceList = await getDevices(true); // availableOnly=true
      setDevices(deviceList);
    } catch (err) {
      console.error('기기 목록 로드 실패:', err);
      setDeviceLoadError(err instanceof Error ? err.message : '기기 목록을 불러오는데 실패했습니다.');
      // 에러가 발생해도 폼은 보여줌
    } finally {
      setIsLoadingDevices(false);
    }
  };

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setError('');
    setSuccessMessage('');

    if (!name.trim()) {
      setError('이름은 필수 입력 항목입니다.');
      return;
    }

    if (!deviceId) {
      setError('연결할 기기를 선택해주세요.');
      return;
    }

    setIsSubmitting(true);

    try {
      const response = await registerUser({
        name: name.trim(),
        birthDate: birthDate || undefined,
        gender: gender || undefined,
        emergencyContact: emergencyContact.trim() || undefined,
        medicalNotes: medicalNotes.trim() || undefined,
        deviceId: parseInt(deviceId),
      });

      // 스토어에 추가
      addUser({
        id: response.id,
        name: response.name,
        birthDate: birthDate || undefined,
        gender: gender || undefined,
        emergencyContact: emergencyContact.trim() || undefined,
        medicalNotes: medicalNotes.trim() || undefined,
        deviceId: response.deviceId,
        status: 'ACTIVE',
        createdAt: response.createdAt,
      });

      setSuccessMessage(`어르신이 성공적으로 등록되었습니다! (ID: ${response.id})`);

      // 2초 후 대시보드로 이동
      setTimeout(() => {
        navigate('/dashboard/admin');
      }, 2000);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : '어르신 등록에 실패했습니다.';
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
    setName('');
    setBirthDate('');
    setGender('');
    setEmergencyContact('');
    setDeviceId('');
    setMedicalNotes('');
    setError('');
    setSuccessMessage('');
  };

  const genderOptions = [
    { value: '', label: '성별 선택 (선택)' },
    { value: 'MALE', label: '남성' },
    { value: 'FEMALE', label: '여성' },
    { value: 'OTHER', label: '기타' },
  ];

  const deviceOptions = [
    { value: '', label: '연결할 기기 선택 (필수)' },
    ...availableDevices.map((device) => ({
      value: device.id.toString(),
      label: `${device.serialNumber}${device.location ? ` (${device.location})` : ''}`,
    })),
  ];

  // 로딩 중일 때 (최대 5초만 표시, 그 이후에는 폼을 보여줌)
  const [showLoadingTimeout, setShowLoadingTimeout] = useState(false);
  
  useEffect(() => {
    if (isLoadingDevices) {
      const timeout = setTimeout(() => {
        setShowLoadingTimeout(true);
      }, 5000);
      return () => clearTimeout(timeout);
    } else {
      setShowLoadingTimeout(false);
    }
  }, [isLoadingDevices]);

  if (isLoadingDevices && !showLoadingTimeout) {
    return (
      <AdminLayout navItems={navigationItems}>
        <div style={{ maxWidth: '600px', margin: '0 auto', padding: '32px', textAlign: 'center' }}>
          <p>기기 목록을 불러오는 중...</p>
        </div>
      </AdminLayout>
    );
  }

  // 사용 가능한 기기가 없을 때 (기기가 등록되어 있지만 모두 사용 중)
  if (!isLoadingDevices && availableDevices.length === 0 && devices.length > 0) {
    return (
      <AdminLayout navItems={navigationItems}>
        <div style={{ maxWidth: '600px', margin: '0 auto', padding: '32px' }}>
          <div style={{ textAlign: 'center', marginBottom: '32px' }}>
            <div style={{ fontSize: '48px', marginBottom: '16px' }}>⚠️</div>
            <h1 style={{ fontSize: '28px', fontWeight: 700, marginBottom: '8px', color: '#213547' }}>
              사용 가능한 기기가 없습니다
            </h1>
            <p style={{ color: '#6b7280', fontSize: '14px', marginBottom: '24px' }}>
              모든 기기가 이미 사용 중입니다. 먼저 기기를 등록해주세요.
            </p>
            <Button variant="primary" onClick={() => navigate('/dashboard/admin/device-register')}>
              기기 등록하기
            </Button>
          </div>
        </div>
      </AdminLayout>
    );
  }

  // 기기가 아예 없을 때도 폼을 보여주되 경고 메시지 표시
  const showNoDeviceWarning = !isLoadingDevices && devices.length === 0 && !deviceLoadError;

  return (
    <AdminLayout navItems={navigationItems}>
      <div style={{ maxWidth: '600px', margin: '0 auto', padding: '32px' }}>
        <div style={{ textAlign: 'center', marginBottom: '32px' }}>
          <div style={{ fontSize: '48px', marginBottom: '16px' }}>👴</div>
          <h1 style={{ fontSize: '28px', fontWeight: 700, marginBottom: '8px', color: '#213547' }}>
            어르신 등록
          </h1>
          <p style={{ color: '#6b7280', fontSize: '14px' }}>
            새로운 어르신을 시스템에 등록합니다
          </p>
        </div>

        <form onSubmit={handleSubmit} style={{ textAlign: 'left' }}>
          <Input
            label="이름 (필수)"
            placeholder="어르신 성함"
            value={name}
            onChange={(e) => setName(e.target.value)}
            disabled={isSubmitting}
            error={error && !name.trim() ? '이름은 필수입니다' : ''}
          />

          <Input
            label="생년월일 (선택)"
            type="date"
            value={birthDate}
            onChange={(e) => setBirthDate(e.target.value)}
            disabled={isSubmitting}
          />

          <Select
            label="성별 (선택)"
            options={genderOptions}
            value={gender}
            onChange={(e) => setGender(e.target.value as Gender | '')}
            disabled={isSubmitting}
          />

          <Input
            label="비상 연락처 (선택)"
            placeholder="010-0000-0000"
            value={emergencyContact}
            onChange={(e) => setEmergencyContact(e.target.value)}
            disabled={isSubmitting}
          />

          {deviceLoadError && (
            <div style={{ 
              padding: '12px', 
              backgroundColor: '#fee2e2', 
              border: '1px solid #ef4444', 
              borderRadius: '8px',
              marginBottom: '16px',
              color: '#991b1b'
            }}>
              ⚠️ 기기 목록 로드 실패: {deviceLoadError}
            </div>
          )}

          {showNoDeviceWarning && (
            <div style={{ 
              padding: '12px', 
              backgroundColor: '#fef3c7', 
              border: '1px solid #fbbf24', 
              borderRadius: '8px',
              marginBottom: '16px',
              color: '#92400e'
            }}>
              ⚠️ 등록된 기기가 없습니다. 먼저 기기를 등록해주세요.
            </div>
          )}

          <Select
            label="연결할 기기 (필수)"
            options={deviceOptions}
            value={deviceId}
            onChange={(e) => setDeviceId(e.target.value)}
            disabled={isSubmitting || showNoDeviceWarning}
            error={error && !deviceId ? '기기 선택은 필수입니다' : ''}
          />

          <Textarea
            label="의료 특이사항 (선택)"
            placeholder="알레르기, 복용 약물 등"
            value={medicalNotes}
            onChange={(e) => setMedicalNotes(e.target.value)}
            disabled={isSubmitting}
            rows={3}
          />

          {error && name.trim() && deviceId && (
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
              disabled={isSubmitting || (!name && !birthDate && !gender && !emergencyContact && !deviceId && !medicalNotes)}
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

export default UserRegisterPage;

