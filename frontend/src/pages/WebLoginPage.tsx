import { useState, type FormEvent } from 'react';
import { useDeviceAuth } from '@/hooks/useDeviceAuth';
import LoadingDots from '@/components/icons/LoadingDots';
import './WebLoginPage.css';

const WebLoginPage = () => {
  const [deviceNumber, setDeviceNumber] = useState('DEVICE001');
  const [userId] = useState('1');
  const { login, isLoading, error } = useDeviceAuth();

  const handleSubmit = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();

    if (!deviceNumber.trim() || !userId.trim()) return;

    const userIdNumber = parseInt(userId, 10);
    if (isNaN(userIdNumber) || userIdNumber <= 0) return;

    await login({
      serialNumber: deviceNumber.trim(),
    });
  };

  return (
    <>
    <div className="user-login-container">
      {isLoading ? (
        <LoadingDots className="login-loading-dots" />
      ) :
      <div className="user-login-section">
        <div className="login-logo">
          <div className="logo-circle">
            <img src="logo.svg" alt="흥의 민족 로고" />
          </div>
          <h1 className="service-name">흥의 민족</h1>
        </div>

        <form onSubmit={handleSubmit} className="login-form">
          <div className="input-wrapper">
            <label className="input-label">기기 일련번호</label>
            <input
              type="text"
              className="login-input"
              placeholder="예: DEVICE001"
              value={deviceNumber}
              onChange={(e) => setDeviceNumber(e.target.value)}
              disabled={isLoading}
            />
          </div>

          <button
            type="submit"
            className="login-btn"
            disabled={isLoading || !deviceNumber.trim() || !userId.trim()}
          >로그인</button>
        </form>

        {error && <div className="error-box">{error}</div>}
      </div>
      }
    </div>
    </>
  );
};

export default WebLoginPage;
