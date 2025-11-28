import { CBadge, CButton, CCol, CRow } from '@coreui/react';
import { useNotificationStore } from '../stores';

interface DashboardHeaderProps {
  title: string;
  onNotificationClick: () => void;
}

const DashboardHeader = ({
  title,
  onNotificationClick,
}: DashboardHeaderProps) => {
  const unreadCount = useNotificationStore((state) => state.unreadCount);
  const showBadge = useNotificationStore((state) => state.showBadge);

  return (
    <div className="mb-4">
      <CRow className="align-items-center g-3">
        <CCol>
          <h1 className="h3 mb-0">{title}</h1>
        </CCol>
        <CCol xs="auto">
          <CButton
            color="light"
            className="position-relative"
            onClick={onNotificationClick}
          >
            🔔
            {showBadge && unreadCount > 0 && (
              <CBadge color="danger" className="position-absolute top-0 start-100 translate-middle">
                {unreadCount}
              </CBadge>
            )}
          </CButton>
        </CCol>
      </CRow>
    </div>
  );
};

export default DashboardHeader;
