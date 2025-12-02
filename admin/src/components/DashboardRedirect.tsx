import { Navigate } from 'react-router-dom';

const DashboardRedirect = () => {
  const role = localStorage.getItem('adminRole');
  const isSuperAdmin = role === 'SUPER_ADMIN' || role === 'superadmin';
  const dashboardPath = isSuperAdmin ? '/dashboard/developer' : '/dashboard/admin';
  
  return <Navigate to={dashboardPath} replace />;
};

export default DashboardRedirect;

