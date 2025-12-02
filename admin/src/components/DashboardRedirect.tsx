import { Navigate } from 'react-router-dom';

const DashboardRedirect = () => {
  const role = localStorage.getItem('adminRole');
  const isSuperAdmin = role === 'SUPER_ADMIN' || role === 'superadmin';
  const redirectPath = isSuperAdmin ? '/dashboard/developer/visualization' : '/dashboard/admin/emergencies';
  
  return <Navigate to={redirectPath} replace />;
};

export default DashboardRedirect;

