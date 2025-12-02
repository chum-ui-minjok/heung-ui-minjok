import type { ButtonHTMLAttributes } from 'react';
import { CButton } from '@coreui/react';

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'success' | 'danger' | 'secondary';
  fullWidth?: boolean;
}

const Button = ({ 
  variant = 'primary', 
  fullWidth = false, 
  children, 
  className = '',
  ...props 
}: ButtonProps) => {
  return (
    <CButton
      color={variant}
      className={`${fullWidth ? 'w-100' : ''} ${className}`.trim()}
      {...props}
    >
      {children}
    </CButton>
  );
};

export default Button;