import React from 'react';

type ButtonVariant = 'primary' | 'secondary' | 'ghost';
type ButtonSize = 'sm' | 'md' | 'lg';

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  size?: ButtonSize;
  fullWidth?: boolean;
  children: React.ReactNode;
  leftIcon?: React.ReactNode;
  rightIcon?: React.ReactNode;
}

const variantStyles: Record<ButtonVariant, string> = {
  primary:
    'bg-gradient-to-b from-[#00F] to-[#0000D0] text-white border-0 rounded-xl',
  secondary: 'bg-transparent text-white border border-white rounded-full',
  ghost: 'bg-transparent text-zinc-400 border border-transparent rounded-xl',
};

const sizeStyles: Record<ButtonSize, string> = {
  sm: 'px-3 py-1.5 text-sm',
  md: 'px-5 py-3.5 text-base',
  lg: 'px-5 py-3.5 text-md',
};

export const Button: React.FC<ButtonProps> = ({
  variant = 'primary',
  size = 'md',
  fullWidth = false,
  children,
  leftIcon,
  rightIcon,
  className = '',
  disabled,
  ...props
}) => {
  return (
    <button
      className={`
        flex items-center justify-center gap-2 font-medium
        focus:outline-none
        disabled:opacity-50 disabled:cursor-not-allowed
        ${variantStyles[variant]}
        ${sizeStyles[size]}
        ${fullWidth ? 'w-full self-stretch' : ''}
        ${className}
      `}
      disabled={disabled}
      {...props}
    >
      {leftIcon && <span className='flex-shrink-0'>{leftIcon}</span>}
      {children}
      {rightIcon && <span className='flex-shrink-0'>{rightIcon}</span>}
    </button>
  );
};

export default Button;
