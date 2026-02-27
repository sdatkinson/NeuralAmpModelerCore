import React from 'react';
import { AlertCircle, AlertTriangle } from 'lucide-react';

type AlertVariant = 'error' | 'warning' | 'info';

interface AlertProps {
  variant: AlertVariant;
  children: React.ReactNode;
  description?: React.ReactNode;
}

const variantStyles: Record<
  AlertVariant,
  {
    container: string;
    icon: string;
    title: string;
    description: string;
  }
> = {
  error: {
    container: '',
    icon: 'text-[#F00]',
    title: 'text-[#F00]',
    description: 'text-[#F00]',
  },
  warning: {
    container: '',
    icon: 'text-[#F00]',
    title: 'text-[#F00]',
    description: 'text-[#F00]',
  },
  info: {
    container: '',
    icon: 'text-zinc-400',
    title: 'text-zinc-400',
    description: 'text-zinc-400',
  },
};

const AlertIcon: React.FC<{ variant: AlertVariant; className?: string }> = ({
  variant,
  className,
}) => {
  if (variant === 'warning') {
    return <AlertCircle size={20} className={className} />;
  }
  return <AlertTriangle size={20} className={className} />;
};

export const Alert: React.FC<AlertProps> = ({ variant, children }) => {
  const styles = variantStyles[variant];

  return (
    <div className={`flex items-start gap-3 ${styles.container}`}>
      <AlertIcon variant={variant} className={`${styles.icon} flex-shrink-0`} />
      <div className={`flex flex-col gap-1 text-sm ${styles.title}`}>
        {children}
      </div>
    </div>
  );
};
