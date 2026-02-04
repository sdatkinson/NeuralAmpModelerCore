import React from 'react';
import { AlertCircle, AlertTriangle } from 'lucide-react';

type AlertVariant = 'error' | 'warning' | 'info';

interface AlertProps {
  variant: AlertVariant;
  children: React.ReactNode;
  description?: React.ReactNode;
}

const variantStyles: Record<AlertVariant, {
  container: string;
  icon: string;
  title: string;
  description: string;
}> = {
  error: {
    container: 'bg-red-950/50 border-red-900/50',
    icon: 'text-red-400',
    title: 'text-red-300',
    description: 'text-red-400',
  },
  warning: {
    container: 'bg-yellow-950/50 border-yellow-900/50',
    icon: 'text-yellow-400',
    title: 'text-yellow-300',
    description: 'text-yellow-400',
  },
  info: {
    container: 'bg-zinc-800/50 border-zinc-700',
    icon: 'text-zinc-400',
    title: 'text-zinc-300',
    description: 'text-zinc-400',
  },
};

const AlertIcon: React.FC<{ variant: AlertVariant; className?: string }> = ({ variant, className }) => {
  if (variant === 'warning') {
    return <AlertTriangle size={18} className={className} />;
  }
  return <AlertCircle size={18} className={className} />;
};

export const Alert: React.FC<AlertProps> = ({ variant, children, description }) => {
  const styles = variantStyles[variant];

  return (
    <div className={`flex items-start gap-3 p-3 border rounded-md ${styles.container}`}>
      <AlertIcon variant={variant} className={`${styles.icon} flex-shrink-0 mt-0.5`} />
      <div className={`flex flex-col gap-1 text-sm ${styles.title}`}>
        {children}
        {description && (
          <p className={`text-xs ${styles.description}`}>{description}</p>
        )}
      </div>
    </div>
  );
};
