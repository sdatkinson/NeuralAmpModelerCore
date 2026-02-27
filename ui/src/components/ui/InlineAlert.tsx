import React from 'react';
import { AlertTriangle } from 'lucide-react';

interface InlineAlertProps {
  children: React.ReactNode;
  className?: string;
}

export const InlineAlert: React.FC<InlineAlertProps> = ({
  children,
  className = '',
}) => {
  return (
    <div
      className={`flex items-start gap-2 p-2 bg-yellow-950/30 border border-yellow-900/30 rounded text-xs text-yellow-400 ${className}`}
    >
      <AlertTriangle size={14} className='flex-shrink-0 mt-0.5' />
      <span>{children}</span>
    </div>
  );
};
