import React, { useEffect } from 'react';

interface DialogProps {
  isOpen: boolean;
  onClose: () => void;
  header?: React.ReactNode;
  footer?: React.ReactNode;
  children: React.ReactNode;
  /** Whether clicking the backdrop closes the dialog */
  closeOnBackdropClick?: boolean;
  /** Max width class for the dialog */
  maxWidth?: 'sm' | 'md' | 'lg' | 'xl';
}

const maxWidthStyles: Record<string, string> = {
  sm: 'max-w-sm',
  md: 'max-w-md',
  lg: 'max-w-lg',
  xl: 'max-w-xl',
};

export const Dialog: React.FC<DialogProps> = ({
  isOpen,
  onClose,
  header,
  footer,
  children,
  closeOnBackdropClick = true,
  maxWidth = 'md',
}) => {
  // Handle escape key
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isOpen) {
        onClose();
      }
    };

    document.addEventListener('keydown', handleEscape);
    return () => document.removeEventListener('keydown', handleEscape);
  }, [isOpen, onClose]);

  // Prevent body scroll when dialog is open
  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = '';
    }
    return () => {
      document.body.style.overflow = '';
    };
  }, [isOpen]);

  if (!isOpen) return null;

  return (
    <div
      className='fixed inset-0 z-50 flex items-center justify-center'
      role='dialog'
      aria-modal='true'
    >
      {/* Backdrop */}
      <div
        className='absolute inset-0 bg-black/70'
        onClick={closeOnBackdropClick ? onClose : undefined}
        aria-hidden='true'
      />

      {/* Dialog Content */}
      <div
        className={`relative bg-zinc-900 border border-zinc-700 rounded-xl w-full ${maxWidthStyles[maxWidth]} mx-4 shadow-xl`}
      >
        {/* Header */}
        {header && (
          <div className='border-b border-zinc-700'>
            {header}
          </div>
        )}

        {/* Body */}
        <div className='p-6'>
          {children}
        </div>

        {/* Footer */}
        {footer && (
          <div className='border-t border-zinc-700'>
            {footer}
          </div>
        )}
      </div>
    </div>
  );
};

export default Dialog;
