import { Loader2 } from 'lucide-react';
import React, { useEffect, useRef } from 'react';

interface DialogProps {
  isOpen: boolean;
  onClose: () => void;
  /** Called once when the dialog opens. Ref-guarded against StrictMode double-fire. */
  onOpen?: () => void;
  header?: React.ReactNode;
  footer?: React.ReactNode;
  children: React.ReactNode;
  /** Whether clicking the backdrop closes the dialog */
  closeOnBackdropClick?: boolean;
  /** Max width class for the dialog */
  maxWidth?: 'sm' | 'md' | 'lg' | 'xl';
  /** Whether to show a loading indicator in place of the dialog */
  isLoading?: boolean;
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
  onOpen,
  header,
  footer,
  children,
  closeOnBackdropClick = true,
  maxWidth = 'md',
  isLoading = false,
}) => {
  // Call onOpen exactly once per open transition.
  // The ref persists across StrictMode's simulated unmount/remount,
  // preventing the double-fire while still calling on each real mount.
  const didOpenRef = useRef(false);
  useEffect(() => {
    if (isOpen && !didOpenRef.current) {
      didOpenRef.current = true;
      onOpen?.();
    }
    if (!isOpen) {
      didOpenRef.current = false;
    }
  }, [isOpen, onOpen]);

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

      {isLoading && (
        <Loader2
          size={42}
          className='text-zinc-700 flex-shrink-0 animate-spin'
        />
      )}

      {/* Dialog Content */}
      {!isLoading && (
        <div
          className={`relative bg-zinc-900 text-white border border-zinc-700 rounded-xl w-full ${maxWidthStyles[maxWidth]} mx-4 shadow-xl`}
        >
          {/* Header */}
          {header && <div>{header}</div>}

          {/* Body */}
          <div className='px-4 pb-6'>{children}</div>

          {/* Footer */}
          {footer && <div>{footer}</div>}
        </div>
      )}
    </div>
  );
};

export default Dialog;
