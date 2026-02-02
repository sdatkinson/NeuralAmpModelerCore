import React, { forwardRef } from 'react';

export interface ClipIndicatorProps {
  /** Click handler to reset the clip latch */
  onClick?: () => void;
  /** Size of the indicator in pixels */
  size?: number;
  /** Additional CSS classes */
  className?: string;
  /** Label for accessibility */
  label?: string;
}

/**
 * A clip indicator that latches red when clipping is detected.
 *
 * The indicator turns red when the 'clipped' class is added (by useMeterAnimation).
 * Clicking it resets the latch.
 *
 * Usage:
 * ```tsx
 * const clipRef = useRef<HTMLButtonElement>(null);
 * const { resetClipLatch } = useMeterAnimation(
 *   { analyser, meterRef, clipRef },
 *   null,
 *   true
 * );
 * <ClipIndicator ref={clipRef} onClick={() => resetClipLatch('input')} />
 * ```
 */
export const ClipIndicator = forwardRef<HTMLButtonElement, ClipIndicatorProps>(
  (
    {
      onClick,
      size = 12,
      className = '',
      label = 'Clip indicator - click to reset',
    },
    ref
  ) => {
    return (
      <button
        ref={ref}
        type="button"
        onClick={onClick}
        className={`
          rounded-sm transition-colors duration-75
          bg-zinc-700 hover:bg-zinc-600
          focus:outline-none focus:ring-1 focus:ring-zinc-500
          [&.clipped]:bg-red-500 [&.clipped]:hover:bg-red-400
          ${className}
        `}
        style={{
          width: size,
          height: size,
          minWidth: size,
          minHeight: size,
        }}
        aria-label={label}
        aria-pressed={false}
        title="Click to reset clip indicator"
      />
    );
  }
);

ClipIndicator.displayName = 'ClipIndicator';
