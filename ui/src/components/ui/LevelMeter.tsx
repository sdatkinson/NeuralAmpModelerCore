import React, { forwardRef } from 'react';

export interface LevelMeterProps {
  /** Orientation of the meter */
  orientation?: 'vertical' | 'horizontal';
  /** Height in pixels (for vertical) or width (for horizontal) */
  size?: number;
  /** Thickness in pixels */
  thickness?: number;
  /** Additional CSS classes */
  className?: string;
  /** Label for accessibility */
  label?: string;
}

/**
 * A real-time audio level meter component.
 *
 * Animation is controlled via the --level CSS custom property (0 to 1),
 * which should be set by useMeterAnimation hook for 60fps updates
 * without React re-renders.
 *
 * Usage:
 * ```tsx
 * const meterRef = useRef<HTMLDivElement>(null);
 * useMeterAnimation({ analyser, meterRef }, null, true);
 * <LevelMeter ref={meterRef} />
 * ```
 */
export const LevelMeter = forwardRef<HTMLDivElement, LevelMeterProps>(
  (
    {
      orientation = 'vertical',
      size = 80,
      thickness = 8,
      className = '',
      label = 'Audio level meter',
    },
    ref
  ) => {
    const isVertical = orientation === 'vertical';

    return (
      <div
        className={`relative overflow-hidden rounded-sm ${className}`}
        style={{
          width: isVertical ? thickness : size,
          height: isVertical ? size : thickness,
          backgroundColor: '#27272a', // zinc-800
        }}
        role="meter"
        aria-label={label}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={0}
      >
        {/* Meter fill - animated via --level CSS variable */}
        <div
          ref={ref}
          className="absolute transition-transform duration-[16ms] ease-linear"
          style={{
            // Position at bottom (vertical) or left (horizontal)
            bottom: isVertical ? 0 : undefined,
            left: isVertical ? 0 : 0,
            right: isVertical ? 0 : undefined,
            top: isVertical ? undefined : 0,
            // Full dimension on the non-animated axis
            width: isVertical ? '100%' : '100%',
            height: isVertical ? '100%' : '100%',
            // Gradient from green (bottom/left) through yellow to red (top/right)
            background: isVertical
              ? 'linear-gradient(to top, #22c55e 0%, #22c55e 60%, #eab308 75%, #ef4444 90%, #ef4444 100%)'
              : 'linear-gradient(to right, #22c55e 0%, #22c55e 60%, #eab308 75%, #ef4444 90%, #ef4444 100%)',
            // Scale transform controlled by --level (0 to 1)
            transform: isVertical
              ? 'scaleY(var(--level, 0))'
              : 'scaleX(var(--level, 0))',
            transformOrigin: isVertical ? 'bottom' : 'left',
          }}
        />

        {/* Optional: -6dB marker line */}
        <div
          className="absolute opacity-30"
          style={{
            backgroundColor: '#71717a', // zinc-500
            [isVertical ? 'bottom' : 'left']: '50%',
            [isVertical ? 'width' : 'height']: '100%',
            [isVertical ? 'height' : 'width']: 1,
          }}
        />
      </div>
    );
  }
);

LevelMeter.displayName = 'LevelMeter';
