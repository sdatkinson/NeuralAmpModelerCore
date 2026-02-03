import React, { useCallback, useRef, useState, useEffect } from 'react';

export interface GainControlProps {
  /** Current value in dB */
  value: number;
  /** Callback when value changes */
  onChange: (value: number) => void;
  /** Minimum value in dB */
  min?: number;
  /** Maximum value in dB */
  max?: number;
  /** Step size for fine adjustment */
  step?: number;
  /** Label text */
  label?: string;
  /** Size of the knob in pixels */
  size?: number;
  /** Additional CSS classes */
  className?: string;
  /** Disabled state */
  disabled?: boolean;
}

/**
 * A gain control with rotary knob and text input for precise adjustment.
 * Values are in decibels (dB).
 *
 * - Drag knob to adjust value
 * - Scroll wheel on knob for fine adjustment
 * - Type directly in text input for precision
 * - Arrow keys for increment/decrement
 */
export function GainControl({
  value,
  onChange,
  min = -20,
  max = 20,
  step = 0.5,
  label = 'Gain',
  size = 48,
  className = '',
  disabled = false,
}: GainControlProps) {
  const knobRef = useRef<HTMLDivElement>(null);
  const [isDragging, setIsDragging] = useState(false);
  const dragStartRef = useRef<{ y: number; startValue: number } | null>(null);

  const clamp = useCallback(
    (v: number) => Math.min(Math.max(v, min), max),
    [min, max]
  );

  // Convert value to rotation angle (min = -135°, max = 135°)
  const valueToAngle = useCallback(
    (v: number) => {
      const normalized = (v - min) / (max - min); // 0 to 1
      return -135 + normalized * 270; // -135° to 135°
    },
    [min, max]
  );

  // Handle text input change
  const handleInputChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const parsed = parseFloat(e.target.value);
      if (!isNaN(parsed)) {
        onChange(clamp(parsed));
      }
    },
    [onChange, clamp]
  );

  // Handle keyboard input
  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLInputElement>) => {
      if (e.key === 'ArrowUp') {
        e.preventDefault();
        onChange(clamp(value + step));
      } else if (e.key === 'ArrowDown') {
        e.preventDefault();
        onChange(clamp(value - step));
      }
    },
    [onChange, value, step, clamp]
  );

  // Handle increment/decrement buttons
  const handleIncrement = useCallback(() => {
    if (disabled) return;
    onChange(clamp(value + step));
  }, [disabled, onChange, clamp, value, step]);

  const handleDecrement = useCallback(() => {
    if (disabled) return;
    onChange(clamp(value - step));
  }, [disabled, onChange, clamp, value, step]);

  // Handle mouse/touch drag on knob
  const handlePointerDown = useCallback(
    (e: React.PointerEvent) => {
      if (disabled) return;
      e.preventDefault();
      setIsDragging(true);
      dragStartRef.current = { y: e.clientY, startValue: value };
      (e.target as HTMLElement).setPointerCapture(e.pointerId);
    },
    [disabled, value]
  );

  const handlePointerMove = useCallback(
    (e: React.PointerEvent) => {
      if (!isDragging || !dragStartRef.current) return;

      // Vertical drag: up = increase, down = decrease
      // 100px of drag = full range
      const deltaY = dragStartRef.current.y - e.clientY;
      const range = max - min;
      const sensitivity = range / 100; // Full range over 100px
      const newValue = dragStartRef.current.startValue + deltaY * sensitivity;
      onChange(clamp(newValue));
    },
    [isDragging, onChange, clamp, min, max]
  );

  const handlePointerUp = useCallback(
    (e: React.PointerEvent) => {
      setIsDragging(false);
      dragStartRef.current = null;
      (e.target as HTMLElement).releasePointerCapture(e.pointerId);
    },
    []
  );

  // Handle scroll wheel on knob
  const handleWheel = useCallback(
    (e: React.WheelEvent) => {
      if (disabled) return;
      e.preventDefault();
      const delta = e.deltaY > 0 ? -step : step;
      onChange(clamp(value + delta));
    },
    [disabled, onChange, value, step, clamp]
  );

  // Handle double-click to reset to 0 dB
  const handleReset = useCallback(() => {
    if (disabled) return;
    onChange(clamp(0));
  }, [disabled, onChange, clamp]);

  // Prevent scroll when hovering over knob
  useEffect(() => {
    const knob = knobRef.current;
    if (!knob) return;

    const preventScroll = (e: WheelEvent) => {
      e.preventDefault();
    };

    knob.addEventListener('wheel', preventScroll, { passive: false });
    return () => knob.removeEventListener('wheel', preventScroll);
  }, []);

  const angle = valueToAngle(value);

  return (
    <div className={`flex flex-col items-center gap-1 ${className}`}>
      {label && (
        <label className="text-xs text-zinc-400">{label}</label>
      )}

      {/* Rotary knob */}
      <div
        ref={knobRef}
        className={`
          relative rounded-full select-none
          bg-zinc-800 border-2 border-zinc-600
          ${isDragging ? 'border-zinc-400' : 'hover:border-zinc-500'}
          ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
          transition-colors
        `}
        style={{ width: size, height: size }}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onPointerCancel={handlePointerUp}
        onWheel={handleWheel}
        onDoubleClick={handleReset}
        role="slider"
        aria-label={label}
        aria-valuemin={min}
        aria-valuemax={max}
        aria-valuenow={value}
        tabIndex={disabled ? -1 : 0}
      >
        {/* Indicator line */}
        <div
          className="absolute top-1/2 left-1/2 origin-bottom"
          style={{
            width: 2,
            height: size / 2 - 6,
            backgroundColor: disabled ? '#71717a' : '#fafafa',
            transform: `translate(-50%, -100%) rotate(${angle}deg)`,
            transformOrigin: 'bottom center',
            borderRadius: 1,
          }}
        />

        {/* Center dot */}
        <div
          className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 rounded-full bg-zinc-600"
          style={{ width: 6, height: 6 }}
        />
      </div>

      {/* Value input with +/- buttons */}
      <div className="flex items-center gap-1">
        {/* Decrement button */}
        <button
          type="button"
          onClick={handleDecrement}
          disabled={disabled || value <= min}
          className={`
            w-5 h-5 flex items-center justify-center
            rounded text-xs font-medium
            bg-zinc-700 text-zinc-300
            hover:bg-zinc-600 active:bg-zinc-500
            disabled:opacity-40 disabled:cursor-not-allowed disabled:hover:bg-zinc-700
            transition-colors
          `}
          aria-label={`Decrease ${label}`}
        >
          −
        </button>

        <div className="relative">
          <input
            type="number"
            value={value.toFixed(1)}
            onChange={handleInputChange}
            onKeyDown={handleKeyDown}
            onDoubleClick={handleReset}
            disabled={disabled}
            min={min}
            max={max}
            step={step}
            className="
              w-14 h-5 px-1
              text-center text-xs text-zinc-300
              bg-transparent border-none
              focus:outline-none
              disabled:opacity-50 disabled:cursor-not-allowed
              [appearance:textfield]
              [&::-webkit-outer-spin-button]:appearance-none
              [&::-webkit-inner-spin-button]:appearance-none
            "
            aria-label={`${label} value in dB`}
          />
          <span className="absolute right-0 bottom-1 text-[10px] text-zinc-500 pointer-events-none leading-none">
            dB
          </span>
        </div>

        {/* Increment button */}
        <button
          type="button"
          onClick={handleIncrement}
          disabled={disabled || value >= max}
          className={`
            w-5 h-5 flex items-center justify-center
            rounded text-xs font-medium
            bg-zinc-700 text-zinc-300
            hover:bg-zinc-600 active:bg-zinc-500
            disabled:opacity-40 disabled:cursor-not-allowed disabled:hover:bg-zinc-700
            transition-colors
          `}
          aria-label={`Increase ${label}`}
        >
          +
        </button>
      </div>
    </div>
  );
}
