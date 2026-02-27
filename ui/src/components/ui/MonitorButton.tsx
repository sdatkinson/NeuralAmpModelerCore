import React, { forwardRef, useRef, useEffect, useState } from 'react';
import { calculateLevels, scaleForDisplay } from '../../utils/metering';

const LED_SIZE = 36;
const INNER_CIRCLE_SIZE = 16;
const RING_GAP = 2;
const RING_STROKE = 2;

interface MonitorButtonProps {
  isMonitoring: boolean;
  onClick: () => void;
  analyserNode: AnalyserNode | null;
  size?: number;
}

export const MonitorButton = forwardRef<HTMLButtonElement, MonitorButtonProps>(
  ({ isMonitoring, onClick, analyserNode, size = LED_SIZE }, ref) => {
    const internalRef = useRef<HTMLButtonElement>(null);
    const buttonRef =
      (ref as React.RefObject<HTMLButtonElement>) || internalRef;
    const bufferRef = useRef<Float32Array>(new Float32Array(2048));

    const center = size / 2;
    const innerRadius = INNER_CIRCLE_SIZE / 2;
    const ringMinRadius = innerRadius + RING_GAP + RING_STROKE / 2;
    const ringMaxRadius = size / 2 - RING_STROKE / 2 - 1;

    const [ringRadius, setRingRadius] = useState(ringMinRadius);

    useEffect(() => {
      if (!isMonitoring || !analyserNode) {
        setRingRadius(ringMinRadius);
        return;
      }

      let running = true;
      const buffer = bufferRef.current;

      const animate = () => {
        if (!running) return;
        const levels = calculateLevels(analyserNode, buffer);
        const glow = scaleForDisplay(levels.rms);
        const radius = ringMinRadius + glow * (ringMaxRadius - ringMinRadius);
        setRingRadius(radius);
        requestAnimationFrame(animate);
      };

      requestAnimationFrame(animate);
      return () => {
        running = false;
      };
    }, [isMonitoring, analyserNode, ringMinRadius, ringMaxRadius]);

    const innerColor = isMonitoring ? '#f00' : '#a1a1aa';
    const ringColor = isMonitoring ? '#f00' : '#a1a1aa';

    return (
      <div className='flex items-center justify-center'>
        <button
          ref={buttonRef}
          type='button'
          onClick={onClick}
          aria-label={isMonitoring ? 'Stop monitoring' : 'Start monitoring'}
          aria-pressed={isMonitoring}
          className={`
            monitor-button-reset-focus rounded-full flex-shrink-0 transition-all duration-150 ease-in
            focus:outline-none focus:ring-0 focus:ring-offset-0
          `}
          style={{
            width: size,
            height: size,
            minWidth: size,
            minHeight: size,
            background: 'transparent',
            border: 'none',
            padding: 0,
            position: 'relative',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            WebkitTapHighlightColor: 'transparent',
          }}
        >
          <svg
            width={size}
            height={size}
            viewBox={`0 0 ${size} ${size}`}
            style={{ position: 'absolute', inset: 0, overflow: 'visible' }}
          >
            {/* Inner filled circle - 16x16 */}
            <circle cx={center} cy={center} r={innerRadius} fill={innerColor} />
            {/* Outer ring - grows with sound */}
            <circle
              cx={center}
              cy={center}
              r={ringRadius}
              fill='none'
              stroke={ringColor}
              strokeWidth={RING_STROKE}
            />
          </svg>
        </button>
        <span
          className={`text-xs ${isMonitoring ? 'text-white' : 'text-zinc-400'}`}
        >
          LIVE
        </span>
      </div>
    );
  }
);

MonitorButton.displayName = 'MonitorButton';
