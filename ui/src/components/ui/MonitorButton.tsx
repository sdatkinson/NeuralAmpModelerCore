import React, { forwardRef, useRef, useEffect, useState } from 'react';
import { calculateLevels, scaleForDisplay } from '../../utils/metering';

const LED_SIZE = 36;

interface MonitorButtonProps {
  isMonitoring: boolean;
  onClick: () => void;
  analyserNode: AnalyserNode | null;
  disabled?: boolean;
  size?: number;
}

export const MonitorButton = forwardRef<HTMLButtonElement, MonitorButtonProps>(
  ({ isMonitoring, onClick, analyserNode, disabled = false, size = LED_SIZE }, ref) => {
    const internalRef = useRef<HTMLButtonElement>(null);
    const buttonRef = (ref as React.RefObject<HTMLButtonElement>) || internalRef;
    const bufferRef = useRef<Float32Array>(new Float32Array(2048));

    const [justActivated, setJustActivated] = useState(false);
    const prevMonitoring = useRef(isMonitoring);

    useEffect(() => {
      if (isMonitoring && !prevMonitoring.current) {
        setJustActivated(true);
        const timer = setTimeout(() => setJustActivated(false), 300);
        prevMonitoring.current = isMonitoring;
        return () => clearTimeout(timer);
      }
      prevMonitoring.current = isMonitoring;
    }, [isMonitoring]);

    useEffect(() => {
      if (!isMonitoring || !analyserNode) {
        buttonRef.current?.style.setProperty('--glow', '0');
        return;
      }

      let running = true;
      const buffer = bufferRef.current;

      const animate = () => {
        if (!running) return;
        const levels = calculateLevels(analyserNode, buffer);
        const glow = scaleForDisplay(levels.rms);
        buttonRef.current?.style.setProperty('--glow', String(glow));
        requestAnimationFrame(animate);
      };

      requestAnimationFrame(animate);
      return () => { running = false; };
    }, [isMonitoring, analyserNode, buttonRef]);

    return (
      <button
        ref={buttonRef}
        type="button"
        onClick={onClick}
        disabled={disabled}
        aria-label={isMonitoring ? 'Stop monitoring' : 'Start monitoring'}
        aria-pressed={isMonitoring}
        className={`
          rounded-full flex-shrink-0 transition-all duration-150 ease-in
          focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-zinc-900
          ${disabled ? 'opacity-30 cursor-not-allowed' : 'cursor-pointer'}
          ${isMonitoring ? 'focus:ring-green-500/50' : 'focus:ring-zinc-500/50'}
          ${justActivated ? 'animate-monitor-pop' : ''}
        `}
        style={{
          width: size,
          height: size,
          minWidth: size,
          minHeight: size,
          background: isMonitoring
            ? 'radial-gradient(circle at 50% 40%, #4ade80 0%, #22c55e 50%, #16a34a 100%)'
            : 'radial-gradient(circle at 50% 40%, #3f3f46 0%, #27272a 100%)',
          border: isMonitoring
            ? '1.5px solid #22c55e'
            : '1.5px solid #3f3f46',
          boxShadow: isMonitoring
            ? `inset 0 1px 2px rgba(255,255,255,0.15), 0 0 calc(var(--glow, 0) * 14px + 6px) calc(var(--glow, 0) * 5px + 2px) rgba(34,197,94, calc(var(--glow, 0) * 0.5 + 0.25)), 0 0 calc(var(--glow, 0) * 28px + 4px) calc(var(--glow, 0) * 10px + 1px) rgba(34,197,94, calc(var(--glow, 0) * 0.3 + 0.1))`
            : 'inset 0 1px 2px rgba(0,0,0,0.4), 0 1px 0 rgba(255,255,255,0.03)',
        }}
      />
    );
  }
);

MonitorButton.displayName = 'MonitorButton';
