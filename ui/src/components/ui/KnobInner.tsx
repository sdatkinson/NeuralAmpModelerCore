import React from 'react';

interface KnobInnerProps {
  value: number; // 0 to 1
  size: number;
}

export const KnobInner: React.FC<KnobInnerProps> = ({ value, size }) => {
  // Angle calculation: value 0 = -135°, value 0.5 = 0° (top), value 1 = +135°
  const angleDeg = value * 270 - 135;
  // Convert to radians, offset by -90 so 0° is at top
  const angleRad = (angleDeg - 90) * (Math.PI / 180);

  // Sweep angle from start (0) to current value position (rounded to reduce jitter)
  const sweepDeg = Math.round(value * 270 * 10) / 10;

  const outerRadius = size / 2;
  const innerRadius = outerRadius * 0.5; // Thinner arc, larger inner circle
  const arcCenterRadius = (outerRadius + innerRadius) / 2;

  // Handle dimensions - positioned in the middle of the arc ring
  const handleWidth = Math.round(size * 0.12);
  const handleHeight = Math.round((outerRadius - innerRadius) * 1.2);

  // Calculate handle offset from center (use transform for smoother animation)
  const handleOffsetX = arcCenterRadius * Math.cos(angleRad);
  const handleOffsetY = arcCenterRadius * Math.sin(angleRad);

  // Build dynamic gradient that only extends to the handle position
  const buildGradient = () => {
    if (sweepDeg <= 0) return 'transparent';

    // Create gradient stops proportionally within the sweep range
    const stops = [
      { pos: 0, color: 'rgba(30, 30, 30, 0.3)' }, // was 0.5
      { pos: 0.2, color: 'rgba(50, 50, 50, 0.6)' },
      { pos: 0.4, color: 'rgba(100, 100, 100, 0.75)' },
      { pos: 0.6, color: 'rgba(160, 160, 160, 0.85)' },
      { pos: 0.8, color: 'rgba(220, 220, 220, 0.9)' },
      { pos: 1, color: 'rgba(255, 255, 255, 1)' },
    ];

    const gradientStops = stops
      .map(s => `${s.color} ${s.pos * sweepDeg}deg`)
      .join(', ');

    return `conic-gradient(from 225deg, ${gradientStops}, transparent ${sweepDeg}deg)`;
  };

  return (
    <div
      style={{
        width: size,
        height: size,
        position: 'relative',
        borderRadius: '50%',
        pointerEvents: 'none',
      }}
    >
      {/* Base track - always shows full 270° arc */}
      <div
        style={{
          position: 'absolute',
          width: '100%',
          height: '100%',
          borderRadius: '50%',
          background: `conic-gradient(from 225deg, rgba(50, 50, 50, 0.8) 0deg, rgba(50, 50, 50, 0.1) 270deg, transparent 270deg)`,
        }}
      />

      {/* Gradient overlay - extends from start to handle position */}
      <div
        style={{
          position: 'absolute',
          width: '100%',
          height: '100%',
          borderRadius: '50%',
          background: buildGradient(),
          willChange: 'background',
          backfaceVisibility: 'hidden',
        }}
      />

      {/* Inner dark circle - cuts out the center to create the ring */}
      <div
        style={{
          position: 'absolute',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          width: innerRadius * 2,
          height: innerRadius * 2,
          borderRadius: '50%',
          backgroundColor: '#1C1C1E',
        }}
      />

      {/* Handle/indicator - uses transform for smoother animation */}
      <div
        style={{
          position: 'absolute',
          left: '50%',
          top: '50%',
          width: handleWidth,
          height: handleHeight,
          backgroundColor: '#ffffff',
          borderLeft: '2px solid #000000',
          transform: `translate(-50%, -50%) translate(${handleOffsetX}px, ${handleOffsetY}px) rotate(${angleDeg}deg)`,
          transformOrigin: 'center center',
          willChange: 'transform',
          backfaceVisibility: 'hidden',
        }}
      />
    </div>
  );
};
