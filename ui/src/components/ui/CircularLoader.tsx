import React from 'react';

type CircularLoaderProps = {
  size?: number;
  color?: string;
};

export const CircularLoader: React.FC<CircularLoaderProps> = ({
  size = 40,
  color = '#000',
}) => {
  const circleSize = size - 20;
  const strokeWidth = 3;
  const radius = (circleSize - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;

  return (
    <div
      className='flex items-center justify-center animate-spin bg-white rounded-full'
      style={{ width: size, height: size }}
    >
      <svg
        width={circleSize}
        height={circleSize}
        viewBox={`0 0 ${circleSize} ${circleSize}`}
        style={{ transform: 'rotate(-90deg)' }}
      >
        <circle
          cx={circleSize / 2}
          cy={circleSize / 2}
          r={radius}
          fill='none'
          stroke={color}
          strokeWidth={strokeWidth}
          strokeLinecap='round'
          strokeDasharray={circumference}
          strokeDashoffset={circumference * 0.75}
        />
      </svg>
    </div>
  );
};

export default CircularLoader;
