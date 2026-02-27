import React from 'react';

interface PlayIconProps {
  size?: number;
}

export const PlayIcon = ({ size = 20 }: PlayIconProps) => (
  <svg
    width={size}
    height={size}
    viewBox='0 0 20 20'
    fill='none'
    xmlns='http://www.w3.org/2000/svg'
  >
    <circle cx='10' cy='10' r='8.5' stroke='currentColor' />
    <circle cx='10' cy='10' r='6' fill='currentColor' />
  </svg>
);
