import React from 'react';

interface InterfaceIconProps {
  size?: number;
}

export const InterfaceIcon = ({ size = 20 }: InterfaceIconProps) => (
  <svg
    width={size}
    height={size}
    viewBox='0 0 20 20'
    fill='none'
    xmlns='http://www.w3.org/2000/svg'
  >
    <path
      d='M8.3335 13.3333H8.34183'
      stroke='currentColor'
      strokeWidth='1.66667'
      strokeLinecap='round'
      strokeLinejoin='round'
    />
    <path
      d='M1.84341 9.64749C1.72726 9.87926 1.66677 10.1349 1.66675 10.3942V15C1.66675 15.442 1.84234 15.8659 2.1549 16.1785C2.46746 16.4911 2.89139 16.6667 3.33341 16.6667H16.6667C17.1088 16.6667 17.5327 16.4911 17.8453 16.1785C18.1578 15.8659 18.3334 15.442 18.3334 15V10.3942C18.3334 10.1349 18.2729 9.87926 18.1567 9.64749L15.4584 4.25833C15.3204 3.98065 15.1077 3.74697 14.8442 3.58356C14.5807 3.42015 14.2768 3.33349 13.9667 3.33333H6.03341C5.72334 3.33349 5.41947 3.42015 5.15595 3.58356C4.89244 3.74697 4.67973 3.98065 4.54175 4.25833L1.84341 9.64749Z'
      stroke='currentColor'
      strokeWidth='1.66667'
      strokeLinecap='round'
      strokeLinejoin='round'
    />
    <path
      d='M18.2886 10.0108H1.71191'
      stroke='currentColor'
      strokeWidth='1.66667'
      strokeLinecap='round'
      strokeLinejoin='round'
    />
    <path
      d='M5 13.3333H5.00833'
      stroke='currentColor'
      strokeWidth='1.66667'
      strokeLinecap='round'
      strokeLinejoin='round'
    />
  </svg>
);
