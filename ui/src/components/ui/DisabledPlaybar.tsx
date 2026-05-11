import React from 'react';
import { Play } from './Play';
import { Skip } from './Skip';

export interface DisabledPlaybarProps {
  infoSlot?: React.ReactNode;
}

export const DisabledPlaybar: React.FC<DisabledPlaybarProps> = ({
  infoSlot,
}) => (
  <div className='flex items-center gap-4 overflow-hidden w-full'>
    <div
      className='p-0 cursor-not-allowed opacity-50'
      aria-label='Play (disabled)'
      aria-disabled='true'
    >
      <Play />
    </div>

    <div
      className='p-0 cursor-not-allowed opacity-50'
      aria-label='Skip to start (disabled)'
      aria-disabled='true'
    >
      <Skip opacity={0.6} />
    </div>

    <div className='hidden sm:flex text-sm font-mono gap-2 text-zinc-400 opacity-50'>
      <span>0:00</span>
      <span> / </span>
      <span>0:00</span>
    </div>

    <div className='flex-1 h-px bg-zinc-700' />

    {infoSlot}
  </div>
);
