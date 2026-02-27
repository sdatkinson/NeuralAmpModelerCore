import React from 'react';
import { Play } from './Play';
import { Skip } from './Skip';
import { Pause } from './Pause';
import { CircularLoader } from './CircularLoader';
import { formatTime } from '../../utils/player';

const CANVAS_STYLE = {
  marginBottom: -20,
  marginTop: -20,
  width: '100%',
  height: '130px',
} as const;

export interface DemoPlaybarProps {
  togglePlay: () => Promise<void>;
  isThisPlayerActive: boolean;
  isLoading: boolean;
  handleSkipToStart: () => void;
  currentTime: number;
  duration: number;
  canvasWrapperRef: React.RefObject<HTMLDivElement>;
  visualizerRef: React.RefObject<HTMLCanvasElement>;
  infoSlot?: React.ReactNode;
}

export const DemoPlaybar: React.FC<DemoPlaybarProps> = ({
  togglePlay,
  isThisPlayerActive,
  isLoading,
  handleSkipToStart,
  currentTime,
  duration,
  canvasWrapperRef,
  visualizerRef,
  infoSlot,
}) => (
  <div className='flex items-center gap-4 overflow-hidden w-full'>
    <button
      onClick={togglePlay}
      className='p-0 focus:outline-none'
      aria-label={isThisPlayerActive ? 'Pause' : 'Play'}
    >
      {isThisPlayerActive ? (
        <Pause />
      ) : isLoading ? (
        <CircularLoader size={48} />
      ) : (
        <Play />
      )}
    </button>

    <button
      onClick={handleSkipToStart}
      className='p-0 focus:outline-none'
      aria-label='Skip to start'
    >
      <Skip opacity={currentTime > 0 ? 1 : 0.6} />
    </button>

    <div className='hidden sm:flex text-sm font-mono gap-2 text-zinc-400'>
      <span>{formatTime(currentTime)}</span>
      <span> / </span>
      <span>{formatTime(duration)}</span>
    </div>

    <div ref={canvasWrapperRef} className='flex-1'>
      <canvas ref={visualizerRef} height={130} style={CANVAS_STYLE} />
    </div>
    {infoSlot && infoSlot}
  </div>
);
