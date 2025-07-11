import React from 'react';
import { PREVIEW_MODE } from '../../types';
import { LogoSm } from '../ui/LogoSm';

interface PlayerSkeletonProps {
  previewMode: PREVIEW_MODE;
}

export const PlayerSkeleton: React.FC<PlayerSkeletonProps> = ({ previewMode }) => {
  return (
    <div className='bg-zinc-900 border border-zinc-700 text-white p-4 lg:p-8 pt-0 lg:pt-2 rounded-xl w-full opacity-50 flex flex-col gap-6'>
      <div className='flex items-center gap-4 overflow-hidden'>
        <button className='p-0 focus:outline-none cursor-not-allowed' disabled>
          <LogoSm />
        </button>
        <button
          className='p-0 focus:outline-none cursor-not-allowed opacity-60'
          disabled
        >
          <span className='text-xs text-zinc-400'>v1.0.23</span>
        </button>
      </div>

      <div className='flex text-sm font-mono gap-2 text-zinc-400'>
        <span>Loading...</span>
      </div>

      <div className='flex-1'>
        <canvas
          className='w-full h-[130px] bg-zinc-900 rounded'
          width={800}
          height={130}
        />
      </div>

      <div className='flex flex-col gap-2'>
        <div className='flex flex-row items-center gap-4 flex-wrap'>
          <div className='flex-1 min-w-[0px]'>
            <div className='flex w-full flex-1 min-w-[0px]'>
              <div className='w-full h-8 bg-zinc-800 rounded animate-pulse'></div>
            </div>
          </div>
        </div>
      </div>

      <div className='flex items-center pt-[24px] flex-shrink-0'>
        <div className='w-full h-8 bg-zinc-800 rounded animate-pulse'></div>
      </div>

      <div className='flex flex-col sm:flex-row items-center gap-2 sm:gap-6'>
        <div className='w-full sm:w-1/2'>
          <div className='w-full h-8 bg-zinc-800 rounded animate-pulse'></div>
        </div>
        <div className='w-full sm:w-1/2'>
          <div className='w-full h-8 bg-zinc-800 rounded animate-pulse'></div>
        </div>
      </div>

      <div className='flex flex-row gap-2 items-center self-end'>
        <LogoSm />
        <p className='text-zinc-400 text-xs'>Powered by</p>
      </div>
    </div>
  );
};
