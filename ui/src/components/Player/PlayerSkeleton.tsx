import React, { useRef } from 'react';
import { PREVIEW_MODE } from '../../types';
import { LogoSm } from '../ui/LogoSm';
import { Play } from '../ui/Play';
import { Skip } from '../ui/Skip';
import { Select } from '../ui/Select';
import { ToggleSimple } from '../ui/ToggleSimple';

interface PlayerSkeletonProps {
  previewMode: PREVIEW_MODE;
}

export const PlayerSkeleton: React.FC<PlayerSkeletonProps> = ({
  previewMode,
}) => {
  const visualizerRef = useRef<HTMLCanvasElement>(null);
  const canvasWrapperRef = useRef<HTMLDivElement>(null);

  const renderModelSelect = () => (
    <Select
      options={[]}
      label='Model'
      onChange={() => {}}
      defaultOption={''}
      disabled={true}
    />
  );

  const renderIrSelect = () => (
    <Select
      options={[]}
      label='IR'
      onChange={() => {}}
      defaultOption={''}
      disabled={true}
    />
  );

  return (
    <div className='bg-zinc-900 border border-zinc-700 text-white p-4 lg:p-8 pt-0 lg:pt-2 rounded-xl w-full flex flex-col gap-6 opacity-50 cursor-not-allowed'>
      {/* Player Controls */}
      <div className='flex items-center gap-4 overflow-hidden'>
        <button
          className='p-0 focus:outline-none'
          aria-label={'Play'}
          disabled={true}
        >
          <Play />
        </button>

        <button
          className='p-0 focus:outline-none'
          aria-label='Skip to start'
          disabled={true}
        >
          <Skip opacity={0.6} />
        </button>

        <div className='flex text-sm font-mono gap-2 text-zinc-400'>
          <span>00:00</span>
          <span> / </span>
          <span>00:00</span>
        </div>

        <div ref={canvasWrapperRef} className='flex-1'>
          <canvas
            ref={visualizerRef}
            height={130}
            style={{
              marginBottom: -20,
              marginTop: -20,
              width: '100%',
              height: '130px',
            }}
          />
        </div>
      </div>

      {/* Settings */}
      <div className='flex flex-col gap-2'>
        <div className='flex flex-row items-center gap-4 flex-wrap'>
          <div className='flex-1 min-w-[0px]'>
            <div className={`flex w-full flex-1 min-w-[0px]`}>
              {previewMode === PREVIEW_MODE.MODEL
                ? renderModelSelect()
                : renderIrSelect()}
            </div>
          </div>

          <div className='flex items-center pt-[24px] flex-shrink-0'>
            <ToggleSimple
              label=''
              onChange={() => {}}
              isChecked={true}
              ariaLabel='Bypass'
              disabled={true}
            />
          </div>
        </div>

        <div className='flex flex-col sm:flex-row items-center gap-2 sm:gap-6'>
          <div className='w-full sm:w-1/2'>
            <Select
              options={[]}
              label='Input'
              onChange={() => {}}
              defaultOption={''}
              disabled={true}
            />
          </div>

          <div className={`w-full sm:w-1/2`}>
            {previewMode === PREVIEW_MODE.MODEL
              ? renderIrSelect()
              : renderModelSelect()}
          </div>
        </div>
      </div>

      {/* Footer */}
      <a
        href='https://www.tone3000.com'
        target='_blank'
        className='flex flex-row gap-2 items-center self-end'
      >
        <p className='text-zinc-400 text-xs'>Powered by</p>
        <LogoSm width={42} height={14} />
      </a>
    </div>
  );
};
