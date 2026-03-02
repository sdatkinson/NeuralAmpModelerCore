import React from 'react';
import { Button } from '../ui/Button';
import { Guitar } from '../ui/Guitar';

interface LiveTakeoverProps {
  onConnect: () => void | Promise<void>;
}

// Height of placeholder is 254px to match the "demo" mode height so there is not a jump when switching modes

export const LiveTakeover: React.FC<LiveTakeoverProps> = ({ onConnect }) => {
  return (
    <div className='flex flex-col items-center justify-center gap-6 bg-zinc-900 rounded-lg min-h-[254px]'>
      <div className='flex flex-col items-center gap-6'>
        <div className='flex flex-col items-center gap-2 text-center'>
          <Guitar />
          <h3 className='font-semibold text-white'>Play Live</h3>
          <p className='text-base text-zinc-400 max-w-[360px]'>
            Play tones directly on the TONE3000 website by connecting your
            instrument to an audio interface.
          </p>
        </div>
        <Button variant='secondary' onClick={onConnect}>
          Connect
        </Button>
      </div>
    </div>
  );
};
