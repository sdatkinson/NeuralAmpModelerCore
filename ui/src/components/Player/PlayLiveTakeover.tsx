import React from 'react';
import { Button } from '../ui/Button';
import { Guitar } from '../ui/Guitar';

interface PlayLiveTakeoverProps {
  onConnect: () => void | Promise<void>;
}

export const PlayLiveTakeover: React.FC<PlayLiveTakeoverProps> = ({
  onConnect,
}) => {
  return (
    <div className='flex flex-col items-center justify-center gap-6 bg-zinc-900 rounded-lg'>
      <div className='flex flex-col items-center gap-6'>
        <div className='flex flex-col items-center gap-2 text-center'>
          <Guitar />
          <h3 className='font-semibold text-white'>Play Live</h3>
          <p className='text-sm text-zinc-400 max-w-[360px]'>
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
