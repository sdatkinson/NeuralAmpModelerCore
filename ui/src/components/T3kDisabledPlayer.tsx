import React from 'react';
import { DisabledPlaybar } from './ui/DisabledPlaybar';

export interface T3kDisabledPlayerProps {
  infoSlot?: React.ReactNode;
}

export const T3kDisabledPlayer: React.FC<T3kDisabledPlayerProps> = ({
  infoSlot,
}) => (
  <div className='neural-amp-modeler'>
    <div className='bg-zinc-900 border border-zinc-700 text-white px-4 lg:px-8 py-4 rounded-xl w-full flex items-center min-h-[80px]'>
      <DisabledPlaybar infoSlot={infoSlot} />
    </div>
  </div>
);

export default T3kDisabledPlayer;
