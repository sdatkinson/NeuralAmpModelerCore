import React from 'react';
import { PREVIEW_MODE, T3kPlayerProps } from '../types';
import Player from './Player/Player';
import { PlayerSkeleton } from './Player/PlayerSkeleton';

export const T3kPlayer: React.FC<T3kPlayerProps> = ({
  isLoading,
  previewMode = PREVIEW_MODE.MODEL,
  ...props
}) => {
  if (isLoading) {
    return (
      <div className='neural-amp-modeler'>
        <PlayerSkeleton previewMode={previewMode} />
      </div>
    );
  }

  return (
    <div className='neural-amp-modeler'>
      <Player {...props} previewMode={previewMode} />
    </div>
  );
};

export default T3kPlayer;
