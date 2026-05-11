import React, { useMemo } from 'react';
import { PREVIEW_MODE, T3kPlayerProps } from '../types';
import Player from './Player/Player';
import { PlayerSkeleton } from './Player/PlayerSkeleton';
import { T3kDisabledPlayer } from './T3kDisabledPlayer';
import { filterModelsByArchitecture } from '../utils/player';

export const T3kPlayer: React.FC<T3kPlayerProps> = ({
  isLoading,
  previewMode = PREVIEW_MODE.MODEL,
  architecture,
  disabled,
  models,
  infoSlot,
  ...props
}) => {
  const filteredModels = useMemo(() => {
    if (!architecture || !models) return models;
    const filtered = filterModelsByArchitecture(models, architecture);
    return filtered.length > 0
      ? (filtered as NonNullable<typeof models>)
      : undefined;
  }, [architecture, models]);

  if (isLoading) {
    return (
      <div className='neural-amp-modeler'>
        <PlayerSkeleton previewMode={previewMode} />
      </div>
    );
  }

  if (disabled || (architecture && !filteredModels)) {
    return <T3kDisabledPlayer infoSlot={infoSlot} />;
  }

  return (
    <div className='neural-amp-modeler'>
      <Player
        {...props}
        models={filteredModels}
        previewMode={previewMode}
        infoSlot={infoSlot}
      />
    </div>
  );
};

export default T3kPlayer;
