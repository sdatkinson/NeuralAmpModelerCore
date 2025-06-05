import React from 'react';
import { T3kPlayerProps } from '../types';
import Player from './Player/Player';
import { PlayerSkeleton } from './Player/PlayerSkeleton';
import { DEFAULT_INPUTS, DEFAULT_IRS, DEFAULT_MODELS } from '../constants';

export const T3kPlayer: React.FC<T3kPlayerProps> = ({
  models,
  irs,
  inputs,
  isLoading,
}) => {
  // use default models, irs, and inputs if not provided
  const _models = models?.length ? models : DEFAULT_MODELS;
  const _irs = irs?.length ? irs : DEFAULT_IRS;
  const _inputs = inputs?.length ? inputs : DEFAULT_INPUTS;

  if (isLoading) {
    return <PlayerSkeleton />;
  }

  return (
    <div className="neural-amp-modeler">
      <Player
        models={_models}
        irs={_irs}
        inputs={_inputs}
      />
    </div>
  );
};

export default T3kPlayer; 