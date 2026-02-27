import React from 'react';
import { MonitorButton } from './MonitorButton';
import { useT3kPlayerContext } from '../../context/T3kPlayerContext';
import { SourceMode } from '../../types';

const CANVAS_STYLE = {
  marginBottom: -20,
  marginTop: -20,
  width: '100%',
  height: '130px',
} as const;

export interface PlayPlaybarProps {
  togglePlay: () => Promise<void>;
  isThisPlayerActive: boolean;
  sourceMode: SourceMode;
  isPlayConfigured: boolean;
  canvasWrapperRef: React.RefObject<HTMLDivElement>;
  visualizerRef: React.RefObject<HTMLCanvasElement>;
  infoSlot?: React.ReactNode;
  onOpenSettings: () => void;
}

export const PlayPlaybar: React.FC<PlayPlaybarProps> = ({
  togglePlay,
  isThisPlayerActive,
  sourceMode,
  isPlayConfigured,
  canvasWrapperRef,
  visualizerRef,
  infoSlot,
  onOpenSettings,
}) => {
  const isMonitoring = isThisPlayerActive && sourceMode === 'play';
  const { getAudioNodes, audioState } = useT3kPlayerContext();
  const nodes = getAudioNodes();
  const isReady = audioState.initState === 'ready';

  return (
    <div className='flex items-center gap-4 overflow-hidden w-full'>
      <MonitorButton
        isMonitoring={isMonitoring}
        onClick={!isPlayConfigured ? onOpenSettings : togglePlay}
        analyserNode={isMonitoring && isReady ? nodes.inputMeterNode : null}
        size={48}
      />

      <div ref={canvasWrapperRef} className='flex-1'>
        <canvas ref={visualizerRef} height={130} style={CANVAS_STYLE} />
      </div>
      {infoSlot && infoSlot}
    </div>
  );
};
