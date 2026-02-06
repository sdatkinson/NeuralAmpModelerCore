import React, { memo } from 'react';
import { PREVIEW_MODE, T3kPlayerProps } from '../../types';
import { Play } from '../ui/Play';
import { Skip } from '../ui/Skip';
import { Pause } from '../ui/Pause';
import { LogoSm } from '../ui/LogoSm';
import { DEFAULT_INPUTS, DEFAULT_MODELS, DEFAULT_IRS } from '../../constants';
import { CircularLoader } from '../ui/CircularLoader';
import { InputControlStrip } from '../InputControlStrip';
import { usePlayerCore, formatTime } from '../../hooks/usePlayerCore';
import { PlayerSettings } from './PlayerSettings';

const PlayerFC: React.FC<T3kPlayerProps> = ({
  models = DEFAULT_MODELS,
  irs = DEFAULT_IRS,
  inputs = DEFAULT_INPUTS,
  previewMode,
  onPlay,
  onModelChange,
  onInputChange,
  onIrChange,
  id,
  infoSlot,
}) => {
  const core = usePlayerCore({
    id,
    previewMode,
    models,
    irs,
    inputs,
    onPlay,
    onModelChange,
    onInputChange,
    onIrChange,
  });

  return (
    <div className='bg-zinc-900 border border-zinc-700 text-white p-4 lg:p-8 pt-0 lg:pt-2 rounded-xl w-full flex flex-col gap-6'>
      {/* Player Controls */}
      <div className='flex items-center gap-4 overflow-hidden'>
        <button
          onClick={core.togglePlay}
          className={`p-0 focus:outline-none ${core.sourceMode === 'live' && !core.isLiveConfigured ? 'opacity-50 cursor-not-allowed' : ''}`}
          disabled={core.sourceMode === 'live' && !core.isLiveConfigured}
          aria-label={core.isThisPlayerActive ? 'Pause' : 'Play'}
        >
          {core.isThisPlayerActive ? (
            <Pause />
          ) : core.isLoading ? (
            <CircularLoader size={48} />
          ) : (
            <Play />
          )}
        </button>

        <button
          onClick={core.handleSkipToStart}
          className={`p-0 focus:outline-none ${core.sourceMode === 'live' ? 'opacity-50 cursor-not-allowed' : ''}`}
          disabled={core.sourceMode === 'live'}
          aria-label='Skip to start'
        >
          <Skip opacity={core.sourceMode === 'live' ? 0.6 : core.currentTime > 0 ? 1 : 0.6} />
        </button>

        <div className='hidden sm:flex text-sm font-mono gap-2 text-zinc-400'>
          <span>{core.sourceMode === 'live' ? '-:--' : formatTime(core.currentTime)}</span>
          <span> / </span>
          <span>{core.sourceMode === 'live' ? '-:--' : formatTime(core.duration)}</span>
        </div>

        <div ref={core.canvasWrapperRef} className='flex-1'>
          <canvas
            ref={core.visualizerRef}
            height={130}
            style={{
              marginBottom: -20,
              marginTop: -20,
              width: '100%',
              height: '130px',
            }}
          />
        </div>
        {infoSlot && infoSlot}
      </div>

      {/* Settings */}
      <PlayerSettings
        previewMode={previewMode}
        bypassed={core.bypassed}
        bypassedStyles={core.bypassedStyles}
        onBypassToggle={core.handleBypassToggle}
        sourceMode={core.sourceMode}
        onSourceModeChange={core.handleSourceModeChange}
        showPlaybackPausedMessage={core.showPlaybackPausedMessage}
        toastMessage={core.toastMessage}
        onOpenSettings={core.openSettingsDialog}
        modelOptions={core.modelOptions}
        irOptions={core.irOptions}
        audioOptions={core.audioOptions}
        selectedModelUrl={core.selectedModel?.url ?? ''}
        selectedIrUrl={core.selectedIr?.url ?? ''}
        selectedInputUrl={core.selectedInput?.url ?? ''}
        onModelChange={core.handleModelChange}
        onIrChange={core.handleIrChange}
        onInputChange={core.handleInputChange}
        isLiveConfigured={core.isLiveConfigured}
        currentDeviceId={core.currentDeviceId}
        liveDeviceOptions={core.liveDeviceOptions}
        onLiveDeviceChange={core.handleLiveDeviceChange}
        inputModeType={core.inputModeType}
        audioInputError={core.audioInputDevices.error}
      />

      {/* Live Input Control Strip */}
      {core.sourceMode === 'live' && core.isLiveConfigured && (
        <InputControlStrip isActive={core.isThisPlayerActive} />
      )}

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

const Player = memo(
  (props: T3kPlayerProps) => {
    return <PlayerFC {...props} />;
  },
  (prevProps, nextProps) => {
    return (
      JSON.stringify(prevProps.models) === JSON.stringify(nextProps.models) &&
      JSON.stringify(prevProps.irs) === JSON.stringify(nextProps.irs) &&
      JSON.stringify(prevProps.inputs) === JSON.stringify(nextProps.inputs)
    );
  }
);

Player.displayName = 'Player';

export default Player;
