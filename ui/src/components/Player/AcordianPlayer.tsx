import React, { memo, useCallback, useState } from 'react';
import { Input, IR, Model, T3kAcordianPlayerProps } from '../../types';
import { Play } from '../ui/Play';
import { Skip } from '../ui/Skip';
import { Pause } from '../ui/Pause';
import { LogoSm } from '../ui/LogoSm';
import { DEFAULT_INPUTS, DEFAULT_MODELS, DEFAULT_IRS } from '../../constants';
import { CircularLoader } from '../ui/CircularLoader';
import { ChevronDown, ChevronUp } from 'lucide-react';
import { InputControlStrip } from '../InputControlStrip';
import { usePlayerCore, formatTime } from '../../hooks/usePlayerCore';
import { PlayerSettings } from './PlayerSettings';
import { getDefault } from '../../utils/player';

const PlayerFC: React.FC<T3kAcordianPlayerProps> = ({
  getData = async () => ({ models: DEFAULT_MODELS, irs: DEFAULT_IRS, inputs: DEFAULT_INPUTS }),
  previewMode,
  onPlay,
  onModelChange,
  onInputChange,
  onIrChange,
  id,
  disabled = false,
  infoSlot,
}) => {
  const [expanded, setExpanded] = useState(false);
  const [models, setModels] = useState<Model[] | null>(null);
  const [inputs, setInputs] = useState<Input[] | null>(null);
  const [irs, setIrs] = useState<IR[] | null>(null);

  // Lazy data resolver — fetches from getData and caches in local state
  const resolveData = useCallback(async () => {
    const data = await getData();
    setModels(data.models);
    setIrs(data.irs);
    setInputs(data.inputs);
    return {
      model: getDefault(data.models),
      ir: getDefault(data.irs),
      input: getDefault(data.inputs),
    };
  }, [getData]);

  const core = usePlayerCore({
    id,
    previewMode,
    disabled,
    models,
    irs,
    inputs,
    resolveData,
    onPlay,
    onModelChange,
    onInputChange,
    onIrChange,
  });

  const handleToggleExpand = useCallback(async () => {
    // Ensure data is loaded before expanding
    if (!models) await resolveData();
    setExpanded(prev => !prev);
  }, [models, resolveData]);

  return (
    <div className={`bg-zinc-900 border border-t-0 border-zinc-700 text-white px-4 sm:px-6 pt-0 pb-0 rounded-xl w-full flex flex-col gap-0 md:gap-1 rounded-t-none ${expanded ? 'pb-4 lg:pb-8' : 'pb-0'}`}>
      {/* Player Controls */}
      <div className={`flex items-center gap-4 overflow-hidden ${disabled ? 'opacity-50 touch-none cursor-not-allowed' : ''}`}>
        <button
          onClick={core.togglePlay}
          className={`p-0 focus:outline-none ${disabled || (core.sourceMode === 'live' && !core.isLiveConfigured) ? 'cursor-not-allowed opacity-50' : ''}`}
          aria-label={core.isThisPlayerActive ? 'Pause' : 'Play'}
          disabled={disabled || (core.sourceMode === 'live' && !core.isLiveConfigured)}
        >
          {core.isThisPlayerActive ? (
            <Pause size={40} />
          ) : core.isLoading ? (
            <CircularLoader size={40} />
          ) : (
            <Play size={40} />
          )}
        </button>

        <button
          onClick={core.handleSkipToStart}
          className={`p-0 focus:outline-none ${disabled || core.sourceMode === 'live' ? 'cursor-not-allowed' : ''}`}
          aria-label='Skip to start'
          disabled={disabled || core.sourceMode === 'live'}
        >
          <Skip size={24} opacity={core.sourceMode === 'live' ? 0.6 : core.currentTime > 0 ? 1 : 0.6} />
        </button>

        <div className='hidden sm:flex text-sm font-mono gap-2 text-zinc-400'>
          <span>{core.sourceMode === 'live' ? '-:--' : formatTime(core.currentTime)}</span>
          <span> / </span>
          <span>{core.sourceMode === 'live' ? '-:--' : formatTime(core.duration)}</span>
        </div>

        <div ref={core.canvasWrapperRef} className='flex-1'>
          <canvas
            ref={core.visualizerRef}
            height={120}
            style={{
              marginBottom: -20,
              marginTop: -20,
              width: '100%',
              height: '120px',
            }}
          />
        </div>

        {infoSlot && infoSlot}

        <button
          onClick={handleToggleExpand}
          className={`p-0 focus:outline-none ${disabled ? 'cursor-not-allowed' : ''}`}
          aria-label='Toggle accordion'
          disabled={disabled}
        >
          {expanded ? <ChevronUp size={24} /> : <ChevronDown size={24} />}
        </button>
      </div>

      {expanded && (
        <>
          {/* Settings */}
          <PlayerSettings
            previewMode={previewMode}
            disabled={disabled}
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
            <div className='pt-4'>
              <InputControlStrip isActive={core.isThisPlayerActive} />
            </div>
          )}

          {/* Footer */}
          <a
            href='https://www.tone3000.com'
            target='_blank'
            className='hidden flex flex-row gap-2 items-center self-end'
          >
            <p className='text-zinc-400 text-xs'>Powered by</p>
            <LogoSm width={42} height={14} />
          </a>
        </>
      )}
    </div>
  );
};

const T3kAcordianPlayer = memo(
  (props: T3kAcordianPlayerProps) => {
    return <PlayerFC {...props} />;
  },
  (prevProps, nextProps) => {
    return (
      JSON.stringify(prevProps.id) === JSON.stringify(nextProps.id) &&
      JSON.stringify(prevProps.previewMode) === JSON.stringify(nextProps.previewMode) &&
      prevProps.getData === nextProps.getData &&
      prevProps.disabled === nextProps.disabled
    );
  }
);

T3kAcordianPlayer.displayName = 'T3kAcordianPlayer';

export default T3kAcordianPlayer;
