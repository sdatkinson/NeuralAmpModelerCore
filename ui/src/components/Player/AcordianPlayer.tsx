import React, { memo, useCallback, useState } from 'react';
import { Input, IR, Model, T3kAcordianPlayerProps } from '../../types';
import { Play } from '../ui/Play';
import { Skip } from '../ui/Skip';
import { Pause } from '../ui/Pause';
import { LogoSm } from '../ui/LogoSm';
import { DEFAULT_INPUTS, DEFAULT_MODELS, DEFAULT_IRS, SOURCE_MODE_OPTIONS } from '../../constants';
import { CircularLoader } from '../ui/CircularLoader';
import { ChevronDown, ChevronUp, Settings } from 'lucide-react';
import { InputControlStrip } from '../InputControlStrip';
import { SegmentedControl } from '../ui/SegmentedControl';
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
    if (!models) await resolveData();
    setExpanded(prev => !prev);
  }, [models, resolveData]);

  return (
    <div className={`bg-zinc-900 border border-t-0 border-zinc-700 text-white px-4 sm:px-6 pt-0 pb-0 rounded-xl w-full flex flex-col gap-0 md:gap-1 rounded-t-none ${expanded ? 'pb-4 lg:pb-8' : 'pb-4'}`}>
      {/* Top Bar */}
      <div className={`flex items-center justify-between pt-2 ${disabled ? 'opacity-50' : ''}`}>
        <SegmentedControl
          options={SOURCE_MODE_OPTIONS}
          value={core.sourceMode}
          onChange={core.handleSourceModeChange}
          disabled={disabled}
        />
        <div className='flex items-center gap-2'>
          {core.showPlaybackPausedMessage && (
            <span className='text-xs text-zinc-400 animate-pulse'>Playback paused</span>
          )}
          {core.toastMessage && (
            <span className='text-xs text-zinc-400 animate-pulse'>{core.toastMessage}</span>
          )}
          <button
            onClick={core.openSettingsDialog}
            className='p-2 rounded-md transition-colors border border-zinc-700 hover:bg-zinc-800'
            aria-label='Settings'
            disabled={disabled}
          >
            <Settings size={20} className='text-zinc-400' />
          </button>
        </div>
      </div>

      {/* Player Area — fixed height so preview↔live doesn't shift layout */}
      <div className={`flex items-center gap-4 overflow-hidden min-h-[96px] ${disabled ? 'opacity-50 touch-none cursor-not-allowed' : ''}`}>
        {core.sourceMode === 'preview' ? (
          <>
            <button
              onClick={core.togglePlay}
              className='p-0 focus:outline-none'
              aria-label={core.isThisPlayerActive ? 'Pause' : 'Play'}
              disabled={disabled}
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
              className='p-0 focus:outline-none'
              aria-label='Skip to start'
              disabled={disabled}
            >
              <Skip size={24} opacity={core.currentTime > 0 ? 1 : 0.6} />
            </button>

            <div className='hidden sm:flex text-sm font-mono gap-2 text-zinc-400'>
              <span>{formatTime(core.currentTime)}</span>
              <span> / </span>
              <span>{formatTime(core.duration)}</span>
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
          </>
        ) : (
          <div className='flex-1'>
            <InputControlStrip
              isActive={core.isThisPlayerActive}
              isMonitoring={core.isThisPlayerActive && core.sourceMode === 'live'}
              onToggleMonitoring={core.togglePlay}
              disabled={disabled || !core.isLiveConfigured}
              compact
            />
          </div>
        )}

        <button
          onClick={handleToggleExpand}
          className={`p-0 focus:outline-none flex-shrink-0 ${disabled ? 'cursor-not-allowed' : ''}`}
          aria-label='Toggle accordion'
          disabled={disabled}
        >
          {expanded ? <ChevronUp size={24} /> : <ChevronDown size={24} />}
        </button>
      </div>

      {expanded && (
        <>
          <PlayerSettings
            previewMode={previewMode}
            disabled={disabled}
            bypassed={core.bypassed}
            bypassedStyles={core.bypassedStyles}
            onBypassToggle={core.handleBypassToggle}
            sourceMode={core.sourceMode}
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
