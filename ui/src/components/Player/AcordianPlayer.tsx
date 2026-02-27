import React, { memo, useCallback, useState } from 'react';
import { Input, IR, Model, T3kAcordianPlayerProps } from '../../types';
import { PlayIcon } from '../ui/PlayIcon';
import { Demo } from '../ui/Demo';
import { LogoSm } from '../ui/LogoSm';
import {
  DEFAULT_INPUTS,
  DEFAULT_MODELS,
  DEFAULT_IRS,
  SOURCE_MODE_OPTIONS,
} from '../../constants';
import { ChevronDown, ChevronUp } from 'lucide-react';
import { Tabs } from '../ui/Tabs';
import { usePlayerCore } from '../../hooks/usePlayerCore';
import { PlayerSettings } from './PlayerSettings';
import { getDefault } from '../../utils/player';
import { DemoPlaybar } from '../ui/DemoPlaybar';
import { PlayPlaybar } from '../ui/PlayPlaybar';

const PlayerFC: React.FC<T3kAcordianPlayerProps> = ({
  getData = async () => ({
    models: DEFAULT_MODELS,
    irs: DEFAULT_IRS,
    inputs: DEFAULT_INPUTS,
  }),
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
    <div
      className={`bg-zinc-900 border border-t-0 border-zinc-700 text-white px-4 sm:px-6 pt-0 pb-0 rounded-xl w-full flex flex-col gap-0 rounded-t-none ${expanded ? 'pb-4 lg:pb-8' : 'pb-0'}`}
    >
      <div
        className={`flex items-center gap-4 overflow-hidden min-h-[80px] ${disabled ? 'opacity-50 touch-none cursor-not-allowed' : ''}`}
      >
        {core.sourceMode === 'demo' ? (
          <DemoPlaybar
            togglePlay={core.togglePlay}
            isThisPlayerActive={core.isThisPlayerActive}
            isLoading={core.isLoading}
            handleSkipToStart={core.handleSkipToStart}
            currentTime={core.currentTime}
            duration={core.duration}
            canvasWrapperRef={core.canvasWrapperRef}
            visualizerRef={core.visualizerRef}
            infoSlot={infoSlot}
          />
        ) : (
          <PlayPlaybar
            togglePlay={core.togglePlay}
            isThisPlayerActive={core.isThisPlayerActive}
            sourceMode={core.sourceMode}
            isPlayConfigured={core.isPlayConfigured}
            canvasWrapperRef={core.canvasWrapperRef}
            visualizerRef={core.visualizerRef}
            infoSlot={infoSlot}
            onOpenSettings={core.openSettingsDialog}
          />
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
          <div className='flex flex-col gap-6'>
            {/* Toggle between demo and play */}
            <Tabs
              tabs={SOURCE_MODE_OPTIONS.map(o => (
                <div className='flex gap-2 items-center'>
                  {o.value === 'demo' ? (
                    <Demo size={20} />
                  ) : (
                    <PlayIcon size={20} />
                  )}
                  <span>{o.label}</span>
                </div>
              ))}
              activeTab={SOURCE_MODE_OPTIONS.findIndex(
                o => o.value === core.sourceMode
              )}
              setActiveTab={index =>
                core.handleSourceModeChange(SOURCE_MODE_OPTIONS[index].value)
              }
            />
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
              isPlayConfigured={core.isPlayConfigured}
              currentDeviceId={core.currentDeviceId}
              playDeviceOptions={core.playDeviceOptions}
              inputModeType={core.inputModeType}
              audioInputError={core.audioInputDevices.error}
              isThisPlayerActive={core.isThisPlayerActive}
            />
          </div>

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
      JSON.stringify(prevProps.previewMode) ===
        JSON.stringify(nextProps.previewMode) &&
      prevProps.getData === nextProps.getData &&
      prevProps.disabled === nextProps.disabled
    );
  }
);

T3kAcordianPlayer.displayName = 'T3kAcordianPlayer';

export default T3kAcordianPlayer;
