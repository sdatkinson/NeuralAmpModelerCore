import React, { memo } from 'react';
import { T3kPlayerProps } from '../../types';
import { LogoSm } from '../ui/LogoSm';
import {
  DEFAULT_INPUTS,
  DEFAULT_MODELS,
  DEFAULT_IRS,
  SOURCE_MODE_OPTIONS,
} from '../../constants';
import { Tabs } from '../ui/Tabs';
import { usePlayerCore } from '../../hooks/usePlayerCore';
import { PlayerSettings } from './PlayerSettings';
import { Demo } from '../ui/Demo';
import { PlayIcon } from '../ui/PlayIcon';
import { DemoPlaybar } from '../ui/DemoPlaybar';
import { LivePlaybar } from '../ui/LivePlaybar';

const PlayerFC: React.FC<T3kPlayerProps> = ({
  models = DEFAULT_MODELS,
  irs = DEFAULT_IRS,
  inputs = DEFAULT_INPUTS,
  previewMode,
  onPlayDemo,
  onPlayLive,
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
    onPlayDemo,
    onPlayLive,
    onModelChange,
    onInputChange,
    onIrChange,
  });

  return (
    <div className='bg-zinc-900 border border-zinc-700 text-white p-4 lg:p-8 pt-0 lg:pt-2 rounded-xl w-full flex flex-col gap-6'>
      <div className='flex flex-col'>
        <div className='flex items-center min-h-[80px]'>
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
            <LivePlaybar
              togglePlay={core.togglePlay}
              isThisPlayerActive={core.isThisPlayerActive}
              sourceMode={core.sourceMode}
              isLiveConfigured={core.isLiveConfigured}
              canvasWrapperRef={core.canvasWrapperRef}
              visualizerRef={core.visualizerRef}
              infoSlot={infoSlot}
              onOpenSettings={core.openSettingsDialog}
            />
          )}
        </div>
        <div className='flex flex-col gap-6'>
          {/* Toggle between demo and live */}
          <Tabs
            tabs={SOURCE_MODE_OPTIONS.map(o => (
              <div className='flex gap-2 items-center'>
                {o.value === 'demo' ? (
                  <Demo size={20} />
                ) : (
                  <PlayIcon size={20} />
                )}
                <span>{o.label === 'Live' ? 'Play' : o.label}</span>
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
            inputModeType={core.inputModeType}
            audioInputError={core.audioInputDevices.error}
            isThisPlayerActive={core.isThisPlayerActive}
          />
        </div>
      </div>

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
