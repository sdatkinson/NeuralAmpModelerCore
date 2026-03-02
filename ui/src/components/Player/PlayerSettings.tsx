import React from 'react';
import { PREVIEW_MODE, SourceMode } from '../../types';
import { Select } from '../ui/Select';
import { ToggleSimple } from '../ui/ToggleSimple';
import { Settings } from 'lucide-react';
import { LiveTakeover } from './LiveTakeover';
import { InputControls } from './InputControls';
import { Alert } from '../ui/Alert';

interface PlayerSettingsProps {
  previewMode?: PREVIEW_MODE;
  disabled?: boolean;

  bypassed: boolean;
  bypassedStyles: string;
  onBypassToggle: () => void;

  sourceMode: SourceMode;
  onOpenSettings: () => void | Promise<void>;

  modelOptions: Array<{ label: string; value: string }>;
  irOptions: Array<{ label: string; value: string }>;
  audioOptions: Array<{ label: string; value: string }>;
  selectedModelUrl: string;
  selectedIrUrl: string;
  selectedInputUrl: string;
  onModelChange: (value: string | number) => Promise<void>;
  onIrChange: (value: string | number) => Promise<void>;
  onInputChange: (value: string | number) => Promise<void>;

  isLiveConfigured: boolean;
  currentDeviceId: string | null;
  liveDeviceOptions: Array<{ label: string; value: string }>;
  inputModeType: string;
  audioInputError: string | null;
  isThisPlayerActive: boolean;
}

export const PlayerSettings: React.FC<PlayerSettingsProps> = ({
  previewMode,
  disabled,
  bypassed,
  bypassedStyles,
  onBypassToggle,
  sourceMode,
  onOpenSettings,
  modelOptions,
  irOptions,
  audioOptions,
  selectedModelUrl,
  selectedIrUrl,
  selectedInputUrl,
  onModelChange,
  onIrChange,
  onInputChange,
  isLiveConfigured,
  currentDeviceId,
  liveDeviceOptions,
  inputModeType,
  audioInputError,
  isThisPlayerActive,
}) => {
  const isMonitoring = isThisPlayerActive && sourceMode === 'live';
  const isConnecting = inputModeType === 'connecting';

  const renderModelSelect = () => (
    <Select
      options={modelOptions}
      label='Model'
      onChange={onModelChange}
      value={selectedModelUrl}
      disabled={bypassed}
    />
  );

  const renderIrSelect = () => (
    <Select
      options={irOptions}
      label='IR'
      onChange={onIrChange}
      value={selectedIrUrl}
      disabled={bypassed}
    />
  );

  if (sourceMode === 'live' && !isLiveConfigured && !isConnecting) {
    return <LiveTakeover onConnect={onOpenSettings} />;
  }

  return (
    <div className='flex flex-col gap-4'>
      {/* Primary select + bypass toggle */}
      <div className='flex flex-row items-center gap-4 flex-wrap'>
        <div className='flex-1 min-w-[0px]'>
          <div className={`flex w-full flex-1 min-w-[0px] ${bypassedStyles}`}>
            {previewMode === PREVIEW_MODE.MODEL
              ? renderModelSelect()
              : renderIrSelect()}
          </div>
        </div>

        <div className='flex items-center pt-[24px] flex-shrink-0'>
          <ToggleSimple
            label=''
            onChange={onBypassToggle}
            isChecked={!bypassed}
            ariaLabel='Bypass'
            disabled={disabled}
          />
        </div>
      </div>

      {/* Secondary select */}
      <div className={`w-full sm:flex-1 min-w-0 ${bypassedStyles}`}>
        {previewMode === PREVIEW_MODE.MODEL
          ? renderIrSelect()
          : renderModelSelect()}
      </div>

      {/* Input / Play device row */}
      <div className='w-full sm:flex-1 min-w-0'>
        {sourceMode === 'demo' && (
          <Select
            options={audioOptions}
            label='Input'
            onChange={onInputChange}
            value={selectedInputUrl}
          />
        )}

        {sourceMode === 'live' && isLiveConfigured && (
          <div className='w-full'>
            <div className='flex flex-col gap-1 w-full'>
              <div className='flex items-center justify-between w-full'>
                <span className='text-sm text-zinc-400'>
                  {isConnecting
                    ? 'Connecting...'
                    : (liveDeviceOptions?.find(
                        option => option.value === currentDeviceId
                      )?.label ?? 'No device selected')}
                </span>
                <button
                  type='button'
                  className='text-white'
                  onClick={onOpenSettings}
                >
                  <Settings size={18} />
                </button>
              </div>
              <InputControls
                isMonitoring={isMonitoring}
                isConnecting={isConnecting}
              />
              {audioInputError && (
                <Alert variant='error'>{audioInputError}</Alert>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
