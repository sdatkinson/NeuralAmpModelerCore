import React from 'react';
import { PREVIEW_MODE, SourceMode } from '../../types';
import { Select } from '../ui/Select';
import { ToggleSimple } from '../ui/ToggleSimple';
import { Loader2, Plug } from 'lucide-react';

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
  onLiveDeviceChange: (deviceId: string) => Promise<void>;
  inputModeType: string;
  audioInputError: string | null;
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
  onLiveDeviceChange,
  inputModeType,
  audioInputError,
}) => {
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

  return (
    <div className='flex flex-col gap-2'>
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

      {/* Input / Live device row + secondary select */}
      <div className='flex flex-col sm:flex-row items-start gap-2 sm:gap-6'>
        <div className='w-full sm:flex-1 min-w-0'>
          {sourceMode === 'preview' && (
            <Select
              options={audioOptions}
              label='Input'
              onChange={onInputChange}
              value={selectedInputUrl}
            />
          )}

          {sourceMode === 'live' && !isLiveConfigured && (
            <div className='flex flex-col gap-1 w-full'>
              <div className='flex justify-between items-end'>
                <span className='text-sm text-zinc-400'>Live Input</span>
              </div>
              <div className='relative'>
                <button
                  onClick={onOpenSettings}
                  className='flex items-center gap-2 w-full px-4 py-3 text-md border border-dashed border-zinc-700 rounded-md bg-transparent hover:bg-zinc-800 hover:border-zinc-600 transition-colors focus:outline-none'
                >
                  <Plug size={16} className='text-zinc-400 flex-shrink-0' />
                  <span className='text-zinc-400'>Enable Live Input</span>
                </button>
                {audioInputError && (
                  <span className='absolute top-full mt-1 text-xs text-red-400'>
                    {audioInputError}
                  </span>
                )}
              </div>
            </div>
          )}

          {sourceMode === 'live' && isLiveConfigured && (
            <div className='w-full'>
              {inputModeType === 'connecting' ? (
                <div className='flex flex-col gap-1 w-full'>
                  <span className='text-sm text-zinc-400'>Live Input</span>
                  <div className='flex items-center justify-between w-full px-4 py-3 text-md border border-zinc-700 rounded-md bg-transparent opacity-50 cursor-wait'>
                    <span className='text-zinc-400'>Switching device...</span>
                    <Loader2 size={24} className='text-zinc-400 animate-spin' />
                  </div>
                </div>
              ) : (
                <Select
                  options={liveDeviceOptions}
                  label='Live Input'
                  onChange={(value) => onLiveDeviceChange(String(value))}
                  value={currentDeviceId ?? ''}
                />
              )}
            </div>
          )}
        </div>

        <div className={`w-full sm:flex-1 min-w-0 ${bypassedStyles}`}>
          {previewMode === PREVIEW_MODE.MODEL
            ? renderIrSelect()
            : renderModelSelect()}
        </div>
      </div>
    </div>
  );
};
