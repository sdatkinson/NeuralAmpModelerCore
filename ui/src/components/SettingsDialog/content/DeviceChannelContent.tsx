import React, { useRef, useEffect } from 'react';
import { Loader2 } from 'lucide-react';
import { AudioInputDevice, AudioOutputDevice, ChannelSelection, SourceMode } from '../../../types';
import { Button } from '../../ui/Button';
import { Alert } from '../../ui/Alert';
import { InlineAlert } from '../../ui/InlineAlert';
import { SegmentedControl, SegmentOption } from '../../ui/SegmentedControl';
import { LevelMeter } from '../../ui/LevelMeter';
import { ClipIndicator } from '../../ui/ClipIndicator';
import { GainControl } from '../../ui/GainControl';
import { Select } from '../../ui/Select';
import { useMeterAnimation } from '../../../hooks/useMeterAnimation';
import { isSafari, needsMediaStreamWorkaround } from '../../../utils/browser';

interface DeviceChannelContentProps {
  sourceMode: SourceMode;
  devices: AudioInputDevice[];
  selectedDeviceId: string;
  selectedChannel: ChannelSelection;
  channelCount: number;
  isMonitoring: boolean;
  isWetSignalEnabled: boolean;
  inputGain: number;
  onDeviceChange: (deviceId: string) => void;
  onChannelChange: (channel: ChannelSelection) => void;
  onMonitoringChange: (enabled: boolean) => void;
  onWetSignalToggle: () => void;
  onInputGainChange: (gainDb: number) => void;
  isLoading: boolean;
  isConnecting: boolean;
  error: string | null;
  connectionError: string | null;
  onRefresh: () => void;
  channel0Meter: AnalyserNode | null;
  channel1Meter: AnalyserNode | null;
  // Output device selection
  outputDevices: AudioOutputDevice[];
  selectedOutputDeviceId: string | null;
  onOutputDeviceChange: (deviceId: string | null) => void;
}

const channelOptions: SegmentOption<ChannelSelection>[] = [
  { value: 'first', label: 'Left / 1' },
  { value: 'second', label: 'Right / 2' },
];

export const DeviceChannelContent: React.FC<DeviceChannelContentProps> = ({
  sourceMode,
  devices,
  selectedDeviceId,
  selectedChannel,
  channelCount,
  isMonitoring,
  isWetSignalEnabled,
  inputGain,
  onDeviceChange,
  onChannelChange,
  onMonitoringChange,
  onWetSignalToggle,
  onInputGainChange,
  isLoading,
  isConnecting,
  error,
  connectionError,
  onRefresh,
  channel0Meter,
  channel1Meter,
  outputDevices,
  selectedOutputDeviceId,
  onOutputDeviceChange,
}) => {
  // Refs for meter animation
  const channel0MeterRef = useRef<HTMLDivElement>(null);
  const channel1MeterRef = useRef<HTMLDivElement>(null);
  const channel0ClipRef = useRef<HTMLButtonElement>(null);
  const channel1ClipRef = useRef<HTMLButtonElement>(null);

  const hasDevices = devices.length > 0;
  const isStereo = channelCount >= 2;

  const isLiveMode = sourceMode === 'live';

  // Set up meter animation for preview meters (only in live mode)
  const { resetClipLatch } = useMeterAnimation(
    isLiveMode && channel0Meter ? { analyser: channel0Meter, meterRef: channel0MeterRef, clipRef: channel0ClipRef } : null,
    isLiveMode && channel1Meter && isStereo ? { analyser: channel1Meter, meterRef: channel1MeterRef, clipRef: channel1ClipRef } : null,
    isLiveMode
  );

  // Reset clip indicators when device or channel changes
  useEffect(() => {
    if (isLiveMode) {
      resetClipLatch('all');
    }
  }, [isLiveMode, selectedDeviceId, selectedChannel, resetClipLatch]);

  // Loading skeleton
  if (isLoading) {
    return (
      <div className='flex flex-col gap-4'>
        <div className='flex flex-col gap-1'>
          <div className='h-4 bg-zinc-800 rounded animate-pulse w-24' />
          <div className='h-12 bg-zinc-800 rounded animate-pulse w-full' />
        </div>
        <div className='h-3 bg-zinc-800 rounded animate-pulse w-3/4' />
      </div>
    );
  }

  // In live mode, show error state if failed to load devices
  if (isLiveMode && error && !hasDevices) {
    return (
      <div className='flex flex-col gap-4'>
        <Alert variant='error' description='Please check your connections and try again.'>
          {error}
        </Alert>
        <Button variant='secondary' onClick={onRefresh}>
          Refresh Devices
        </Button>
      </div>
    );
  }

  // In live mode, show no devices state
  if (isLiveMode && !hasDevices) {
    return (
      <div className='flex flex-col gap-4'>
        <Alert variant='warning' description='Please connect an audio interface or microphone and try again.'>
          No audio input devices found.
        </Alert>
        <Button variant='secondary' onClick={onRefresh}>
          Refresh Devices
        </Button>
      </div>
    );
  }

  return (
    <div className='flex flex-col gap-5'>
      {/* Warning if a device was disconnected but others are available (live mode only) */}
      {isLiveMode && error && hasDevices && (
        <Alert variant='warning'>
          A device was disconnected. Please select another device.
        </Alert>
      )}

      {/* Input Device Selection - only in live mode */}
      {isLiveMode && (
        <div>
          {isConnecting ? (
            <div className='flex flex-col gap-1'>
              <span className='text-sm text-zinc-400'>Input Device</span>
              <div className='flex items-center justify-between w-full overflow-hidden px-4 py-3 text-md border border-zinc-700 rounded-md bg-transparent opacity-50 cursor-wait'>
                <span className='text-ellipsis text-nowrap overflow-hidden min-w-0'>Initializing audio...</span>
                <Loader2 size={20} className='text-zinc-400 flex-shrink-0 animate-spin' />
              </div>
            </div>
          ) : (
            <Select
              options={devices.map(d => ({ label: d.label, value: d.deviceId }))}
              value={selectedDeviceId}
              label='Input Device'
              onChange={(value) => onDeviceChange(String(value))}
            />
          )}
          {/* Safari stereo limitation warning */}
          {!isStereo && isSafari && (
            <InlineAlert className='mt-1'>
              Safari does not support stereo audio input. For stereo, use Chrome or Firefox.
            </InlineAlert>
          )}
        </div>
      )}

      {/* Connection error - shown when startLiveInput fails */}
      {isLiveMode && connectionError && (
        <Alert variant='error'>
          {connectionError}
        </Alert>
      )}

      {/* Output Device Selection - always shown */}
      {outputDevices.length > 0 && (
        <div>
          <Select
            options={[
              ...(needsMediaStreamWorkaround ? [{ label: 'System Default', value: '' }] : []),
              ...outputDevices.map(d => ({ label: d.label, value: d.deviceId })),
            ]}
            value={selectedOutputDeviceId ?? ''}
            label='Output Device'
            onChange={(value) => onOutputDeviceChange(value === '' ? null : String(value))}
          />
          {/* Headphones warning - only in live mode */}
          {isLiveMode && (
            <InlineAlert className='mt-1'>
              Use headphones to avoid feedback
            </InlineAlert>
          )}
        </div>
      )}

      {/* Channel Selection - only in live mode for stereo devices */}
      {isLiveMode && isStereo && (
        <div className='flex flex-col gap-1 items-start'>
          <span className='text-sm text-zinc-400'>Input Channel</span>
          <p className='text-xs text-zinc-500 mb-1'>Select which channel of your audio interface to use.</p>
          <SegmentedControl
            options={channelOptions}
            value={selectedChannel}
            onChange={onChannelChange}
          />
        </div>
      )}

      {/* Gain and Meters Section - only in live mode */}
      {isLiveMode && (
        <div className='flex flex-col gap-1'>
          <span className='text-sm text-zinc-400'>Input Level & Gain</span>
          <p className='text-xs text-zinc-500 mb-1'>Controls impact the selected channel.</p>
          <div className='flex flex-wrap items-center gap-4'>
            {/* Meters */}
            <div className='flex flex-col gap-2 flex-1 min-w-[200px]'>
              {isStereo ? (
                <>
                  {/* Left channel meter */}
                  <div className='flex items-center gap-2'>
                    <span className={`text-xs w-4 ${selectedChannel === 'first' ? 'font-bold text-zinc-300' : 'text-zinc-500'}`}>L</span>
                    <LevelMeter
                      ref={channel0MeterRef}
                      orientation='horizontal'
                      size={140}
                      thickness={12}
                      label='Left channel level'
                      inactive={selectedChannel !== 'first'}
                    />
                    <ClipIndicator
                      ref={channel0ClipRef}
                      onClick={() => resetClipLatch('input')}
                      size={12}
                    />
                  </div>
                  {/* Right channel meter */}
                  <div className='flex items-center gap-2'>
                    <span className={`text-xs w-4 ${selectedChannel === 'second' ? 'font-bold text-zinc-300' : 'text-zinc-500'}`}>R</span>
                    <LevelMeter
                      ref={channel1MeterRef}
                      orientation='horizontal'
                      size={140}
                      thickness={12}
                      label='Right channel level'
                      inactive={selectedChannel !== 'second'}
                    />
                    <ClipIndicator
                      ref={channel1ClipRef}
                      onClick={() => resetClipLatch('output')}
                      size={12}
                    />
                  </div>
                </>
              ) : (
                /* Single meter for mono */
                <div className='flex items-center gap-2'>
                  <LevelMeter
                    ref={channel0MeterRef}
                    orientation='horizontal'
                    size={180}
                    thickness={12}
                    label='Input level'
                  />
                  <ClipIndicator
                    ref={channel0ClipRef}
                    onClick={() => resetClipLatch('input')}
                    size={12}
                  />
                </div>
              )}
            </div>

            {/* Input Gain Control */}
            <GainControl
              value={inputGain}
              onChange={onInputGainChange}
              min={-20}
              max={20}
              step={0.5}
              label='Gain'
              size={44}
              disabled={!selectedDeviceId || isConnecting}
            />
          </div>
        </div>
      )}

      {/* Monitor Toggle - only in live mode when a device is selected */}
      {isLiveMode && selectedDeviceId && (
        <div className='flex flex-col gap-2'>
          <label className='flex items-center gap-3 cursor-pointer'>
            <input
              type='checkbox'
              checked={isMonitoring}
              onChange={e => onMonitoringChange(e.target.checked)}
              className='w-4 h-4 rounded border-zinc-600 bg-zinc-800 text-blue-500 focus:ring-blue-500 focus:ring-offset-zinc-900'
            />
            <span className='text-sm text-zinc-300'>Monitor input (hear yourself)</span>
          </label>

          {/* Wet Signal Toggle - only show when monitoring is enabled */}
          {isMonitoring && (
            <label className='flex items-center gap-3 cursor-pointer ml-7'>
              <input
                type='checkbox'
                checked={isWetSignalEnabled}
                onChange={onWetSignalToggle}
                className='w-4 h-4 rounded border-zinc-600 bg-zinc-800 text-blue-500 focus:ring-blue-500 focus:ring-offset-zinc-900'
              />
              <span className='text-sm text-zinc-300'>Enable rig (hear processed signal)</span>
            </label>
          )}
        </div>
      )}

      {/* Helper text for mono devices - only in live mode */}
      {isLiveMode && !isStereo && selectedDeviceId && (
        <p className='text-xs text-zinc-500'>
          Your device provides a single mono input.
        </p>
      )}
    </div>
  );
};

export default DeviceChannelContent;
