import React, { useState, useEffect, useRef } from 'react';
import { ChevronDown, AlertCircle, AlertTriangle, Loader2 } from 'lucide-react';
import { AudioInputDevice, ChannelSelection } from '../../../types';
import { Button } from '../../ui/Button';
import { SegmentedControl, SegmentOption } from '../../ui/SegmentedControl';
import { LevelMeter } from '../../ui/LevelMeter';
import { ClipIndicator } from '../../ui/ClipIndicator';
import { GainControl } from '../../ui/GainControl';
import { useMeterAnimation } from '../../../hooks/useMeterAnimation';

interface LiveDeviceChannelContentProps {
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
  onRefresh: () => void;
  channel0Meter: AnalyserNode | null;
  channel1Meter: AnalyserNode | null;
}

const channelOptions: SegmentOption<ChannelSelection>[] = [
  { value: 'first', label: 'Left / 1' },
  { value: 'second', label: 'Right / 2' },
];

export const LiveDeviceChannelContent: React.FC<LiveDeviceChannelContentProps> = ({
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
  onRefresh,
  channel0Meter,
  channel1Meter,
}) => {
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);

  // Refs for meter animation
  const channel0MeterRef = useRef<HTMLDivElement>(null);
  const channel1MeterRef = useRef<HTMLDivElement>(null);
  const channel0ClipRef = useRef<HTMLButtonElement>(null);
  const channel1ClipRef = useRef<HTMLButtonElement>(null);

  const selectedDevice = devices.find(d => d.deviceId === selectedDeviceId);
  const hasDevices = devices.length > 0;
  const isStereo = channelCount >= 2;

  // Check if selected device is still available
  const selectedDeviceDisconnected = selectedDeviceId && !selectedDevice && hasDevices;

  // Auto-select first device if selected device was disconnected
  useEffect(() => {
    if (selectedDeviceDisconnected && devices.length > 0) {
      onDeviceChange(devices[0].deviceId);
    }
  }, [selectedDeviceDisconnected, devices, onDeviceChange]);

  // Set up meter animation for preview meters
  const { resetClipLatch } = useMeterAnimation(
    channel0Meter ? { analyser: channel0Meter, meterRef: channel0MeterRef, clipRef: channel0ClipRef } : null,
    channel1Meter && isStereo ? { analyser: channel1Meter, meterRef: channel1MeterRef, clipRef: channel1ClipRef } : null,
    true
  );

  // Reset clip indicators when device or channel changes
  useEffect(() => {
    resetClipLatch('all');
  }, [selectedDeviceId, selectedChannel, resetClipLatch]);

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

  // Error state (failed to load devices)
  if (error && !hasDevices) {
    return (
      <div className='flex flex-col gap-4'>
        <div className='flex items-start gap-3 p-3 bg-red-950/50 border border-red-900/50 rounded-md'>
          <AlertCircle size={18} className='text-red-400 flex-shrink-0 mt-0.5' />
          <div className='flex flex-col gap-1'>
            <p className='text-sm text-red-300'>{error}</p>
            <p className='text-xs text-red-400'>
              Please check your connections and try again.
            </p>
          </div>
        </div>
        <Button variant='secondary' onClick={onRefresh}>
          Refresh Devices
        </Button>
      </div>
    );
  }

  // No devices found
  if (!hasDevices) {
    return (
      <div className='flex flex-col gap-4'>
        <div className='flex items-start gap-3 p-3 bg-yellow-950/50 border border-yellow-900/50 rounded-md'>
          <AlertCircle size={18} className='text-yellow-400 flex-shrink-0 mt-0.5' />
          <div className='flex flex-col gap-1'>
            <p className='text-sm text-yellow-300'>No audio input devices found.</p>
            <p className='text-xs text-yellow-400'>
              Please connect an audio interface or microphone and try again.
            </p>
          </div>
        </div>
        <Button variant='secondary' onClick={onRefresh}>
          Refresh Devices
        </Button>
      </div>
    );
  }

  const selectedDeviceLabel = selectedDevice?.label ?? 'Select device';

  return (
    <div className='flex flex-col gap-5'>
      {/* Warning if a device was disconnected but others are available */}
      {error && hasDevices && (
        <div className='flex items-start gap-3 p-3 bg-yellow-950/50 border border-yellow-900/50 rounded-md'>
          <AlertTriangle size={18} className='text-yellow-400 flex-shrink-0 mt-0.5' />
          <p className='text-sm text-yellow-300'>
            A device was disconnected. Please select another device.
          </p>
        </div>
      )}

      {/* Device Selection */}
      <div className='flex flex-col gap-1'>
        <span className='text-sm text-zinc-400'>Input Device</span>
        <div className='relative'>
          <button
            onClick={() => !isConnecting && setIsDropdownOpen(!isDropdownOpen)}
            disabled={isConnecting}
            className={`flex items-center justify-between w-full overflow-hidden px-4 py-3 text-md border border-zinc-700 rounded-md bg-transparent focus:outline-none transition-colors ${
              isConnecting ? 'opacity-50 cursor-wait' : 'hover:bg-zinc-800'
            }`}
          >
            <span className='text-ellipsis text-nowrap overflow-hidden min-w-0'>
              {isConnecting ? 'Initializing audio...' : selectedDeviceLabel}
            </span>
            {isConnecting ? (
              <Loader2 size={20} className='text-zinc-400 flex-shrink-0 animate-spin' />
            ) : (
              <ChevronDown
                size={20}
                className={`text-zinc-400 flex-shrink-0 transition-transform ${
                  isDropdownOpen ? 'rotate-180' : ''
                }`}
              />
            )}
          </button>

          {/* Dropdown */}
          {isDropdownOpen && (
            <div className='absolute z-10 w-full mt-1 bg-zinc-900 rounded-md shadow-lg border border-zinc-700'>
              <ul className='py-1 overflow-auto text-base rounded-md max-h-48'>
                {devices.map(device => (
                  <li
                    key={device.deviceId}
                    className={`cursor-pointer select-none py-2 px-3 hover:bg-zinc-800 ${
                      selectedDeviceId === device.deviceId ? 'bg-zinc-800' : ''
                    }`}
                    onClick={() => {
                      onDeviceChange(device.deviceId);
                      setIsDropdownOpen(false);
                    }}
                  >
                    {device.label}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </div>

      {/* Channel Selection - only show for stereo devices */}
      {isStereo && (
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

      {/* Gain and Meters Section */}
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

      {/* Monitor Toggle */}
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

      {/* Headphones warning */}
      <div className='flex items-start gap-2 p-2 bg-yellow-950/30 border border-yellow-900/30 rounded text-xs text-yellow-400'>
        <AlertTriangle size={14} className='flex-shrink-0 mt-0.5' />
        <span>Use headphones to avoid feedback</span>
      </div>

      {/* Helper text for mono devices */}
      {!isStereo && (
        <p className='text-xs text-zinc-500'>
          Your device provides a single mono input.
        </p>
      )}
    </div>
  );
};

export default LiveDeviceChannelContent;
