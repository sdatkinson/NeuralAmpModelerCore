import React from 'react';
import { Loader2 } from 'lucide-react';
import {
  AudioInputDevice,
  AudioOutputDevice,
  ChannelSelection,
} from '../../../types';
import { Button } from '../../ui/Button';
import { Alert } from '../../ui/Alert';
import { Select } from '../../ui/Select';
import { InputMeter } from '../../ui/InputMeter';
import { Radio } from '../../ui/Radio';
import { ToggleSimple } from '../../ui/ToggleSimple';
import { isSafari, needsMediaStreamWorkaround } from '../../../utils/browser';
import { InterfaceIcon } from '../../ui/Interface';

interface DeviceContentProps {
  devices: AudioInputDevice[];
  selectedDeviceId: string;
  selectedChannel: ChannelSelection;
  channelCount: number;
  isMonitoring: boolean;
  onDeviceChange: (deviceId: string) => void | Promise<void>;
  onChannelChange: (channel: ChannelSelection) => void;
  onMonitoringChange: (enabled: boolean) => void;
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

export const DeviceContent: React.FC<DeviceContentProps> = ({
  devices,
  selectedDeviceId,
  selectedChannel,
  channelCount,
  isMonitoring,
  onDeviceChange,
  onChannelChange,
  onMonitoringChange,
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
  const hasDevices = devices.length > 0;
  const isStereo = channelCount >= 2;

  // Show error state if failed to load devices
  if (error && !hasDevices) {
    return (
      <div className='flex flex-col gap-4'>
        <Alert variant='error'>
          {error} Failed to load audio input devices. Please check your
          connections and try again.
        </Alert>
        <Button variant='secondary' onClick={onRefresh}>
          Refresh Devices
        </Button>
      </div>
    );
  }

  // Show no devices state
  if (!hasDevices) {
    return (
      <div className='flex flex-col gap-4'>
        <Alert variant='error'>
          No audio input devices found. Please connect an audio interface and
          try again.
        </Alert>
        <Button variant='secondary' onClick={onRefresh}>
          Refresh Devices
        </Button>
      </div>
    );
  }

  return (
    <div className='flex flex-col gap-5'>
      {/* Input Device Selection */}
      <div className='flex flex-col gap-4 w-full border border-zinc-700 rounded-lg p-4'>
        <div className='flex gap-4 items-center'>
          <InterfaceIcon size={20} />
          <h3 className='text-base font-mono text-white font-semibold'>
            AUDIO INTERFACE
          </h3>
        </div>
        {isConnecting ? (
          <div className='flex flex-col gap-2'>
            <div className='flex flex-col'>
              <span className='text-base text-white font-semibold'>
                Select Interface
              </span>
              <span className='text-sm text-zinc-400'>
                Plug your instrument into Input 1 or 2 of your interface.
              </span>
            </div>
            <div className='flex items-center justify-between w-full overflow-hidden px-4 py-3 text-md border border-zinc-700 rounded-md bg-transparent opacity-50 cursor-wait'>
              <span className='text-ellipsis text-nowrap overflow-hidden min-w-0'>
                Initializing audio...
              </span>
              <Loader2
                size={20}
                className='text-zinc-400 flex-shrink-0 animate-spin'
              />
            </div>
          </div>
        ) : (
          <Select
            options={devices.map(d => ({
              label: d.label,
              value: d.deviceId,
            }))}
            value={selectedDeviceId}
            label='Plug your instrument into Input 1 or 2 of your interface.'
            heading='Select Interface'
            onChange={async value => {
              await onDeviceChange(String(value));
              onMonitoringChange(true); // Enable monitoring by default when selecting a device
            }}
          />
        )}
        {/* Safari stereo limitation warning */}
        {!isStereo && isSafari && (
          <Alert variant='warning'>
            Safari supports Input 1 only. Chrome and Firefox support Input 1 or
            2.
          </Alert>
        )}
        {/* Warning if a device was disconnected but others are available */}
        {error && hasDevices && (
          <Alert variant='warning'>
            A device was disconnected. Please select another device.
          </Alert>
        )}
        {/* Input Channel - only for stereo devices */}
        {isStereo && (
          <div className='flex flex-col gap-2'>
            <div className='flex flex-col gap-1'>
              <span className='text-base text-white font-semibold'>
                Input Channel
              </span>
              <span className='text-sm text-zinc-400'>
                Play your instrument and select the channel with signal.
              </span>
            </div>
            <div className='flex flex-col gap-2 mt-1'>
              <label className='flex items-center gap-3 cursor-pointer'>
                <Radio
                  name='input-channel'
                  value='first'
                  checked={selectedChannel === 'first'}
                  onChange={() => onChannelChange('first')}
                />
                <span className='text-sm text-zinc-300 w-4'>1</span>
                <InputMeter
                  analyser={channel0Meter}
                  label='Channel 1 level'
                  inactive={selectedChannel !== 'first'}
                  className='flex-1 min-w-0'
                />
              </label>
              <label className='flex items-center gap-3 cursor-pointer'>
                <Radio
                  name='input-channel'
                  value='second'
                  checked={selectedChannel === 'second'}
                  onChange={() => onChannelChange('second')}
                />
                <span className='text-sm text-zinc-300 w-4'>2</span>
                <InputMeter
                  analyser={channel1Meter}
                  label='Channel 2 level'
                  inactive={selectedChannel !== 'second'}
                  className='flex-1 min-w-0'
                />
              </label>
            </div>
          </div>
        )}
      </div>

      {/* Connection error - shown when startLiveInput fails */}
      {connectionError && (
        <Alert variant='error'>
          {connectionError ||
            'Failed to connect to audio input device. Please check your connections and try again.'}
        </Alert>
      )}

      {/* Output Device Selection - always shown */}
      {isConnecting ? (
        <div className='flex flex-col gap-2'>
          <div className='flex flex-col'>
            <span className='text-base text-white font-semibold'>
              Output Device
            </span>
            <span className='text-sm text-zinc-400'>
              Select where you want to hear the sound.
            </span>
          </div>
          <div className='flex items-center justify-between w-full overflow-hidden px-4 py-3 text-md border border-zinc-700 rounded-md bg-transparent opacity-50 cursor-wait'>
            <span className='text-ellipsis text-nowrap overflow-hidden min-w-0'>
              Initializing audio...
            </span>
            <Loader2
              size={20}
              className='text-zinc-400 flex-shrink-0 animate-spin'
            />
          </div>
        </div>
      ) : (
        outputDevices.length > 0 && (
          <div>
            <Select
              options={[
                ...(needsMediaStreamWorkaround
                  ? [{ label: 'System Default', value: '' }]
                  : []),
                ...outputDevices.map(d => ({
                  label: d.label,
                  value: d.deviceId,
                })),
              ]}
              value={selectedOutputDeviceId ?? ''}
              heading='Output Device'
              label='Select where you want to hear the sound.'
              onChange={value =>
                onOutputDeviceChange(value === '' ? null : String(value))
              }
            />
          </div>
        )
      )}

      {/* Input Monitoring - when a device is selected */}
      <div className='flex flex-col'>
        <h3 className='text-base text-white font-semibold'>Input Monitoring</h3>
        <p className='text-sm text-zinc-400 pb-2'>
          Hear your instrument while playing.
        </p>
        <div className='flex items-center justify-between w-full px-4 py-3 border border-zinc-700 rounded-lg bg-transparent'>
          <span className='text-base text-white'>Hear Yourself</span>
          <ToggleSimple
            label=''
            isChecked={isMonitoring}
            onChange={onMonitoringChange}
            ariaLabel='Hear yourself'
            disabled={isConnecting || !selectedDeviceId}
          />
        </div>
      </div>
    </div>
  );
};

export default DeviceContent;
