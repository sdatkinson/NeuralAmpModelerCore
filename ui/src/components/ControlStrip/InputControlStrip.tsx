import React, { useRef, useCallback, useEffect } from 'react';
import { useT3kPlayerContext } from '../../context/T3kPlayerContext';
import { useMeterAnimation } from '../../hooks/useMeterAnimation';
import { LevelMeter } from '../ui/LevelMeter';
import { ClipIndicator } from '../ui/ClipIndicator';
import { GainControl } from '../ui/GainControl';
import { SegmentedControl, SegmentOption } from '../ui/SegmentedControl';
import { ChannelSelection } from '../../types';

// Channel options for the compact selector
const channelOptions: SegmentOption<ChannelSelection>[] = [
  { value: 'first', label: '1' },
  { value: 'second', label: '2' },
];

/**
 * Input control strip for live input mode.
 *
 * Displays:
 * - Channel selector (for stereo devices)
 * - Input gain control (-20dB to +20dB)
 * - Input level meter with clip indicator
 * - Output level meter with clip indicator
 * - Info text with instructions
 *
 * Only renders when audio is initialized and in live input mode.
 */
export function InputControlStrip() {
  const { getAudioNodes, audioState, selectLiveInputChannel, setLiveInputGain } = useT3kPlayerContext();
  const nodes = getAudioNodes();

  // Derive channel and gain info from live input mode
  const liveInputMode = audioState.inputMode.type === 'live' ? audioState.inputMode : null;
  const currentChannel = liveInputMode?.selectedChannel ?? 'first';
  const channelCount = liveInputMode?.channelCount ?? 1;
  const inputGain = liveInputMode?.channelGains?.[currentChannel] ?? 0;
  const isStereo = channelCount >= 2;

  // Handle channel change
  const handleChannelChange = useCallback((channel: ChannelSelection) => {
    selectLiveInputChannel(channel);
  }, [selectLiveInputChannel]);

  // Handle gain change - use context's setLiveInputGain
  const handleGainChange = useCallback((gainDb: number) => {
    setLiveInputGain(gainDb);
  }, [setLiveInputGain]);

  // Refs for meter DOM elements (used by animation hook)
  const inputMeterRef = useRef<HTMLDivElement>(null);
  const outputMeterRef = useRef<HTMLDivElement>(null);
  const inputClipRef = useRef<HTMLButtonElement>(null);
  const outputClipRef = useRef<HTMLButtonElement>(null);

  // Start meter animation when audio is initialized
  const { resetClipLatch } = useMeterAnimation(
    nodes.inputMeterNode
      ? {
          analyser: nodes.inputMeterNode,
          meterRef: inputMeterRef,
          clipRef: inputClipRef,
        }
      : null,
    nodes.outputMeterNode
      ? {
          analyser: nodes.outputMeterNode,
          meterRef: outputMeterRef,
          clipRef: outputClipRef,
        }
      : null,
    audioState.initState === 'ready'
  );

  // Reset clip indicators when device or channel changes
  const currentDeviceId = liveInputMode?.deviceId;
  useEffect(() => {
    resetClipLatch('all');
  }, [currentDeviceId, currentChannel, resetClipLatch]);

  // Don't render if audio not ready or not in live mode
  if (audioState.initState !== 'ready') {
    return null;
  }

  if (audioState.inputMode.type !== 'live') {
    return null;
  }

  const isConnecting = audioState.isLiveConnecting;

  return (
    <div className={`p-4 bg-zinc-800/50 rounded-lg border border-zinc-700 transition-opacity ${
      isConnecting ? 'opacity-50 pointer-events-none' : ''
    }`}>
      <div className='grid grid-cols-1 md:grid-cols-2 gap-4 md:gap-6'>
        {/* Controls */}
        <div className='flex flex-wrap items-start justify-between gap-4'>
          {/* Channel Selector - only for stereo devices */}
          {isStereo && (
            <div className='flex flex-col items-center gap-1'>
              <span className='text-xs text-zinc-400'>Ch</span>
              <SegmentedControl
                options={channelOptions}
                value={currentChannel}
                onChange={handleChannelChange}
                size='sm'
              />
            </div>
          )}

          {/* Input Gain Control */}
          <div className='flex flex-col items-center'>
            <GainControl
              value={inputGain}
              onChange={handleGainChange}
              min={-20}
              max={20}
              step={0.5}
              label='Input Gain'
              size={36}
              disabled={isConnecting}
            />
          </div>

          {/* Input & Output Meters - grouped together */}
          <div className='flex items-start gap-3 md:gap-4'>
            {/* Input Meter Section */}
            <div className='flex flex-col items-start gap-1'>
              <span className='text-xs text-zinc-400'>In</span>
              <div className='flex items-center gap-1'>
                <LevelMeter
                  ref={inputMeterRef}
                  orientation='vertical'
                  size={60}
                  thickness={8}
                  label='Input level'
                />
                <ClipIndicator
                  ref={inputClipRef}
                  onClick={() => resetClipLatch('input')}
                  size={8}
                />
              </div>
            </div>

            {/* Output Meter Section */}
            <div className='flex flex-col items-start gap-1'>
              <span className='text-xs text-zinc-400'>Out</span>
              <div className='flex items-center gap-1'>
                <LevelMeter
                  ref={outputMeterRef}
                  orientation='vertical'
                  size={60}
                  thickness={8}
                  label='Output level'
                />
                <ClipIndicator
                  ref={outputClipRef}
                  onClick={() => resetClipLatch('output')}
                  size={8}
                />
              </div>
            </div>
          </div>
        </div>

        {/* Info Text */}
        <div className='text-xs text-zinc-500 space-y-1'>
          <p>Adjust <span className='text-zinc-400'>Input Gain</span> to optimize signal level.</p>
          <p>Red clip indicators latch when signal peaks. Click to reset.</p>
          <p>Output is normalized by default.</p>
        </div>
      </div>
    </div>
  );
}

export default InputControlStrip;
