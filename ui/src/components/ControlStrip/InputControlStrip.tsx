import React, { useRef, useEffect } from 'react';
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

interface InputControlStripProps {
  /** Whether this player is the active one (controls meter animation and colors) */
  isActive?: boolean;
}

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
export function InputControlStrip({ isActive = true }: InputControlStripProps) {
  const { getAudioNodes, audioState, selectLiveInputChannel, setLiveInputGain } = useT3kPlayerContext();
  const nodes = getAudioNodes();

  // Derive channel and gain info from liveInputConfig (persists even when preview is active)
  const liveInputConfig = audioState.liveInputConfig;
  const currentChannel = liveInputConfig?.selectedChannel ?? 'first';
  const channelCount = liveInputConfig?.channelCount ?? 1;
  const inputGain = liveInputConfig?.channelGains?.[currentChannel] ?? 0;
  const isStereo = channelCount >= 2;

  // Refs for meter DOM elements (used by animation hook)
  const inputMeterRef = useRef<HTMLDivElement>(null);
  const outputMeterRef = useRef<HTMLDivElement>(null);
  const inputClipRef = useRef<HTMLButtonElement>(null);
  const outputClipRef = useRef<HTMLButtonElement>(null);

  // Start meter animation when audio is initialized
  // Animation always runs so users can see signal levels; inactive state only affects colors
  const { resetClipLatch } = useMeterAnimation(
    nodes.inputMeterNode
      ? {
          analyser: nodes.inputMeterNode,
          meterRef: inputMeterRef,
          clipRef: isActive ? inputClipRef : undefined,  // Only latch clips when active
        }
      : null,
    nodes.outputMeterNode
      ? {
          analyser: nodes.outputMeterNode,
          meterRef: outputMeterRef,
          clipRef: isActive ? outputClipRef : undefined,  // Only latch clips when active
        }
      : null,
    audioState.initState === 'ready'
  );

  // Reset clip indicators when device, channel changes, or when becoming inactive (paused)
  const currentDeviceId = liveInputConfig?.deviceId;
  useEffect(() => {
    resetClipLatch('all');
    // Also clear DOM directly — when isActive flips to false, resetClipLatch
    // may have stale (undefined) clipRefs since they're conditionally passed
    inputClipRef.current?.classList.remove('clipped');
    outputClipRef.current?.classList.remove('clipped');
  }, [currentDeviceId, currentChannel, isActive, resetClipLatch]);

  // Don't render if audio not ready
  // Note: Parent component controls visibility based on sourceMode and liveInputConfig
  if (audioState.initState !== 'ready') {
    return null;
  }

  const isConnecting = audioState.inputMode.type === 'connecting';

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
                onChange={selectLiveInputChannel}
                size='sm'
              />
            </div>
          )}

          {/* Input Gain Control */}
          <div className='flex flex-col items-center'>
            <GainControl
              value={inputGain}
              onChange={setLiveInputGain}
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
                  inactive={!isActive}
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
                  inactive={!isActive}
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
