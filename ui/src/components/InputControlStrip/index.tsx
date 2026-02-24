import React, { useRef, useEffect } from 'react';
import { useT3kPlayerContext } from '../../context/T3kPlayerContext';
import { useMeterAnimation } from '../../hooks/useMeterAnimation';
import { LevelMeter } from '../ui/LevelMeter';
import { ClipIndicator } from '../ui/ClipIndicator';
import { GainControl } from '../ui/GainControl';
import { SegmentedControl, SegmentOption } from '../ui/SegmentedControl';
import { MonitorButton } from '../ui/MonitorButton';
import { ChannelSelection } from '../../types';

const channelOptions: SegmentOption<ChannelSelection>[] = [
  { value: 'first', label: '1' },
  { value: 'second', label: '2' },
];

interface InputControlStripProps {
  isActive?: boolean;
  isMonitoring: boolean;
  onToggleMonitoring: () => void;
  disabled?: boolean;
  compact?: boolean;
}

export function InputControlStrip({
  isActive = true,
  isMonitoring,
  onToggleMonitoring,
  disabled = false,
  compact = false,
}: InputControlStripProps) {
  const { getAudioNodes, audioState, selectLiveInputChannel, setLiveInputGain } = useT3kPlayerContext();
  const nodes = getAudioNodes();
  const isReady = audioState.initState === 'ready';

  const liveInputConfig = audioState.liveInputConfig;
  const currentChannel = liveInputConfig?.selectedChannel ?? 'first';
  const channelCount = liveInputConfig?.channelCount ?? 1;
  const inputGain = liveInputConfig?.channelGains?.[currentChannel] ?? 0;

  const knobSize = compact ? 36 : 48;
  const meterSize = compact ? 64 : 80;

  const inputMeterRef = useRef<HTMLDivElement>(null);
  const outputMeterRef = useRef<HTMLDivElement>(null);
  const inputClipRef = useRef<HTMLButtonElement>(null);
  const outputClipRef = useRef<HTMLButtonElement>(null);

  const { resetClipLatch } = useMeterAnimation(
    isReady && nodes.inputMeterNode
      ? {
          analyser: nodes.inputMeterNode,
          meterRef: inputMeterRef,
          clipRef: isActive ? inputClipRef : undefined,
        }
      : null,
    isReady && nodes.outputMeterNode
      ? {
          analyser: nodes.outputMeterNode,
          meterRef: outputMeterRef,
          clipRef: isActive ? outputClipRef : undefined,
        }
      : null,
    isReady
  );

  const currentDeviceId = liveInputConfig?.deviceId;
  useEffect(() => {
    resetClipLatch('all');
    inputClipRef.current?.classList.remove('clipped');
    outputClipRef.current?.classList.remove('clipped');
  }, [currentDeviceId, currentChannel, isActive, resetClipLatch]);

  const isConnecting = audioState.inputMode.type === 'connecting';
  const isFullyDisabled = disabled || !isReady;
  const isDisabledOrConnecting = isFullyDisabled || isConnecting;

  return (
    <div
      className={`flex items-stretch w-full transition-opacity ${
        isDisabledOrConnecting ? 'opacity-50 pointer-events-none' : ''
      }`}
    >
      {/* Section 1: Monitor */}
      <div className='flex flex-col items-center gap-1 px-4 sm:px-6 flex-shrink-0'>
        <span className='text-sm invisible' aria-hidden='true'>&nbsp;</span>
        <div className='flex items-center' style={{ height: knobSize }}>
          <MonitorButton
            isMonitoring={isMonitoring}
            onClick={onToggleMonitoring}
            analyserNode={isMonitoring && isReady ? nodes.inputMeterNode : null}
            disabled={isFullyDisabled}
            size={compact ? 28 : undefined}
          />
        </div>
      </div>

      {/* Divider */}
      <div className='w-px self-stretch bg-zinc-700/50 my-2' />

      {/* Section 2: Controls */}
      <div className='flex-1 flex items-start justify-center gap-4 sm:gap-6 px-4'>
        {/* Channel selector */}
        <div className='flex flex-col items-center gap-1'>
          <span className='text-sm text-zinc-400'>Channel</span>
          <div className='flex items-center' style={{ height: knobSize }}>
            <SegmentedControl
              options={channelOptions}
              value={currentChannel}
              onChange={selectLiveInputChannel}
              size='sm'
              disabled={isDisabledOrConnecting || channelCount < 2}
            />
          </div>
        </div>

        {/* Gain */}
        <GainControl
          value={inputGain}
          onChange={setLiveInputGain}
          min={-20}
          max={20}
          step={0.5}
          label='Input'
          size={knobSize}

          disabled={isDisabledOrConnecting}
        />
      </div>

      {/* Divider */}
      <div className='w-px self-stretch bg-zinc-700/50 my-2' />

      {/* Section 3: Meters */}
      <div className='flex items-start gap-4 px-4 sm:px-6 flex-shrink-0'>
        <div className='flex flex-col items-start gap-1'>
          <span className='text-sm text-zinc-400'>In</span>
          <div className='flex items-center gap-1'>
            <LevelMeter
              ref={inputMeterRef}
              orientation='vertical'
              size={meterSize}
              thickness={10}
              label='Input level'
              inactive={!isActive || !isReady}
            />
            <ClipIndicator
              ref={inputClipRef}
              onClick={() => resetClipLatch('input')}
              size={10}
            />
          </div>
        </div>

        <div className='flex flex-col items-start gap-1'>
          <span className='text-sm text-zinc-400'>Out</span>
          <div className='flex items-center gap-1'>
            <LevelMeter
              ref={outputMeterRef}
              orientation='vertical'
              size={meterSize}
              thickness={10}
              label='Output level'
              inactive={!isActive || !isReady}
            />
            <ClipIndicator
              ref={outputClipRef}
              onClick={() => resetClipLatch('output')}
              size={10}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
