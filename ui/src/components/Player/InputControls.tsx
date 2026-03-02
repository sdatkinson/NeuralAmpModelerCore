import React, { useCallback, useRef, useEffect } from 'react';
import { KnobControl } from '../ui/KnobControl';
import { InputMeter } from '../ui/InputMeter';
import { ClipIndicator } from '../ui/ClipIndicator';
import { useT3kPlayerContext } from '../../context/T3kPlayerContext';
import { useMeterAnimation } from '../../hooks/useMeterAnimation';

const DB_MIN = -12;
const DB_MAX = 12;

/** Map dB (-12..+12) to knob value (0..1) */
function dbToKnob(db: number): number {
  return Math.max(0, Math.min(1, (db - DB_MIN) / (DB_MAX - DB_MIN)));
}

/** Map knob value (0..1) to dB (-12..+12) */
function knobToDb(knob: number): number {
  return knob * (DB_MAX - DB_MIN) + DB_MIN;
}

interface InputControlsProps {
  isMonitoring: boolean;
  isConnecting: boolean;
}

export const InputControls: React.FC<InputControlsProps> = ({
  isMonitoring,
  isConnecting,
}) => {
  const { getAudioNodes, audioState, setLiveInputGain } = useT3kPlayerContext();
  const nodes = getAudioNodes();
  const isReady = audioState.initState === 'ready';
  const liveInputConfig = audioState.liveInputConfig;
  const currentChannel = liveInputConfig?.selectedChannel ?? 'first';
  const inputGainDb = liveInputConfig?.channelGains?.[currentChannel] ?? 0;

  const inputMeterRef = useRef<HTMLDivElement>(null);
  const inputClipRef = useRef<HTMLButtonElement>(null);

  const isDisabledOrConnecting = !isReady || !isMonitoring || isConnecting;

  const { resetClipLatch } = useMeterAnimation(
    isReady && nodes.inputMeterNode
      ? {
          analyser: nodes.inputMeterNode,
          meterRef: inputMeterRef,
          clipRef: inputClipRef,
        }
      : null,
    null,
    isReady
  );

  useEffect(() => {
    resetClipLatch('input');
    inputClipRef.current?.classList.remove('clipped');
  }, [liveInputConfig?.selectedChannel, resetClipLatch]);

  const handleKnobChange = useCallback(
    (knobValue: number) => setLiveInputGain(knobToDb(knobValue)),
    [setLiveInputGain]
  );

  return (
    <div className='flex flex-row items-center gap-4 border py-3 px-4 border-zinc-700 rounded-md w-full'>
      <div
        className={`flex flex-col gap-4 items-center ${isDisabledOrConnecting ? 'opacity-50 cursor-not-allowed' : ''}`}
      >
        <span className='text-xs text-zinc-400 text-center tabular-nums'>
          {inputGainDb >= 0 ? '+' : ''}
          {inputGainDb.toFixed(1)} dB
        </span>
        <KnobControl
          label='Input Gain'
          value={dbToKnob(inputGainDb)}
          onChange={handleKnobChange}
          size={36}
          isDisabled={isDisabledOrConnecting}
        />
      </div>
      <div
        ref={inputMeterRef}
        className='flex items-center gap-4 flex-1 min-w-0'
      >
        <InputMeter
          analyser={nodes.inputMeterNode}
          label='Input level'
          inactive={isDisabledOrConnecting}
        />
        <ClipIndicator
          ref={inputClipRef}
          onClick={() => resetClipLatch('input')}
          size={16}
          inactive={isDisabledOrConnecting}
        />
      </div>
    </div>
  );
};
