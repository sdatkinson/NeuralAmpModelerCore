/**
 * Test component for verifying meter and gain control components.
 * Uses the existing audio graph (inputMeterNode, outputMeterNode) to drive meters.
 *
 * Add this temporarily to Player.tsx to test:
 * ```tsx
 * import { MeterTest } from './MeterTest';
 * // Inside Player render:
 * <MeterTest />
 * ```
 */
import React, { useRef, useState, useEffect } from 'react';
import { useT3kPlayerContext } from '../context/T3kPlayerContext';
import { useMeterAnimation } from '../hooks/useMeterAnimation';
import { LevelMeter } from './ui/LevelMeter';
import { ClipIndicator } from './ui/ClipIndicator';
import { GainControl } from './ui/GainControl';
import { dbToLinear } from '../utils/metering';

export function MeterTest() {
  const { getAudioNodes, audioState } = useT3kPlayerContext();
  const nodes = getAudioNodes();

  // Refs for meter DOM elements (used by animation hook)
  const inputMeterRef = useRef<HTMLDivElement>(null);
  const outputMeterRef = useRef<HTMLDivElement>(null);
  const inputClipRef = useRef<HTMLButtonElement>(null);
  const outputClipRef = useRef<HTMLButtonElement>(null);

  // Gain control state (for demo - not connected to actual audio yet)
  const [inputGain, setInputGain] = useState(0);

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

  // Apply input gain to the audio graph (demo)
  useEffect(() => {
    if (nodes.inputGainNode && nodes.audioContext) {
      const linearGain = dbToLinear(inputGain);
      nodes.inputGainNode.gain.setTargetAtTime(
        linearGain,
        nodes.audioContext.currentTime,
        0.02 // 20ms smoothing
      );
    }
  }, [inputGain, nodes.inputGainNode, nodes.audioContext]);

  if (audioState.initState !== 'ready') {
    return (
      <div className="p-4 bg-zinc-900 rounded-lg border border-zinc-700">
        <p className="text-zinc-400 text-sm">
          Audio not initialized. Load a model to test meters.
        </p>
      </div>
    );
  }

  return (
    <div className="p-4 bg-zinc-900 rounded-lg border border-zinc-700">
      <h3 className="text-sm font-medium text-zinc-300 mb-4">
        Meter Test (play audio to see meters)
      </h3>

      <div className="flex items-end gap-8">
        {/* Input Section */}
        <div className="flex flex-col items-center gap-2">
          <span className="text-xs text-zinc-500">Input</span>
          <div className="flex items-end gap-2">
            <LevelMeter
              ref={inputMeterRef}
              orientation="vertical"
              size={80}
              thickness={12}
              label="Input level"
            />
            <ClipIndicator
              ref={inputClipRef}
              onClick={() => resetClipLatch('input')}
              size={12}
            />
          </div>
        </div>

        {/* Gain Control */}
        <div className="flex flex-col items-center">
          <GainControl
            value={inputGain}
            onChange={setInputGain}
            min={-20}
            max={20}
            step={0.5}
            label="Input Gain"
            size={56}
          />
        </div>

        {/* Output Section */}
        <div className="flex flex-col items-center gap-2">
          <span className="text-xs text-zinc-500">Output</span>
          <div className="flex items-end gap-2">
            <LevelMeter
              ref={outputMeterRef}
              orientation="vertical"
              size={80}
              thickness={12}
              label="Output level"
            />
            <ClipIndicator
              ref={outputClipRef}
              onClick={() => resetClipLatch('output')}
              size={12}
            />
          </div>
        </div>
      </div>

      {/* Instructions */}
      <div className="mt-4 text-xs text-zinc-500 space-y-1">
        <p>• Play audio to see meters animate</p>
        <p>• Drag the knob or use scroll wheel to adjust gain</p>
        <p>• Clip indicators turn red when signal peaks - click to reset</p>
        <p>• Increase gain to test clipping detection</p>
      </div>
    </div>
  );
}
