import { useEffect, useRef, useCallback } from 'react';
import {
  calculateLevels,
  isClipping,
  scaleForDisplay,
  MeterLevels,
} from '../utils/metering';

/**
 * Async iterator for requestAnimationFrame loop
 * Allows clean cancellation via break or return
 */
const rafIter = () => {
  let id: number;
  let cancelled = false;

  return {
    async next() {
      if (cancelled) {
        return { value: undefined, done: true };
      }
      const promise = new Promise<number>(resolve => {
        id = requestAnimationFrame(resolve);
      });
      await promise;
      return { value: undefined, done: false };
    },
    async return() {
      cancelled = true;
      cancelAnimationFrame(id);
      return { value: undefined, done: true };
    },
    [Symbol.asyncIterator]() {
      return this;
    },
  };
};

/** Configuration for a single meter */
export interface MeterConfig {
  /** The AnalyserNode to read from */
  analyser: AnalyserNode | null;
  /** DOM element ref for the meter fill (will set --level CSS variable) */
  meterRef: React.RefObject<HTMLElement>;
  /** Optional: DOM element ref for clip indicator (will add/remove 'clipped' class) */
  clipRef?: React.RefObject<HTMLElement>;
}

/** State tracking for clip latching */
export interface ClipState {
  inputClipLatched: boolean;
  outputClipLatched: boolean;
}

/**
 * Hook for animating audio level meters at 60fps
 *
 * Uses direct DOM manipulation via CSS custom properties for performance.
 * Does NOT trigger React re-renders.
 *
 * @param inputConfig - Configuration for input meter
 * @param outputConfig - Configuration for output meter
 * @param enabled - Whether animation is running (default: true)
 * @returns Object with resetClipLatch function and current levels ref
 */
export function useMeterAnimation(
  inputConfig: MeterConfig | null,
  outputConfig: MeterConfig | null,
  enabled: boolean = true
) {
  // Reusable buffer for reading analyser data (must match fftSize of AnalyserNodes)
  const bufferRef = useRef<Float32Array>(new Float32Array(2048));

  // Track current levels (for external access if needed)
  const levelsRef = useRef<{ input: MeterLevels; output: MeterLevels }>({
    input: { peak: 0, rms: 0 },
    output: { peak: 0, rms: 0 },
  });

  // Track clip latch state
  const clipStateRef = useRef<ClipState>({
    inputClipLatched: false,
    outputClipLatched: false,
  });

  // Animation loop cleanup ref
  const cleanupRef = useRef<(() => void) | null>(null);

  // Reset clip latch indicators
  const resetClipLatch = useCallback(
    (which: 'input' | 'output' | 'all' = 'all') => {
      if (which === 'input' || which === 'all') {
        clipStateRef.current.inputClipLatched = false;
        inputConfig?.clipRef?.current?.classList.remove('clipped');
      }
      if (which === 'output' || which === 'all') {
        clipStateRef.current.outputClipLatched = false;
        outputConfig?.clipRef?.current?.classList.remove('clipped');
      }
    },
    [inputConfig?.clipRef, outputConfig?.clipRef]
  );

  useEffect(() => {
    // Clean up any existing animation loop
    cleanupRef.current?.();
    cleanupRef.current = null;

    // Don't start if disabled or no analysers
    if (!enabled) return;
    if (!inputConfig?.analyser && !outputConfig?.analyser) return;

    const buffer = bufferRef.current;
    let running = true;

    // Start animation loop
    (async () => {
      for await (const _ of rafIter()) {
        if (!running) break;

        // Update input meter
        if (inputConfig?.analyser && inputConfig.meterRef.current) {
          const levels = calculateLevels(inputConfig.analyser, buffer);
          levelsRef.current.input = levels;

          // Update meter display (using scaled value for visual response)
          const displayLevel = scaleForDisplay(levels.rms);
          inputConfig.meterRef.current.style.setProperty(
            '--level',
            String(displayLevel)
          );

          // Check for clipping
          if (isClipping(levels)) {
            clipStateRef.current.inputClipLatched = true;
            inputConfig.clipRef?.current?.classList.add('clipped');
          }
        }

        // Update output meter
        if (outputConfig?.analyser && outputConfig.meterRef.current) {
          const levels = calculateLevels(outputConfig.analyser, buffer);
          levelsRef.current.output = levels;

          // Update meter display
          const displayLevel = scaleForDisplay(levels.rms);
          outputConfig.meterRef.current.style.setProperty(
            '--level',
            String(displayLevel)
          );

          // Check for clipping
          if (isClipping(levels)) {
            clipStateRef.current.outputClipLatched = true;
            outputConfig.clipRef?.current?.classList.add('clipped');
          }
        }
      }
    })();

    // Cleanup function
    cleanupRef.current = () => {
      running = false;
    };

    return () => {
      running = false;
      cleanupRef.current = null;
    };
  }, [
    enabled,
    inputConfig?.analyser,
    inputConfig?.meterRef,
    inputConfig?.clipRef,
    outputConfig?.analyser,
    outputConfig?.meterRef,
    outputConfig?.clipRef,
  ]);

  return {
    /** Reset clip latch indicator(s) */
    resetClipLatch,
    /** Current meter levels (updated every frame, not reactive) */
    levelsRef,
    /** Current clip state (updated every frame, not reactive) */
    clipStateRef,
  };
}
