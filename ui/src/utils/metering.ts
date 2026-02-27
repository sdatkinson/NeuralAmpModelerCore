/**
 * Audio metering utilities for real-time level calculation
 */

/** Threshold for clip detection (slightly below 1.0 to catch near-clips) */
export const CLIP_THRESHOLD = 0.99;

/** Meter level readings */
export interface MeterLevels {
  /** Peak level (0.0 to 1.0+, can exceed 1.0 if clipping) */
  peak: number;
  /** RMS level (0.0 to 1.0) */
  rms: number;
}

/**
 * Calculate peak and RMS levels from an AnalyserNode
 *
 * @param analyser - The AnalyserNode to read from
 * @param buffer - A reusable Float32Array buffer (should be analyser.fftSize length)
 * @returns Peak and RMS levels
 */
export function calculateLevels(
  analyser: AnalyserNode,
  buffer: Float32Array
): MeterLevels {
  analyser.getFloatTimeDomainData(buffer as Float32Array<ArrayBuffer>);

  let peak = 0;
  let sumSquares = 0;

  for (let i = 0; i < buffer.length; i++) {
    const absSample = Math.abs(buffer[i]);
    if (absSample > peak) peak = absSample;
    sumSquares += buffer[i] * buffer[i];
  }

  return {
    peak,
    rms: Math.sqrt(sumSquares / buffer.length),
  };
}

/**
 * Convert linear amplitude to decibels
 *
 * @param linear - Linear amplitude (0.0 to 1.0+)
 * @returns Decibel value (-Infinity for silence)
 */
export function linearToDb(linear: number): number {
  if (linear <= 0) return -Infinity;
  return 20 * Math.log10(linear);
}

/**
 * Convert decibels to linear amplitude
 *
 * @param db - Decibel value
 * @returns Linear amplitude
 */
export function dbToLinear(db: number): number {
  return Math.pow(10, db / 20);
}

/**
 * Check if the given levels indicate clipping
 *
 * @param levels - Meter levels to check
 * @returns True if peak exceeds clip threshold
 */
export function isClipping(levels: MeterLevels): boolean {
  return levels.peak >= CLIP_THRESHOLD;
}

/**
 * Clamp a value between min and max
 */
export function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

/**
 * Map a linear level (0-1) to a display value with pseudo-logarithmic scaling
 * This makes the meter more visually responsive across the dynamic range
 *
 * @param level - Linear level (0.0 to 1.0)
 * @returns Scaled display value (0.0 to 1.0)
 */
export function scaleForDisplay(level: number): number {
  // Simple power curve - makes quiet signals more visible
  // Adjust exponent for different response curves (0.5 = square root, more responsive to quiet)
  return Math.pow(clamp(level, 0, 1), 0.5);
}

/** dB floor for log-scaled meters (-40 dB ≈ 0.01 linear) */
const LOG_METER_DB_MIN = -40;

/**
 * Map a linear level (0-1) to a pseudo-log display value (0-1) for block meters.
 * Uses dB scaling so blocks fill left-to-right in a log-like manner:
 * quiet levels fill fewer blocks, loud levels fill more, clipping = full bar.
 *
 * @param level - Linear level (0.0 to 1.0+)
 * @returns Display value 0.0 to 1.0
 */
export function levelToLogDisplay(level: number): number {
  if (level <= 0) return 0;
  const db = linearToDb(level);
  const normalized = (db - LOG_METER_DB_MIN) / (0 - LOG_METER_DB_MIN);
  return clamp(normalized, 0, 1);
}
