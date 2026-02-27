import { ReactNode } from 'react';

export interface Model {
  name: string;
  url: string;
  default?: boolean;
}

export interface IR {
  name: string;
  url: string;
  mix?: number;
  gain?: number;
  default?: boolean;
}

export interface Input {
  name: string;
  url: string;
  default?: boolean;
}

export enum PREVIEW_MODE {
  MODEL = 'model',
  IR = 'ir',
}

// Source mode for players: 'demo' uses file playback, 'play' uses direct input
export type SourceMode = 'demo' | 'play';

// Channel selection for multi-channel audio interfaces
export type ChannelSelection = 'first' | 'second';

// What audio source is currently active (connected to the audio engine)
export type InputMode =
  | { type: 'demo' }
  | { type: 'connecting' }
  | { type: 'play' };

// Configured play input settings (persists even when demo is active)
// This allows the UI to show the configured device even when file playback is active
export interface PlayInputConfig {
  deviceId: string;
  channelCount: number;
  selectedChannel: ChannelSelection;
  channelGains: Record<ChannelSelection, number>;
}

export interface AudioInputDevice {
  deviceId: string;
  label: string;
}

export interface AudioOutputDevice {
  deviceId: string;
  label: string;
}

// Snapshot of settings for restore on dialog cancel
export interface SettingsSnapshot {
  outputDeviceId: string | null;
  inputMode: InputMode;
  playInputConfig: PlayInputConfig | null;
  isPlaying: boolean;
  isBypassed: boolean;
  activePlayerId: string | null;
}

// State of the global settings dialog (managed by context)
export interface SettingsDialogState {
  isOpen: boolean;
  sourceMode: SourceMode;
  playerId?: string;
  selectedModel?: Model;
  selectedIr?: IR;
  snapshot: SettingsSnapshot | null;
  hadExistingConfig: boolean;
}

// Microphone permission state (permission concerns only)
// - 'idle': not yet requested
// - 'pending': waiting for user response to browser prompt
// - 'granted': permission granted
// - 'denied': permission denied (can retry)
// - 'blocked': permanently blocked by browser (must reset in browser settings)
// - 'error': other error (device not found, in use, etc.)
export type MicrophonePermissionStatus =
  | 'idle'
  | 'pending'
  | 'granted'
  | 'denied'
  | 'blocked'
  | 'error';

export interface MicrophonePermissionState {
  status: MicrophonePermissionStatus;
  error: string | null;
}

// Audio input device state (device concerns only)
export interface AudioInputDeviceState {
  devices: AudioInputDevice[];
  isLoading: boolean;
  error: string | null;
  preferredDeviceId: string | null; // Device selected by user in browser permission dialog
}

// Audio output device state
export interface AudioOutputDeviceState {
  devices: AudioOutputDevice[];
  selectedDeviceId: string | null; // null means system default
}

// All Web Audio API node references managed by the audio engine
export interface AudioNodes {
  audioContext: AudioContext | null;
  audioElement: HTMLAudioElement | null;
  audioWorkletNode: AudioWorkletNode | null;
  inputGainNode: GainNode | null;
  outputGainNode: GainNode | null;
  bypassNode: GainNode | null;
  irNode: ConvolverNode | null;
  irWetGain: GainNode | null;
  irDryGain: GainNode | null;
  irGain: GainNode | null;
  sourceNode: MediaElementAudioSourceNode | null;
  // Play input nodes
  playSourceNode: MediaStreamAudioSourceNode | null;
  playInputGainNode: GainNode | null;
  mediaStream: MediaStream | null;
  // Metering nodes
  inputMeterNode: AnalyserNode | null;
  outputMeterNode: AnalyserNode | null;
  // Channel selection (for multi-channel interfaces)
  channelSplitterNode: ChannelSplitterNode | null;
  channelMergerNode: ChannelMergerNode | null;
  channel0PlayMeter: AnalyserNode | null;
  channel1PlayMeter: AnalyserNode | null;
  // Output device routing workaround (Firefox/Safari don't support AudioContext.setSinkId)
  outputWorkaroundDestination: MediaStreamAudioDestinationNode | null;
  outputWorkaroundElement: HTMLAudioElement | null;
}

// Explicit initialization states for visibility into the init process
export type AudioInitState = 'uninitialized' | 'initializing' | 'ready';

export interface AudioState {
  initState: AudioInitState;
  isPlaying: boolean; // Whether audio is playing (demo) or monitoring (play)
  activePlayerId: string | null; // Which player is currently controlling playback
  isBypassed: boolean;
  modelUrl: string | null;
  irUrl: string | null;
  audioUrl: string | null;
  // What audio source is currently active (connected to audio engine)
  inputMode: InputMode;
  // Configured play input settings (persists even when demo is active)
  // This allows UI to show configured device while another player uses file playback
  playInputConfig: PlayInputConfig | null;
}

export interface IrConfig {
  url: string;
  wetAmount?: number;
  gainAmount?: number;
}

// Default initial value for AudioNodes ref (all null)
export const EMPTY_AUDIO_NODES: AudioNodes = {
  audioContext: null,
  audioElement: null,
  audioWorkletNode: null,
  inputGainNode: null,
  outputGainNode: null,
  bypassNode: null,
  irNode: null,
  irWetGain: null,
  irDryGain: null,
  irGain: null,
  sourceNode: null,
  playSourceNode: null,
  playInputGainNode: null,
  mediaStream: null,
  inputMeterNode: null,
  outputMeterNode: null,
  channelSplitterNode: null,
  channelMergerNode: null,
  channel0PlayMeter: null,
  channel1PlayMeter: null,
  outputWorkaroundDestination: null,
  outputWorkaroundElement: null,
};

// Utility type to ensure non-empty arrays
type NonEmptyArray<T> = [T, ...T[]];

export interface T3kPlayerProps {
  models?: NonEmptyArray<Model>;
  irs?: NonEmptyArray<IR>;
  inputs?: NonEmptyArray<Input>;
  isLoading?: boolean;
  previewMode?: PREVIEW_MODE;
  onPlay?: ({
    model,
    ir,
    input,
  }: {
    model: Model;
    ir: IR;
    input: Input;
  }) => void;
  onModelChange?: (model: Model) => void;
  onInputChange?: (input: Input) => void;
  onIrChange?: (ir: IR) => void;
  id?: string;
  infoSlot?: ReactNode;
}

export interface T3kSlimPlayerProps extends T3kPlayerProps {
  getData: () => Promise<{
    model: Model;
    ir: IR;
    input: Input;
  }>;
  size?: number;
}

export interface T3kAcordianPlayerProps extends T3kPlayerProps {
  getData: () => Promise<{
    models: NonEmptyArray<Model>;
    irs: NonEmptyArray<IR>;
    inputs: NonEmptyArray<Input>;
  }>;
  disabled?: boolean;
}
