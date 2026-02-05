import { ReactNode } from "react";

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

// Source mode for players: 'preview' uses file playback, 'live' uses direct input
export type SourceMode = 'preview' | 'live';

// Channel selection for multi-channel audio interfaces
export type ChannelSelection = 'first' | 'second';

// What audio source is currently active (connected to the audio engine)
export type InputMode = { type: 'preview' } | { type: 'live' };

// Configured live input settings (persists even when preview is active)
// This allows the UI to show the configured device even when file playback is active
export interface LiveInputConfig {
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
  liveInputConfig: LiveInputConfig | null;
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
export type MicrophonePermissionStatus = 'idle' | 'pending' | 'granted' | 'denied' | 'blocked' | 'error';

export interface MicrophonePermissionState {
  status: MicrophonePermissionStatus;
  error: string | null;
}

// Audio input device state (device concerns only)
export interface AudioInputDeviceState {
  devices: AudioInputDevice[];
  isLoading: boolean;
  error: string | null;
  preferredDeviceId: string | null;  // Device selected by user in browser permission dialog
}

// Audio output device state
export interface AudioOutputDeviceState {
  devices: AudioOutputDevice[];
  selectedDeviceId: string | null;  // null means system default
}

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
