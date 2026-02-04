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

// Channel selection for multi-channel audio interfaces
export type ChannelSelection = 'first' | 'second';

// Discriminated union for input modes
export type InputMode =
  | { type: 'preview' }  // File playback (preexisting functionality)
  | {
      type: 'live';
      deviceId?: string;
      channelCount?: number;              // Number of available channels (1 or 2)
      selectedChannel?: ChannelSelection; // Which channel to route to processing
      channelGains?: Record<ChannelSelection, number>;  // Per-channel input gain in dB
    };

export interface AudioInputDevice {
  deviceId: string;
  label: string;
}

export interface AudioOutputDevice {
  deviceId: string;
  label: string;
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
