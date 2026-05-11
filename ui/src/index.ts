import './index.css';

export { default as T3kPlayer } from './components/T3kPlayer';
export {
  default as T3kDisabledPlayer,
  type T3kDisabledPlayerProps,
} from './components/T3kDisabledPlayer';
export type {
  T3kPlayerProps,
  T3kSlimPlayerProps,
  T3kAcordianPlayerProps,
  Model,
  ModelArchitecture,
  IR,
  Input,
  AudioInputDevice,
  InputMode,
  MicrophonePermissionStatus,
  MicrophonePermissionState,
  AudioInputDeviceState,
} from './types';
export { PREVIEW_MODE } from './types';
export {
  T3kPlayerContextProvider,
  useT3kPlayerContext,
} from './context/T3kPlayerContext';
export { T3kPlayerProvider } from './context/T3kPlayerProvider';
export { default as T3kSlimPlayer } from './components/Player/SlimPlayer';
export { default as T3kAcordianPlayer } from './components/Player/AcordianPlayer';

export { DEFAULT_MODELS, DEFAULT_IRS, DEFAULT_INPUTS } from './constants';
export { filterModelsByArchitecture } from './utils/player';
