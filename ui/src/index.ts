import './index.css';

export { default as T3kPlayer } from './components/T3kPlayer';
export type { T3kPlayerProps, T3kSlimPlayerProps, Model, IR, Input } from './types';
export { PREVIEW_MODE } from './types';
export {
  T3kPlayerContextProvider,
  useT3kPlayerContext,
} from './context/T3kPlayerContext';
export { default as T3kSlimPlayer } from './components/Player/SlimPlayer';

export { DEFAULT_MODELS, DEFAULT_IRS, DEFAULT_INPUTS } from './constants';
