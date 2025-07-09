import './index.css';

export { default as T3kPlayer } from './components/T3kPlayer';
export type { T3kPlayerProps, PREVIEW_MODE } from './types';
export {
  T3kPlayerContextProvider,
  useT3kPlayerContext,
} from './context/T3kPlayerContext';
