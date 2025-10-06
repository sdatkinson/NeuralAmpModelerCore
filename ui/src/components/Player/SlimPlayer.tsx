import React, { memo, useCallback, useEffect, useState } from 'react';
import { useT3kPlayerContext } from '../../context/T3kPlayerContext';
import { T3kSlimPlayerProps } from '../../types';
import { Play } from '../ui/Play';
import { Pause } from '../ui/Pause';
import { CircularLoader } from '../ui/CircularLoader';

const SlimPlayerFC: React.FC<T3kSlimPlayerProps> = ({
  onPlay,
  id,
  getData,
  size = 40,
}) => {
  const {
    audioState,
    getAudioNodes,
    init,
    loadModel,
    loadAudio,
    loadIr,
    removeIr,
  } = useT3kPlayerContext();

  // State
  const [isPlaying, setIsPlaying] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  // Setup play event listener
  useEffect(() => {
    if (!id) return;
    const handlePlay = (event: Event) => {
      if ((event as CustomEvent).detail.id !== id) {
        setIsPlaying(false);
      }
    };
    window.addEventListener('t3k-player-play', handlePlay);
    return () => window.removeEventListener('t3k-player-play', handlePlay);
  }, [id]);

  // Event handlers
  const togglePlay = useCallback(async () => {
    setIsLoading(true);
    const { model, ir, input } = await getData();
    if (!model || !ir || !input) {
      console.error('Error getting data:');
      setIsLoading(false);
      return;
    }
    if (!audioState.isInitialized) await init({ audioUrl: input.url });
    const nodes = getAudioNodes();
    const { audioElement, audioContext } = nodes;

    try {
      if (isPlaying) {
        if (audioElement) audioElement.pause();
        setIsPlaying(false);
      } else {
        // Ensure audio context is running
        if (audioContext) {
          await audioContext.resume();
        }

        // Load audio if needed
        if (!audioState.audioUrl || audioState.audioUrl !== input.url) {
          await loadAudio(input.url);
        }

        // Load model if needed
        if (!audioState.modelUrl || audioState.modelUrl !== model.url) {
          await loadModel(model.url);
        }

        // Handle IR loading
        if (ir.url) {
          if (!audioState.irUrl || audioState.irUrl !== ir.url) {
            await loadIr({
              url: ir.url,
              wetAmount: ir.mix,
              gainAmount: ir.gain,
            });
          }
        } else {
          removeIr();
        }

        if (audioElement) await audioElement.play();
        setIsPlaying(true);
        if (id) {
          try {
            // Emit play event to window
            window.dispatchEvent(
              new CustomEvent('t3k-player-play', { detail: { id } })
            );
          } catch (error) {
            console.error('Error emitting play event:', error);
          }
        }
        onPlay?.({
          model: model,
          ir: ir,
          input: input,
        });
      }
    } catch (error) {
      console.error('Error in togglePlay:', error);
      setIsPlaying(false);
    } finally {
      setIsLoading(false);
    }
  }, [
    isPlaying,
    getAudioNodes,
    audioState,
    loadAudio,
    loadModel,
    loadIr,
    removeIr,
    onPlay,
  ]);

  return (
    <button
      onClick={togglePlay}
      className='p-0 focus:outline-none'
      aria-label={isPlaying ? 'Pause' : 'Play'}
    >
      {isPlaying ? (
        <Pause size={size} />
      ) : isLoading ? (
        <CircularLoader size={size} />
      ) : (
        <Play size={size} />
      )}
    </button>
  );
};

const T3kSlimPlayer = memo(
  (props: T3kSlimPlayerProps) => {
    return <SlimPlayerFC {...props} />;
  },
  (prevProps, nextProps) => {
    return JSON.stringify(prevProps.id) === JSON.stringify(nextProps.id);
  }
);

T3kSlimPlayer.displayName = 'T3kSlimPlayer';

export default T3kSlimPlayer;
