import React, { memo, useCallback, useEffect, useRef, useState } from 'react';
import { useT3kPlayerContext } from '../../context/T3kPlayerContext';
import { T3kSlimPlayerProps } from '../../types';
import { Play } from '../ui/Play';
import { Pause } from '../ui/Pause';
import { CircularLoader } from '../ui/CircularLoader';

const SlimPlayerFC: React.FC<T3kSlimPlayerProps> = ({
  onPlayDemo,
  id,
  getData,
  size = 40,
}) => {
  const {
    audioState,
    getAudioNodes,
    init,
    loadAudio,
    syncEngineSettings,
    setPlaying,
    stopLiveInput,
    cleanup,
  } = useT3kPlayerContext();

  const [isLoading, setIsLoading] = useState(false);

  // Cleanup on unmount (same as usePlayerCore)
  useEffect(() => {
    return () => cleanup();
  }, [cleanup]);

  // Cache getData result so we don't re-fetch on every toggle
  const dataRef = useRef<Awaited<ReturnType<typeof getData>> | null>(null);
  const getDataCached = useCallback(async () => {
    if (!dataRef.current) {
      dataRef.current = await getData();
    }
    return dataRef.current;
  }, [getData]);

  const isThisPlayerActive = audioState.activePlayerId === id;

  // Listen for audio ended — reset playback when this player is active
  useEffect(() => {
    if (!isThisPlayerActive) return;

    const audioElement = getAudioNodes().audioElement;
    if (!audioElement) return;

    const handleEnded = () => setPlaying(false);

    audioElement.addEventListener('ended', handleEnded);
    return () => {
      audioElement.removeEventListener('ended', handleEnded);
    };
  }, [getAudioNodes, isThisPlayerActive, setPlaying]);

  const togglePlay = useCallback(async () => {
    if (!id) return;

    // Stop live input if engine is in live mode
    const { mediaStream } = getAudioNodes();
    if (mediaStream?.active) {
      stopLiveInput();
    }

    setIsLoading(true);

    try {
      const { model, ir, input } = await getDataCached();
      if (!model || !ir || !input) {
        console.error('SlimPlayer: getData returned incomplete data');
        return;
      }

      if (audioState.initState !== 'ready') {
        await init({ audioUrl: input.url });
      }

      if (isThisPlayerActive) {
        setPlaying(false);
      } else {
        // Load audio if needed
        if (!audioState.audioUrl || audioState.audioUrl !== input.url) {
          await loadAudio(input.url);
        }

        // Sync engine settings (model, IR, bypass)
        await syncEngineSettings({
          modelUrl: model.url,
          ir: { url: ir.url, mix: ir.mix, gain: ir.gain },
          bypassed: false,
        });

        setPlaying(true, id);

        onPlayDemo?.({ model, ir, input });
      }
    } catch (error) {
      console.error('Error in togglePlay:', error);
      setPlaying(false);
    } finally {
      setIsLoading(false);
    }
  }, [
    id,
    getAudioNodes,
    stopLiveInput,
    getDataCached,
    audioState.initState,
    audioState.audioUrl,
    isThisPlayerActive,
    init,
    loadAudio,
    syncEngineSettings,
    setPlaying,
    onPlayDemo,
  ]);

  return (
    <button
      onClick={togglePlay}
      className='p-0 focus:outline-none neural-amp-modeler'
      aria-label={isThisPlayerActive ? 'Pause' : 'Play'}
    >
      {isThisPlayerActive ? (
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
    return prevProps.id === nextProps.id;
  }
);

T3kSlimPlayer.displayName = 'T3kSlimPlayer';

export default T3kSlimPlayer;
