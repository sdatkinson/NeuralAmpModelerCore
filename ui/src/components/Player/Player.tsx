import React, { memo, useCallback, useEffect, useRef, useState } from 'react';
import { useT3kPlayerContext } from '../../context/T3kPlayerContext';
import { Input, IR, Model, PREVIEW_MODE, T3kPlayerProps } from '../../types';
import { initVisualizer, setupVisualizer } from '../../utils/visualizer';
import { Play } from '../ui/Play';
import { Skip } from '../ui/Skip';
import { Select } from '../ui/Select';
import { ToggleSimple } from '../ui/ToggleSimple';
import { Pause } from '../ui/Pause';
import { LogoSm } from '../ui/LogoSm';
import { DEFAULT_INPUTS, DEFAULT_MODELS, DEFAULT_IRS } from '../../constants';
import { CircularLoader } from '../ui/CircularLoader';

const PlayerFC: React.FC<T3kPlayerProps> = ({
  models = DEFAULT_MODELS,
  irs = DEFAULT_IRS,
  inputs = DEFAULT_INPUTS,
  previewMode,
  onPlay,
  onModelChange,
  onInputChange,
  onIrChange,
  id,
}) => {
  const {
    audioState,
    getAudioNodes,
    init,
    loadModel,
    loadAudio,
    loadIr,
    removeIr,
    toggleBypass,
    connectVisualizerNode,
  } = useT3kPlayerContext();

  // Helper function to get default item
  const getDefault = useCallback(
    <T extends { default?: boolean }>(items: T[]): T => {
      return items.find(item => item.default) || items[0];
    },
    []
  );

  // State
  const [selectedModel, setSelectedModel] = useState<Model>(() =>
    getDefault(models)
  );
  const [selectedInput, setSelectedInput] = useState<Input>(() =>
    getDefault(inputs)
  );
  const [selectedIr, setSelectedIr] = useState<IR>(() => getDefault(irs));
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [isLoading, setIsLoading] = useState(false);

  // Refs
  const visualizerRef = useRef<HTMLCanvasElement>(null);
  const visualizerNodeRef = useRef<AnalyserNode | null>(null);
  const canvasWrapperRef = useRef<HTMLDivElement>(null);

  // Memoized options
  const modelOptions = React.useMemo(
    () =>
      models.map(model => ({
        label: model.name,
        value: model.url,
      })),
    [models]
  );

  const audioOptions = React.useMemo(
    () =>
      inputs.map(input => ({
        label: input.name,
        value: input.url,
      })),
    [inputs]
  );

  const irOptions = React.useMemo(
    () =>
      irs.map(ir => ({
        label: ir.name,
        value: ir.url,
      })),
    [irs]
  );

  // Utility functions
  const formatTime = useCallback((time: number) => {
    const minutes = Math.floor(time / 60);
    const seconds = Math.floor(time % 60);
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
  }, []);

  // Visualizer resize handler
  const handleResize = useCallback(() => {
    if (canvasWrapperRef.current && visualizerRef.current) {
      const wrapperWidth = canvasWrapperRef.current.clientWidth;
      visualizerRef.current.width = wrapperWidth;
      if (visualizerRef.current) {
        initVisualizer({ canvas: visualizerRef.current });
      }
    }
  }, []);

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

  // Setup resize listener
  useEffect(() => {
    handleResize();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [handleResize]);

  // Audio element event listeners
  useEffect(() => {
    const audioElement = getAudioNodes().audioElement;
    if (!audioElement) return;

    const handleTimeUpdate = () => setCurrentTime(audioElement.currentTime);
    const handleEnded = () => setIsPlaying(false);
    const handleLoadedMetadata = () => setDuration(audioElement.duration);

    // Set initial duration if already loaded
    if (audioElement.duration) {
      setDuration(audioElement.duration);
    }

    audioElement.addEventListener('timeupdate', handleTimeUpdate);
    audioElement.addEventListener('ended', handleEnded);
    audioElement.addEventListener('loadedmetadata', handleLoadedMetadata);

    return () => {
      audioElement.removeEventListener('timeupdate', handleTimeUpdate);
      audioElement.removeEventListener('ended', handleEnded);
      audioElement.removeEventListener('loadedmetadata', handleLoadedMetadata);
    };
  }, [getAudioNodes, audioState]);

  // Visualizer setup
  useEffect(() => {
    if (!audioState.isInitialized || !visualizerRef.current) return;

    const audioContext = getAudioNodes().audioContext;
    if (!audioContext) return;

    // Create and store new visualizer
    const visualizer = setupVisualizer(visualizerRef.current, audioContext);
    visualizerNodeRef.current = visualizer;

    // Connect visualizer using context method
    const disconnect = connectVisualizerNode(visualizer);

    return () => {
      disconnect();
      visualizerNodeRef.current = null;
    };
  }, [audioState.isInitialized, getAudioNodes, connectVisualizerNode]);

  // Event handlers
  const togglePlay = useCallback(async () => {
    setIsLoading(true);
    if (!audioState.isInitialized) await init({ audioUrl: selectedInput.url });
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
        if (!audioState.audioUrl || audioState.audioUrl !== selectedInput.url) {
          await loadAudio(selectedInput.url);
        }

        // Load model if needed
        if (!audioState.modelUrl || audioState.modelUrl !== selectedModel.url) {
          await loadModel(selectedModel.url);
        }

        // Handle IR loading
        if (selectedIr.url) {
          if (!audioState.irUrl || audioState.irUrl !== selectedIr.url) {
            await loadIr({
              url: selectedIr.url,
              wetAmount: selectedIr.mix,
              gainAmount: selectedIr.gain,
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
          model: selectedModel,
          ir: selectedIr,
          input: selectedInput,
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
    selectedInput,
    selectedModel,
    selectedIr,
    loadAudio,
    loadModel,
    loadIr,
    removeIr,
    onPlay,
  ]);

  const handleSkipToStart = useCallback(() => {
    const audioElement = getAudioNodes().audioElement;
    if (audioElement) {
      audioElement.currentTime = 0;
      setCurrentTime(0);
    }
  }, [getAudioNodes]);

  const handleBypassToggle = useCallback(() => {
    toggleBypass();
  }, [toggleBypass]);

  const handleModelChange = useCallback(
    async (value: string | number) => {
      const model = models.find(m => m.url === String(value));
      if (model) {
        setSelectedModel(model);
        try {
          if (audioState.isInitialized) await loadModel(model.url);
          onModelChange?.(model);
        } catch (error) {
          console.error('Error loading model:', error);
        }
      }
    },
    [models, loadModel, onModelChange, audioState.isInitialized]
  );

  const handleInputChange = useCallback(
    async (value: string | number) => {
      const wasPlaying = isPlaying;
      const input = inputs.find(i => i.url === String(value));

      if (!input) return;

      setSelectedInput(input);

      try {
        if (
          audioState.isInitialized &&
          (!audioState.audioUrl || audioState.audioUrl !== input.url)
        ) {
          await loadAudio(input.url);
        }

        const audioElement = getAudioNodes().audioElement;
        if (wasPlaying && audioElement) {
          audioElement.play();
        }

        onInputChange?.(input);
      } catch (error) {
        console.error('Error loading audio:', error);
      }
    },
    [
      isPlaying,
      inputs,
      audioState.audioUrl,
      audioState.isInitialized,
      getAudioNodes,
      loadAudio,
      onInputChange,
    ]
  );

  const handleIrChange = useCallback(
    async (value: string | number) => {
      const ir = irs.find(i => i.url === String(value));

      if (!ir) return;

      setSelectedIr(ir);

      try {
        if (audioState.isInitialized && ir.url) {
          await loadIr({
            url: ir.url,
            wetAmount: ir.mix,
            gainAmount: ir.gain,
          });
        } else {
          removeIr();
        }
        onIrChange?.(ir);
      } catch (error) {
        console.error('Error loading IR:', error);
      }
    },
    [irs, loadIr, removeIr, onIrChange, audioState.isInitialized]
  );

  const bypassedStyles = audioState.isBypassed
    ? 'opacity-50 touch-none cursor-not-allowed grayscale'
    : '';

  const renderModelSelect = () => (
    <Select
      options={modelOptions}
      label='Model'
      onChange={handleModelChange}
      defaultOption={selectedModel.url!}
      disabled={audioState.isBypassed}
    />
  );

  const renderIrSelect = () => (
    <Select
      options={irOptions}
      label='IR'
      onChange={handleIrChange}
      defaultOption={selectedIr.url}
      disabled={audioState.isBypassed}
    />
  );

  return (
    <div className='bg-zinc-900 border border-zinc-700 text-white p-4 lg:p-8 pt-0 lg:pt-2 rounded-xl w-full flex flex-col gap-6'>
      {/* Player Controls */}
      <div className='flex items-center gap-4 overflow-hidden'>
        <button
          onClick={togglePlay}
          className='p-0 focus:outline-none'
          aria-label={isPlaying ? 'Pause' : 'Play'}
        >
          {isPlaying ? (
            <Pause />
          ) : isLoading ? (
            <CircularLoader size={48} />
          ) : (
            <Play />
          )}
        </button>

        <button
          onClick={handleSkipToStart}
          className='p-0 focus:outline-none'
          aria-label='Skip to start'
        >
          <Skip opacity={currentTime > 0 ? 1 : 0.6} />
        </button>

        <div className='flex text-sm font-mono gap-2 text-zinc-400'>
          <span>{formatTime(currentTime)}</span>
          <span> / </span>
          <span>{formatTime(duration)}</span>
        </div>

        <div ref={canvasWrapperRef} className='flex-1'>
          <canvas
            ref={visualizerRef}
            height={130}
            style={{
              marginBottom: -20,
              marginTop: -20,
              width: '100%',
              height: '130px',
            }}
          />
        </div>
      </div>

      {/* Settings */}
      <div className='flex flex-col gap-2'>
        <div className='flex flex-row items-center gap-4 flex-wrap'>
          <div className='flex-1 min-w-[0px]'>
            <div className={`flex w-full flex-1 min-w-[0px] ${bypassedStyles}`}>
              {previewMode === PREVIEW_MODE.MODEL
                ? renderModelSelect()
                : renderIrSelect()}
            </div>
          </div>

          <div className='flex items-center pt-[24px] flex-shrink-0'>
            <ToggleSimple
              label=''
              onChange={handleBypassToggle}
              isChecked={!audioState.isBypassed}
              ariaLabel='Bypass'
            />
          </div>
        </div>

        <div className='flex flex-col sm:flex-row items-center gap-2 sm:gap-6'>
          <div className='w-full sm:w-1/2'>
            <Select
              options={audioOptions}
              label='Input'
              onChange={handleInputChange}
              defaultOption={selectedInput.url}
              disabled={audioState.isBypassed}
            />
          </div>

          <div className={`w-full sm:w-1/2 ${bypassedStyles}`}>
            {previewMode === PREVIEW_MODE.MODEL
              ? renderIrSelect()
              : renderModelSelect()}
          </div>
        </div>
      </div>

      {/* Footer */}
      <a
        href='https://www.tone3000.com'
        target='_blank'
        className='flex flex-row gap-2 items-center self-end'
      >
        <p className='text-zinc-400 text-xs'>Powered by</p>
        <LogoSm width={42} height={14} />
      </a>
    </div>
  );
};

const Player = memo(
  (props: T3kPlayerProps) => {
    return <PlayerFC {...props} />;
  },
  (prevProps, nextProps) => {
    return (
      JSON.stringify(prevProps.models) === JSON.stringify(nextProps.models) &&
      JSON.stringify(prevProps.irs) === JSON.stringify(nextProps.irs) &&
      JSON.stringify(prevProps.inputs) === JSON.stringify(nextProps.inputs)
    );
  }
);

Player.displayName = 'Player';

export default Player;
