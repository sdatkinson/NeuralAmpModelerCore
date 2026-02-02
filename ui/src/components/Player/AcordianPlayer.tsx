import React, { memo, useCallback, useEffect, useRef, useState } from 'react';
import { useT3kPlayerContext } from '../../context/T3kPlayerContext';
import { Input, IR, Model, PREVIEW_MODE, T3kAcordianPlayerProps } from '../../types';
import { initVisualizer, setupVisualizer } from '../../utils/visualizer';
import { Play } from '../ui/Play';
import { Skip } from '../ui/Skip';
import { Select } from '../ui/Select';
import { ToggleSimple } from '../ui/ToggleSimple';
import { Pause } from '../ui/Pause';
import { LogoSm } from '../ui/LogoSm';
import { DEFAULT_INPUTS, DEFAULT_MODELS, DEFAULT_IRS } from '../../constants';
import { CircularLoader } from '../ui/CircularLoader';
import { ChevronDown, ChevronUp } from 'lucide-react';

const PlayerFC: React.FC<T3kAcordianPlayerProps> = ({
  getData = async () => ({ models: DEFAULT_MODELS, irs: DEFAULT_IRS, inputs: DEFAULT_INPUTS }),
  previewMode,
  onPlay,
  onModelChange,
  onInputChange,
  onIrChange,
  id,
  disabled = false,
  infoSlot,
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
    cleanup,
  } = useT3kPlayerContext();

  // Helper function to get default item
  const getDefault = useCallback(
    <T extends { default?: boolean }>(items: T[]): T => {
      return items.find(item => item.default) || items[0];
    },
    []
  );

  // State
  const [selectedModel, setSelectedModel] = useState<Model | null>(null);
  const [selectedInput, setSelectedInput] = useState<Input | null>(null);
  const [selectedIr, setSelectedIr] = useState<IR | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [isLoading, setIsLoading] = useState(false);
  const [expanded, setExpanded] = useState(false);
  const [models, setModels] = useState<Model[] | null>(null);
  const [inputs, setInputs] = useState<Input[] | null>(null);
  const [irs, setIrs] = useState<IR[] | null>(null);
  const [bypassed, setBypassed] = useState(false);

  // Refs
  const visualizerRef = useRef<HTMLCanvasElement>(null);
  const visualizerNodeRef = useRef<AnalyserNode | null>(null);
  const canvasWrapperRef = useRef<HTMLDivElement>(null);

  // Memoized options
  const modelOptions = React.useMemo(
    () =>
      models?.map(model => ({
        label: model.name,
        value: model.url,
      })),
    [models]
  );

  const audioOptions = React.useMemo(
    () =>
      inputs?.map(input => ({
        label: input.name,
        value: input.url,
      })),
    [inputs]
  );

  const irOptions = React.useMemo(
    () =>
      irs?.map(ir => ({
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

  // Setup unload effect to cleanup audio context
  useEffect(() => {
    return () => {
      cleanup();
    };
  }, [cleanup]);

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
    if (!isPlaying) return; // Only listen to time updates if the audio is playing
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
  }, [getAudioNodes, audioState, isPlaying]);

  // Visualizer setup
  useEffect(() => {
    if (!isPlaying) return; // Only setup visualizer if the audio is playing
    if (audioState.initState !== 'ready' || !visualizerRef.current) return;

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
  }, [audioState.initState, getAudioNodes, connectVisualizerNode, isPlaying]);

  const getDataWithCache = useCallback(async () => {
    if (selectedModel && selectedInput && selectedIr) {
      return { model: selectedModel, ir: selectedIr, input: selectedInput };
    }
    const data = await getData();
    setModels(data.models);
    setIrs(data.irs);
    setInputs(data.inputs);
    const model = getDefault(data.models);
    const input = getDefault(data.inputs);
    const ir = getDefault(data.irs);
    setSelectedModel(model);
    setSelectedInput(input);
    setSelectedIr(ir);
    return { model, ir, input };
  }, [getData, selectedModel, selectedInput, selectedIr]);

  // Event handlers
  const togglePlay = useCallback(async () => {
    setIsLoading(true);
    const { model, ir, input } = await getDataWithCache();
    if (audioState.initState !== 'ready') await init({ audioUrl: input.url });
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
        // Toggle bypass if needed
        if (audioState.isBypassed !== bypassed) toggleBypass();
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
    getDataWithCache,
    loadAudio,
    loadModel,
    loadIr,
    removeIr,
    onPlay,
    toggleBypass,
    bypassed,
  ]);

  const handleSkipToStart = useCallback(() => {
    const audioElement = getAudioNodes().audioElement;
    if (audioElement) {
      audioElement.currentTime = 0;
      setCurrentTime(0);
    }
  }, [getAudioNodes]);

  const handleBypassToggle = useCallback(() => {
    const newBypassed = !bypassed;
    setBypassed(newBypassed);
    if (isPlaying && audioState.isBypassed !== newBypassed) {
      toggleBypass();
    }
  }, [toggleBypass, bypassed, audioState, isPlaying]);

  const handleModelChange = useCallback(
    async (value: string | number) => {
      const model = models?.find(m => m.url === String(value));
      if (model) {
        setSelectedModel(model);
        try {
          if (audioState.initState === 'ready' && isPlaying) await loadModel(model.url);
          onModelChange?.(model);
        } catch (error) {
          console.error('Error loading model:', error);
        }
      }
    },
    [models, loadModel, onModelChange, audioState.initState, isPlaying]
  );

  const handleInputChange = useCallback(
    async (value: string | number) => {
      const wasPlaying = isPlaying;
      const input = inputs?.find(i => i.url === String(value));

      if (!input) return;

      setSelectedInput(input);

      try {
        if (wasPlaying) {
          if (
            audioState.initState === 'ready' &&
            (!audioState.audioUrl || audioState.audioUrl !== input.url)
          ) {
            await loadAudio(input.url);
          }

          const audioElement = getAudioNodes().audioElement;
          if (audioElement) {
            audioElement.play();
          }
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
      audioState.initState,
      getAudioNodes,
      loadAudio,
      onInputChange,
    ]
  );

  const handleIrChange = useCallback(
    async (value: string | number) => {
      const ir = irs?.find(i => i.url === String(value));

      if (!ir) return;

      setSelectedIr(ir);

      try {
        if (isPlaying) {
          if (audioState.initState === 'ready' && ir.url) {
            await loadIr({
              url: ir.url,
              wetAmount: ir.mix,
              gainAmount: ir.gain,
            });
          } else {
            removeIr();
          }
        }
        onIrChange?.(ir);
      } catch (error) {
        console.error('Error loading IR:', error);
      }
    },
    [irs, loadIr, removeIr, onIrChange, audioState.initState, isPlaying]
  );

  const bypassedStyles = bypassed
    ? 'opacity-50 touch-none cursor-not-allowed grayscale'
    : '';

  const renderModelSelect = () => (
    <Select
      options={modelOptions || []}
      label='Model'
      onChange={handleModelChange}
      defaultOption={selectedModel?.url!}
      disabled={bypassed}
    />
  );

  const renderIrSelect = () => (
    <Select
      options={irOptions || []}
      label='IR'
      onChange={handleIrChange}
      defaultOption={selectedIr?.url}
      disabled={bypassed}
    />
  );

  return (
    <div className={`bg-zinc-900 border border-t-0 border-zinc-700 text-white px-4 sm:px-6 pt-0 pb-0 rounded-xl w-full flex flex-col gap-0 md:gap-1 rounded-t-none ${expanded ? 'pb-4 lg:pb-8' : 'pb-0'}`}>
      {/* Player Controls */}
      <div className={`flex items-center gap-4 overflow-hidden ${disabled ? 'opacity-50 touch-none cursor-not-allowed' : ''}`}>
        <button
          onClick={togglePlay}
          className={`p-0 focus:outline-none ${disabled ? 'cursor-not-allowed' : ''}`}
          aria-label={isPlaying ? 'Pause' : 'Play'}
          disabled={disabled}
        >
          {isPlaying ? (
            <Pause size={40} />
          ) : isLoading ? (
            <CircularLoader size={40} />
          ) : (
            <Play size={40} />
          )}
        </button>

        <button
          onClick={handleSkipToStart}
          className={`p-0 focus:outline-none ${disabled ? 'cursor-not-allowed' : ''}`}
          aria-label='Skip to start'
          disabled={disabled}
        >
          <Skip size={24} opacity={currentTime > 0 ? 1 : 0.6} />
        </button>

        <div className='hidden sm:flex text-sm font-mono gap-2 text-zinc-400'>
          <span>{formatTime(currentTime)}</span>
          <span> / </span>
          <span>{formatTime(duration)}</span>
        </div>

        <div ref={canvasWrapperRef} className='flex-1'>
          <canvas
            ref={visualizerRef}
            height={120}
            style={{
              marginBottom: -20,
              marginTop: -20,
              width: '100%',
              height: '120px',
            }}
          />
        </div>

        {infoSlot && infoSlot}

        <button
          onClick={async () => {
            await getDataWithCache();
            setExpanded(!expanded)
          }}
          className={`p-0 focus:outline-none ${disabled ? 'cursor-not-allowed' : ''}`}
          aria-label='Toggle accordion'
          disabled={disabled}
        >
          {expanded ? <ChevronUp size={24} /> : <ChevronDown size={24} />}
        </button>

      </div>

      {expanded && (
        <>
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
              isChecked={!bypassed}
              ariaLabel='Bypass'
              disabled={disabled}
            />
          </div>
        </div>

        <div className='flex flex-col sm:flex-row items-center gap-2 sm:gap-6'>
          <div className='w-full sm:w-1/2'>
            <Select
              options={audioOptions || []}
              label='Input'
              onChange={handleInputChange}
              defaultOption={selectedInput?.url}
              // disabled={audioState.isBypassed}
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
        className='hidden flex flex-row gap-2 items-center self-end'
      >
        <p className='text-zinc-400 text-xs'>Powered by</p>
        <LogoSm width={42} height={14} />
      </a>
        </>
      )}
    </div>
  );
};

const T3kAcordianPlayer = memo(
  (props: T3kAcordianPlayerProps) => {
    return <PlayerFC {...props} />;
  },
  (prevProps, nextProps) => {
    return (
      JSON.stringify(prevProps.id) === JSON.stringify(nextProps.id) &&
      JSON.stringify(prevProps.previewMode) === JSON.stringify(nextProps.previewMode) &&
      prevProps.getData === nextProps.getData &&
      prevProps.disabled === nextProps.disabled
    );
  }
);

T3kAcordianPlayer.displayName = 'T3kAcordianPlayer';

export default T3kAcordianPlayer;