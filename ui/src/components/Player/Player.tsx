import React, { memo, useCallback, useEffect, useRef, useState } from 'react';
import { useT3kPlayerContext } from '../../context/T3kPlayerContext';
import { Input, IR, Model, PREVIEW_MODE, SourceMode, T3kPlayerProps } from '../../types';
import { initVisualizer, setupVisualizer } from '../../utils/visualizer';
import { Play } from '../ui/Play';
import { Skip } from '../ui/Skip';
import { Select } from '../ui/Select';
import { ToggleSimple } from '../ui/ToggleSimple';
import { Pause } from '../ui/Pause';
import { LogoSm } from '../ui/LogoSm';
import { DEFAULT_INPUTS, DEFAULT_MODELS, DEFAULT_IRS } from '../../constants';
import { CircularLoader } from '../ui/CircularLoader';
import { SegmentedControl } from '../ui/SegmentedControl';
import { InputControlStrip } from '../ControlStrip';
import { Loader2, Plug, Settings } from 'lucide-react';
import { useSourceMode } from '../../hooks/useSourceMode';
import { useToast } from '../../hooks/useToast';

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
  infoSlot,
}) => {
  const {
    audioState,
    audioInputDevices,
    getAudioNodes,
    init,
    loadModel,
    loadAudio,
    loadIr,
    removeIr,
    setBypass,
    syncEngineSettings,
    connectVisualizerNode,
    cleanup,
    startLiveInput,
    stopLiveInput,
    setPlaying,
  } = useT3kPlayerContext();

  // Source mode hook (per-player)
  const {
    sourceMode,
    showPlaybackPausedMessage,
    liveDeviceOptions,
    handleSourceModeChange,
    handleLiveDeviceChange,
  } = useSourceMode({ playerId: id });

  // Derived from context
  const isLiveConfigured = audioState.liveInputConfig !== null;
  const currentDeviceId = audioState.liveInputConfig?.deviceId ?? null;

  // Global settings dialog from context
  const { openSettingsDialog: openDialog } = useT3kPlayerContext();

  // Toast messages (e.g. output device fallback)
  const toastMessage = useToast();

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
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [isLoading, setIsLoading] = useState(false);
  const [bypassed, setBypassed] = useState(false);

  const openSettingsDialog = useCallback(() => {
    openDialog({
      sourceMode,
      playerId: id,
      selectedModel,
      selectedIr,
    });
  }, [openDialog, sourceMode, id, selectedModel, selectedIr]);

  // Helper: Sync this player's preferences to the audio engine when taking over
  const syncPlayerToEngine = useCallback(async () => {
    await syncEngineSettings({
      modelUrl: selectedModel.url,
      ir: { url: selectedIr.url, mix: selectedIr.mix, gain: selectedIr.gain },
      bypassed,
    });
  }, [syncEngineSettings, selectedModel.url, selectedIr, bypassed]);

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

  // Setup unload effect to cleanup audio context
  useEffect(() => {
    return () => {
      cleanup();
    };
  }, [cleanup]);

  // Setup resize listener
  useEffect(() => {
    handleResize();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [handleResize]);

  // Derived state: is THIS player the active one?
  const isThisPlayerActive = audioState.activePlayerId === id;

  // Audio element event listeners - only when THIS player is active
  useEffect(() => {
    if (!isThisPlayerActive) return; // Only listen when this player is playing

    const audioElement = getAudioNodes().audioElement;
    if (!audioElement) return;

    const handleTimeUpdate = () => setCurrentTime(audioElement.currentTime);
    const handleEnded = () => setPlaying(false);
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
  }, [getAudioNodes, isThisPlayerActive, setPlaying]);

  // Visualizer setup - only when THIS player is active
  useEffect(() => {
    if (!isThisPlayerActive) return; // Only connect visualizer when this player is playing
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
  }, [audioState.initState, getAudioNodes, connectVisualizerNode, isThisPlayerActive]);

  // Event handlers
  const togglePlay = useCallback(async () => {
    const wantsLiveInput = sourceMode === 'live';
    const liveInputActive = audioState.inputMode.type === 'live';
    const isThisPlayerActive = audioState.activePlayerId === id;

    // === LIVE MODE ===
    if (wantsLiveInput) {
      const { audioContext } = getAudioNodes();

      // If audio engine isn't in live mode yet, reconnect using the configured device
      if (!liveInputActive && audioState.liveInputConfig) {
        await startLiveInput(audioState.liveInputConfig.deviceId, {
          initialChannel: audioState.liveInputConfig.selectedChannel,
          initialChannelGains: audioState.liveInputConfig.channelGains,
        });
      }

      // Ensure audio context is running
      if (audioContext && audioContext.state === 'suspended') {
        await audioContext.resume();
      }

      // Sync this player's settings to engine
      await syncPlayerToEngine();

      // Toggle playback
      if (isThisPlayerActive) {
        setPlaying(false);
      } else {
        setPlaying(true, id);
      }
      return;
    }

    // === PREVIEW MODE ===
    // Check actual node state (not React state, which can be stale due to closure capture)
    const { mediaStream } = getAudioNodes();
    if (mediaStream?.active) {
      stopLiveInput();
    }

    setIsLoading(true);

    try {
      if (audioState.initState !== 'ready') {
        await init({ audioUrl: selectedInput.url });
      }

      if (isThisPlayerActive) {
        setPlaying(false);
      } else {
        // Load audio file if needed
        if (!audioState.audioUrl || audioState.audioUrl !== selectedInput.url) {
          await loadAudio(selectedInput.url);
        }

        // Sync this player's settings to engine
        await syncPlayerToEngine();

        // Start playback
        setPlaying(true, id);

        onPlay?.({
          model: selectedModel,
          ir: selectedIr,
          input: selectedInput,
        });
      }
    } catch (error) {
      console.error('Error in togglePlay:', error);
      setPlaying(false);
    } finally {
      setIsLoading(false);
    }
  }, [
    id,
    audioState.inputMode.type,
    audioState.activePlayerId,
    audioState.liveInputConfig,
    audioState.initState,
    audioState.audioUrl,
    sourceMode,
    selectedInput,
    selectedModel,
    selectedIr,
    getAudioNodes,
    init,
    loadAudio,
    startLiveInput,
    stopLiveInput,
    setPlaying,
    syncPlayerToEngine,
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
    const newBypassed = !bypassed;
    setBypassed(newBypassed);
    // Only sync to context if this player is currently active
    if (isThisPlayerActive) {
      setBypass(newBypassed);
    }
  }, [setBypass, bypassed, isThisPlayerActive]);

  const handleModelChange = useCallback(
    async (value: string | number) => {
      const model = models.find(m => m.url === String(value));
      if (model) {
        setSelectedModel(model);
        try {
          // Only load model if this player is active (avoid affecting other players)
          if (audioState.initState === 'ready' && isThisPlayerActive) {
            await loadModel(model.url);
          }
          onModelChange?.(model);
        } catch (error) {
          console.error('Error loading model:', error);
        }
      }
    },
    [models, loadModel, onModelChange, audioState.initState, isThisPlayerActive]
  );

  const handleInputChange = useCallback(
    async (value: string | number) => {
      // Check if THIS player was playing (not just any player)
      const wasThisPlayerPlaying = audioState.activePlayerId === id;
      const input = inputs.find(i => i.url === String(value));

      if (!input) return;

      setSelectedInput(input);

      try {
        if (
          audioState.initState === 'ready' &&
          (!audioState.audioUrl || audioState.audioUrl !== input.url)
        ) {
          await loadAudio(input.url);
        }

        // Resume playback if THIS player was playing before
        if (wasThisPlayerPlaying) {
          setPlaying(true, id);
        }

        onInputChange?.(input);
      } catch (error) {
        console.error('Error loading audio:', error);
      }
    },
    [
      id,
      audioState.activePlayerId,
      audioState.audioUrl,
      audioState.initState,
      inputs,
      loadAudio,
      onInputChange,
      setPlaying,
    ]
  );

  const handleIrChange = useCallback(
    async (value: string | number) => {
      const ir = irs.find(i => i.url === String(value));

      if (!ir) return;

      setSelectedIr(ir);

      try {
        // Only load IR if this player is active (avoid affecting other players)
        if (isThisPlayerActive) {
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
    [irs, loadIr, removeIr, onIrChange, audioState.initState, isThisPlayerActive]
  );

  // Source mode options for segmented control
  const sourceModeOptions = React.useMemo(
    () => [
      { value: 'preview' as const, label: 'Preview' },
      { value: 'live' as const, label: 'Live' },
    ],
    []
  );

  const bypassedStyles = bypassed
    ? 'opacity-50 touch-none cursor-not-allowed grayscale'
    : '';

  const renderModelSelect = () => (
    <Select
      options={modelOptions}
      label='Model'
      onChange={handleModelChange}
      value={selectedModel.url!}
      disabled={bypassed}
    />
  );

  const renderIrSelect = () => (
    <Select
      options={irOptions}
      label='IR'
      onChange={handleIrChange}
      value={selectedIr.url}
      disabled={bypassed}
    />
  );

  return (
    <div className='bg-zinc-900 border border-zinc-700 text-white p-4 lg:p-8 pt-0 lg:pt-2 rounded-xl w-full flex flex-col gap-6'>
      {/* Player Controls */}
      <div className='flex items-center gap-4 overflow-hidden'>
        <button
          onClick={togglePlay}
          className={`p-0 focus:outline-none ${sourceMode === 'live' && !isLiveConfigured ? 'opacity-50 cursor-not-allowed' : ''}`}
          disabled={sourceMode === 'live' && !isLiveConfigured}
          aria-label={isThisPlayerActive ? 'Pause' : 'Play'}
        >
          {isThisPlayerActive ? (
            <Pause />
          ) : isLoading ? (
            <CircularLoader size={48} />
          ) : (
            <Play />
          )}
        </button>

        <button
          onClick={handleSkipToStart}
          className={`p-0 focus:outline-none ${sourceMode === 'live' ? 'opacity-50 cursor-not-allowed' : ''}`}
          disabled={sourceMode === 'live'}
          aria-label='Skip to start'
        >
          <Skip opacity={sourceMode === 'live' ? 0.6 : currentTime > 0 ? 1 : 0.6} />
        </button>

        <div className='hidden sm:flex text-sm font-mono gap-2 text-zinc-400'>
          <span>{sourceMode === 'live' ? '-:--' : formatTime(currentTime)}</span>
          <span> / </span>
          <span>{sourceMode === 'live' ? '-:--' : formatTime(duration)}</span>
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
        {infoSlot && infoSlot}
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
              isChecked={!bypassed}
              ariaLabel='Bypass'
            />
          </div>
        </div>

        {/* Source mode segmented control with settings button */}
        <div className='flex flex-col gap-1 pt-2'>
          <span className='text-sm text-zinc-400'>Source</span>
          <div className='flex items-center gap-3'>
            <SegmentedControl
              options={sourceModeOptions}
              value={sourceMode}
              onChange={handleSourceModeChange}
            />
            {/* Settings button - always visible */}
            <button
              onClick={openSettingsDialog}
              className='p-2 rounded-md transition-colors border border-zinc-700 hover:bg-zinc-800'
              aria-label='Settings'
            >
              <Settings size={20} className='text-zinc-400' />
            </button>
            {showPlaybackPausedMessage && (
              <span className='text-xs text-zinc-400 animate-pulse'>
                Playback paused
              </span>
            )}
            {toastMessage && (
              <span className='text-xs text-zinc-400 animate-pulse'>
                {toastMessage}
              </span>
            )}
          </div>
        </div>

        {/* Input and IR row - always aligned */}
        <div className='flex flex-col sm:flex-row items-start gap-2 sm:gap-6'>
          <div className='w-full sm:w-1/2'>
            {/* Preview mode: show input dropdown */}
            {sourceMode === 'preview' && (
              <Select
                options={audioOptions}
                label='Input'
                onChange={handleInputChange}
                value={selectedInput.url}
              />
            )}
            
            {/* Live mode: not connected - show setup button */}
            {sourceMode === 'live' && !isLiveConfigured && (
              <div className='inline-flex flex-1 flex-col gap-1 w-full'>
                <span className='text-sm text-zinc-400'>Live Input</span>
                <div className='relative flex-1'>
                  <button
                    onClick={openSettingsDialog}
                    className='flex items-center gap-2 w-full overflow-hidden px-4 py-3 text-md border border-zinc-700 rounded-md bg-transparent hover:bg-zinc-800 transition-colors focus:outline-none'
                  >
                    <Plug size={18} className='text-zinc-400 flex-shrink-0' />
                    <span className='text-ellipsis text-nowrap overflow-hidden min-w-0'>Enable Live Input</span>
                  </button>
                  {/* Error or helper text - positioned below without affecting layout */}
                  {audioInputDevices.error ? (
                    <span className='absolute top-full mt-1 text-xs text-red-400'>
                      {audioInputDevices.error}
                    </span>
                  ) : (
                    <span className='absolute top-full mt-1 text-xs text-zinc-500'>
                      Use headphones to avoid feedback.
                    </span>
                  )}
                </div>
              </div>
            )}

            {/* Live mode: connected - show device dropdown (settings button is next to Source selector) */}
            {sourceMode === 'live' && isLiveConfigured && (
              <div className='w-full'>
                {audioState.inputMode.type === 'connecting' ? (
                  <div className='flex flex-col gap-1 w-full'>
                    <span className='text-sm text-zinc-400'>Live Input</span>
                    <div className='flex items-center justify-between w-full px-4 py-3 text-md border border-zinc-700 rounded-md bg-transparent opacity-50 cursor-wait'>
                      <span className='text-zinc-400'>Switching device...</span>
                      <Loader2 size={24} className='text-zinc-400 animate-spin' />
                    </div>
                  </div>
                ) : (
                  <Select
                    options={liveDeviceOptions}
                    label='Live Input'
                    onChange={(value) => handleLiveDeviceChange(String(value))}
                    value={currentDeviceId ?? ''}
                  />
                )}
              </div>
            )}
          </div>

          <div className={`w-full sm:w-1/2 ${bypassedStyles}`}>
            {previewMode === PREVIEW_MODE.MODEL
              ? renderIrSelect()
              : renderModelSelect()}
          </div>
        </div>
      </div>

      {/* Live Input Control Strip - only when live input is connected */}
      {sourceMode === 'live' && isLiveConfigured && (
        <InputControlStrip isActive={isThisPlayerActive} />
      )}

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
