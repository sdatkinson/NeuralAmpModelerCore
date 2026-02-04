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
import { SegmentedControl } from '../ui/SegmentedControl';
import { SettingsDialog } from '../SettingsDialog';
import { InputControlStrip } from '../ControlStrip';
import { Loader2, Plug, Settings } from 'lucide-react';

type SourceMode = 'preview' | 'live';

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
    audioOutputDevices,
    getAudioNodes,
    init,
    loadModel,
    loadAudio,
    loadIr,
    removeIr,
    toggleBypass,
    connectVisualizerNode,
    cleanup,
    startLiveInput,
    stopLiveInput,
    setPlaying,
    saveSettingsSnapshot,
    restoreSettingsSnapshot,
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
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [isLoading, setIsLoading] = useState(false);
  
  // Source mode state (preview = file playback, live = direct input)
  const [sourceMode, setSourceMode] = useState<SourceMode>('preview');
  const [showPlaybackPausedMessage, setShowPlaybackPausedMessage] = useState(false);
  const [showOutputFallbackMessage, setShowOutputFallbackMessage] = useState(false);
  const [isSettingsDialogOpen, setIsSettingsDialogOpen] = useState(false);

  // Refs
  const visualizerRef = useRef<HTMLCanvasElement>(null);
  const visualizerNodeRef = useRef<AnalyserNode | null>(null);
  const canvasWrapperRef = useRef<HTMLDivElement>(null);
  const prevOutputDeviceIdRef = useRef<string | null>(audioOutputDevices.selectedDeviceId);

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

  // Live device options for Select component
  const liveDeviceOptions = React.useMemo(
    () =>
      audioInputDevices.devices.map(device => ({
        label: device.label,
        value: device.deviceId,
      })),
    [audioInputDevices.devices]
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

  // Detect output device fallback (when selected device is disconnected)
  useEffect(() => {
    const prevId = prevOutputDeviceIdRef.current;
    const currentId = audioOutputDevices.selectedDeviceId;

    // Detect transition from non-null to null (fallback to default)
    if (prevId !== null && currentId === null) {
      setShowOutputFallbackMessage(true);
      setTimeout(() => setShowOutputFallbackMessage(false), 3000);
    }

    prevOutputDeviceIdRef.current = currentId;
  }, [audioOutputDevices.selectedDeviceId]);

  // Setup play event listener
  useEffect(() => {
    if (!id) return;
    const handlePlay = (event: Event) => {
      if ((event as CustomEvent).detail.id !== id) {
        setPlaying(false);
      }
    };
    window.addEventListener('t3k-player-play', handlePlay);
    return () => window.removeEventListener('t3k-player-play', handlePlay);
  }, [id, setPlaying]);

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
  }, [getAudioNodes, audioState]);

  // Visualizer setup
  useEffect(() => {
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
  }, [audioState.initState, getAudioNodes, connectVisualizerNode]);

  // Event handlers
  const togglePlay = useCallback(async () => {
    const isLiveMode = sourceMode === 'live' && audioState.inputMode.type === 'live';

    // Live mode: toggle via context (handles output gain)
    if (isLiveMode) {
      const { audioContext } = getAudioNodes();
      // Ensure audio context is running
      if (audioContext && audioContext.state === 'suspended') {
        await audioContext.resume();
      }
      setPlaying(!audioState.isPlaying);
      return;
    }

    // Preview mode: standard file playback
    setIsLoading(true);
    if (audioState.initState !== 'ready') await init({ audioUrl: selectedInput.url });

    try {
      if (audioState.isPlaying) {
        // Pause playback
        setPlaying(false);
      } else {
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

        // Start playback via context (handles audioElement.play() and state)
        setPlaying(true);

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
      setPlaying(false);
    } finally {
      setIsLoading(false);
    }
  }, [
    getAudioNodes,
    audioState,
    sourceMode,
    selectedInput,
    selectedModel,
    selectedIr,
    init,
    loadAudio,
    loadModel,
    loadIr,
    removeIr,
    onPlay,
    setPlaying,
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
          if (audioState.initState === 'ready') await loadModel(model.url);
          onModelChange?.(model);
        } catch (error) {
          console.error('Error loading model:', error);
        }
      }
    },
    [models, loadModel, onModelChange, audioState.initState]
  );

  const handleInputChange = useCallback(
    async (value: string | number) => {
      const wasPlaying = audioState.isPlaying;
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

        // Resume playback if it was playing before
        if (wasPlaying) {
          setPlaying(true);
        }

        onInputChange?.(input);
      } catch (error) {
        console.error('Error loading audio:', error);
      }
    },
    [
      audioState.isPlaying,
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
        if (audioState.initState === 'ready' && ir.url) {
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
    [irs, loadIr, removeIr, onIrChange, audioState.initState]
  );

  // Handle source mode change (Preview <-> Live)
  const handleSourceModeChange = useCallback(
    async (newMode: SourceMode) => {
      if (newMode === sourceMode) return;

      if (newMode === 'live') {
        // Switching from Preview to Live
        if (audioState.isPlaying) {
          // Pause playback but preserve playhead position
          setPlaying(false);
          setShowPlaybackPausedMessage(true);

          // Auto-hide the message after 3 seconds
          setTimeout(() => {
            setShowPlaybackPausedMessage(false);
          }, 3000);
        }

        // Restore live input settings if we have a saved snapshot
        // Don't restore output device - it should persist as currently set
        if (audioState.settingsSnapshot?.deviceId) {
          await restoreSettingsSnapshot({ includeOutputDevice: false });
        }
      } else {
        // Switching from Live to Preview
        setShowPlaybackPausedMessage(false);

        // Save current live config before stopping
        saveSettingsSnapshot({ includeLiveSettings: true });

        // Stop live input and disconnect the source
        stopLiveInput();

        // Ensure playback is stopped when switching to preview
        setPlaying(false);
      }

      setSourceMode(newMode);
    },
    [sourceMode, audioState.isPlaying, audioState.settingsSnapshot, setPlaying, stopLiveInput, saveSettingsSnapshot, restoreSettingsSnapshot]
  );

  // Source mode options for segmented control
  const sourceModeOptions = React.useMemo(
    () => [
      { value: 'preview' as const, label: 'Preview' },
      { value: 'live' as const, label: 'Live' },
    ],
    []
  );

  // Derived state for live input
  const liveInputMode = audioState.inputMode.type === 'live' ? audioState.inputMode : null;
  const isLiveConnected = liveInputMode !== null;
  const currentDeviceId = liveInputMode?.deviceId ?? null;

  // Handle device change
  const handleLiveDeviceChange = useCallback(async (deviceId: string) => {
    if (deviceId !== currentDeviceId) {
      await startLiveInput(deviceId);
    }
  }, [currentDeviceId, startLiveInput]);

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
          className={`p-0 focus:outline-none ${sourceMode === 'live' && !isLiveConnected ? 'opacity-50 cursor-not-allowed' : ''}`}
          disabled={sourceMode === 'live' && !isLiveConnected}
          aria-label={audioState.isPlaying ? 'Pause' : 'Play'}
        >
          {audioState.isPlaying ? (
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
              isChecked={!audioState.isBypassed}
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
              onClick={() => setIsSettingsDialogOpen(true)}
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
            {showOutputFallbackMessage && (
              <span className='text-xs text-zinc-400 animate-pulse'>
                Output switched to default
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
                defaultOption={selectedInput.url}
              />
            )}
            
            {/* Live mode: not connected - show setup button */}
            {sourceMode === 'live' && !isLiveConnected && (
              <div className='inline-flex flex-1 flex-col gap-1 w-full'>
                <span className='text-sm text-zinc-400'>Live Input</span>
                <div className='relative flex-1'>
                  <button
                    onClick={() => setIsSettingsDialogOpen(true)}
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
            {sourceMode === 'live' && isLiveConnected && (
              <div className='w-full'>
                {audioState.isLiveConnecting ? (
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
                    defaultOption={currentDeviceId ?? ''}
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
      {sourceMode === 'live' && isLiveConnected && (
        <InputControlStrip />
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

      {/* Settings Dialog */}
      <SettingsDialog
        isOpen={isSettingsDialogOpen}
        onClose={() => setIsSettingsDialogOpen(false)}
        sourceMode={sourceMode}
        onConnect={(deviceId, channel) => {
          // Connection is handled by the dialog (via startLiveInput)
          // The dialog's Monitor Input checkbox controls isPlaying state
          console.log('Live input connected:', { deviceId, channel });
        }}
      />
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
