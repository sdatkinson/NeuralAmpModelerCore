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
import { LiveSetupDialog } from '../LiveSetupDialog';
import { InputControlStrip } from '../ControlStrip';
import { Plug, Settings } from 'lucide-react';

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
  
  // Source mode state (preview = file playback, live = direct input)
  const [sourceMode, setSourceMode] = useState<SourceMode>('preview');
  const [showPlaybackPausedMessage, setShowPlaybackPausedMessage] = useState(false);
  const [isLiveSetupModalOpen, setIsLiveSetupModalOpen] = useState(false);

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

  // Helper to mute/unmute live output
  const setLiveOutputMuted = useCallback((muted: boolean) => {
    const nodes = getAudioNodes();
    if (nodes.outputGainNode && nodes.audioContext) {
      nodes.outputGainNode.gain.setTargetAtTime(
        muted ? 0 : 1,
        nodes.audioContext.currentTime,
        0.01 // 10ms for quick response
      );
    }
  }, [getAudioNodes]);

  // Event handlers
  const togglePlay = useCallback(async () => {
    const nodes = getAudioNodes();
    const { audioElement, audioContext } = nodes;
    const isLiveMode = sourceMode === 'live' && audioState.inputMode.type === 'live';

    // Live mode: toggle output mute instead of file playback
    if (isLiveMode) {
      if (isPlaying) {
        // Mute output
        setLiveOutputMuted(true);
        setIsPlaying(false);
      } else {
        // Ensure audio context is running
        if (audioContext) {
          await audioContext.resume();
        }
        // Unmute output
        setLiveOutputMuted(false);
        setIsPlaying(true);
      }
      return;
    }

    // Preview mode: standard file playback
    setIsLoading(true);
    if (audioState.initState !== 'ready') await init({ audioUrl: selectedInput.url });

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
    sourceMode,
    selectedInput,
    selectedModel,
    selectedIr,
    loadAudio,
    loadModel,
    loadIr,
    removeIr,
    onPlay,
    setLiveOutputMuted,
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
      const wasPlaying = isPlaying;
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
      audioState.initState,
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
    (newMode: SourceMode) => {
      if (newMode === sourceMode) return;

      if (newMode === 'live') {
        // Switching from Preview to Live
        if (isPlaying) {
          // Pause playback but preserve playhead position
          const audioElement = getAudioNodes().audioElement;
          if (audioElement) {
            audioElement.pause();
          }
          setIsPlaying(false);
          setShowPlaybackPausedMessage(true);

          // Auto-hide the message after 3 seconds
          setTimeout(() => {
            setShowPlaybackPausedMessage(false);
          }, 3000);
        }
        // Ensure live output starts muted (user must click play)
        setLiveOutputMuted(true);
      } else {
        // Switching from Live to Preview
        setShowPlaybackPausedMessage(false);
        setIsPlaying(false);
        // Stop live input and disconnect the source
        stopLiveInput();
        // Restore output gain for file playback
        const nodes = getAudioNodes();
        if (nodes.outputGainNode && nodes.audioContext) {
          nodes.outputGainNode.gain.setTargetAtTime(1, nodes.audioContext.currentTime, 0.01);
        }
      }

      setSourceMode(newMode);
    },
    [sourceMode, isPlaying, getAudioNodes, setLiveOutputMuted, stopLiveInput]
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

        <div className='hidden sm:flex text-sm font-mono gap-2 text-zinc-400'>
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

        {/* Source mode segmented control - its own row */}
        <div className='flex flex-col gap-1 pt-2'>
          <span className='text-sm text-zinc-400'>Source</span>
          <div className='flex items-center gap-3'>
            <SegmentedControl
              options={sourceModeOptions}
              value={sourceMode}
              onChange={handleSourceModeChange}
            />
            {showPlaybackPausedMessage && (
              <span className='text-xs text-zinc-400 animate-pulse'>
                Playback paused
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
                    onClick={() => setIsLiveSetupModalOpen(true)}
                    className='flex items-center gap-2 w-full overflow-hidden px-4 py-3 text-md border border-zinc-700 rounded-md bg-transparent hover:bg-zinc-800 transition-colors focus:outline-none'
                  >
                    <Plug size={18} className='text-zinc-400 flex-shrink-0' />
                    <span className='text-ellipsis text-nowrap overflow-hidden min-w-0'>Enable Live Input</span>
                  </button>
                  {/* Helper text - positioned below without affecting layout */}
                  <span className='absolute top-full mt-1 text-xs text-zinc-500'>
                    Use headphones to avoid feedback.
                  </span>
                </div>
              </div>
            )}

            {/* Live mode: connected - show device dropdown and settings gear */}
            {sourceMode === 'live' && isLiveConnected && (
              <div className='flex items-end gap-2 w-full'>
                <div className='flex-1'>
                  <Select
                    options={liveDeviceOptions}
                    label='Live Input'
                    onChange={(value) => handleLiveDeviceChange(String(value))}
                    defaultOption={currentDeviceId ?? ''}
                  />
                </div>
                {/* Settings/Gear Icon */}
                <button
                  onClick={() => setIsLiveSetupModalOpen(true)}
                  className='p-2.5 mb-[1px] hover:bg-zinc-800 rounded-md transition-colors border border-zinc-700'
                  aria-label='Live input settings'
                >
                  <Settings size={18} className='text-zinc-400' />
                </button>
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

      {/* Live Setup Dialog */}
      <LiveSetupDialog
        isOpen={isLiveSetupModalOpen}
        onClose={() => setIsLiveSetupModalOpen(false)}
        onConnect={(deviceId, channel) => {
          // Connection is already handled by the dialog (via startLiveInput)
          // Ensure output starts muted - user must click play to hear signal
          setLiveOutputMuted(true);
          setIsPlaying(false);
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
