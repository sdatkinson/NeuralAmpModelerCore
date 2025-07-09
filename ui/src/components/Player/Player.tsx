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

const PlayerFC: React.FC<T3kPlayerProps> = ({
  models = DEFAULT_MODELS,
  irs = DEFAULT_IRS,
  inputs = DEFAULT_INPUTS,
  previewMode,
  onPlay,
  onModelChange,
  onInputChange,
  onIrChange,
}) => {
  const {
    audioElement,
    audioContext,
    outputGainNode,
    audioWorkletNode,
    loadProfile,
    setAudioSource,
    toggleBypass,
    isProfileLoaded,
    resetProfile,
    loadIr,
    removeIr,
    isIrLoaded,
    cleanup,
  } = useT3kPlayerContext();

  // Helper functions to get defaults
  const getDefaultModel = () => {
    const defaultModel = models.find(m => m.default);
    return defaultModel || models[0];
  };

  const getDefaultInput = () => {
    const defaultInput = inputs.find(i => i.default);
    return defaultInput ? defaultInput : inputs[0];
  };

  const getDefaultIr = () => {
    const defaultIr = irs.find(i => i.default);
    return defaultIr ? defaultIr : irs[0];
  };

  // State
  const [selectedModel, setSelectedModel] = useState<Model>(getDefaultModel());
  const [selectedInput, setSelectedInput] = useState<Input>(getDefaultInput());
  const [selectedIr, setSelectedIr] = useState<IR>(getDefaultIr());
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [bypassed, setBypassed] = useState(false);

  // Refs
  const visualizerRef = useRef<HTMLCanvasElement>(null);
  const visualizerNodeRef = useRef<AnalyserNode | null>(null);
  const canvasWrapperRef = useRef<HTMLDivElement>(null);

  // Memoized options
  const modelOptions = models.map(model => ({
    label: model.name,
    value: model.url,
  }));

  const audioOptions = inputs.map(input => ({
    label: input.name,
    value: input.url,
  }));

  const irOptions = irs.map(ir => ({
    label: ir.name,
    value: ir.url,
  }));

  // Utility functions
  const formatTime = (time: number) => {
    const minutes = Math.floor(time / 60);
    const seconds = Math.floor(time % 60);
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
  };

  // Visualizer setup
  const handleResize = useCallback(() => {
    if (canvasWrapperRef.current && visualizerRef.current) {
      const wrapperWidth = canvasWrapperRef.current.clientWidth;
      visualizerRef.current.width = wrapperWidth;
      if (visualizerRef.current) {
        initVisualizer({ canvas: visualizerRef.current });
      }
    }
  }, []);

  useEffect(() => {
    handleResize();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [handleResize]);

  // Audio element event listeners
  useEffect(() => {
    if (!audioElement) return;

    const handleTimeUpdate = () => setCurrentTime(audioElement.currentTime);
    const handleEnded = () => {
      setIsPlaying(false);
    };
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
  }, [audioElement, selectedInput, onPlay, selectedModel, selectedIr]);

  // Visualizer setup
  useEffect(() => {
    if (
      audioElement &&
      audioContext &&
      outputGainNode &&
      visualizerRef.current
    ) {
      // Disconnect old visualizer if it exists
      if (visualizerNodeRef.current) {
        visualizerNodeRef.current.disconnect();
      }

      // Create and store new visualizer
      const visualizer = setupVisualizer(visualizerRef.current, audioContext);
      visualizerNodeRef.current = visualizer;

      // Connect output to visualizer
      outputGainNode.connect(visualizer);

      // Cleanup function
      return () => {
        if (visualizerNodeRef.current) {
          visualizerNodeRef.current.disconnect();
        }
      };
    }
  }, [audioElement, audioContext, outputGainNode]);

  // Initialize
  useEffect(() => {
    resetProfile();
  }, []);

  // Initialize audio source
  useEffect(() => {
    setAudioSource(selectedInput.url);
    return () => {
      cleanup();
      setIsPlaying(false);
    };

    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [!!audioElement]);

  // IR loading
  useEffect(() => {
    if (
      selectedIr.url &&
      !isIrLoaded &&
      audioContext &&
      outputGainNode &&
      audioWorkletNode
    ) {
      loadIr(selectedIr.url, selectedIr.mix, selectedIr.gain);
    }
  }, [
    selectedIr,
    isIrLoaded,
    audioContext,
    outputGainNode,
    audioWorkletNode,
    loadIr,
  ]);

  // Event handlers
  const togglePlay = async () => {
    if (!audioElement) return;

    try {
      if (isPlaying) {
        audioElement.pause();
        setIsPlaying(false);
      } else {
        // Ensure audio context is running (especially important for iOS)
        if (audioContext) {
          await audioContext.resume();
        }

        if (!isProfileLoaded) {
          await loadProfile(selectedModel.url);
        }

        await audioElement.play();
        setIsPlaying(true);
        onPlay?.({
          model: selectedModel,
          ir: selectedIr,
          input: selectedInput,
        });
      }
    } catch (error) {
      console.error('Error in togglePlay:', error);
    }
  };

  const handleSkipToStart = () => {
    if (audioElement) {
      audioElement.currentTime = 0;
      setCurrentTime(0);
    }
  };

  const handleBypassToggle = () => {
    toggleBypass();
    setBypassed(!bypassed);
  };

  const handleModelChange = async (value: string | number) => {
    const model = models.find(model => model.url === String(value));
    if (model) {
      setSelectedModel(model);
      await loadProfile(model.url!);
      onModelChange?.(model);
    }
  };

  const handleInputChange = async (value: string | number) => {
    const wasPlaying = isPlaying;

    const selectedInput = inputs.find(input => input.url === String(value))!;
    setSelectedInput(selectedInput);
    setAudioSource(selectedInput.url);

    if (wasPlaying && audioElement) {
      audioElement.addEventListener(
        'loadeddata',
        () => {
          audioElement.play();
        },
        { once: true }
      );
    }
    onInputChange?.(selectedInput);
  };

  const handleIrChange = async (value: string | number) => {
    const selectedIr = irs.find(ir => ir.url === String(value))!;
    setSelectedIr(selectedIr);
    if (selectedIr.url) {
      await loadIr(selectedIr.url, selectedIr.mix, selectedIr.gain);
    } else {
      removeIr();
    }
    onIrChange?.(selectedIr);
  };

  const bypassedStyles = bypassed
    ? 'opacity-50 touch-none cursor-not-allowed grayscale'
    : '';

  const renderModelSelect = () => (
    <Select
      options={modelOptions}
      label='Model'
      onChange={handleModelChange}
      defaultOption={selectedModel.url!}
      disabled={bypassed}
    />
  );

  const renderIrSelect = () => (
    <Select
      options={irOptions}
      label='IR'
      onChange={handleIrChange}
      defaultOption={selectedIr.url}
      disabled={bypassed}
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
          {isPlaying ? <Pause /> : <Play />}
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
              isChecked={!bypassed}
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
              disabled={bypassed}
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
