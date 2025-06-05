import React, { memo, useCallback, useEffect, useRef, useState } from 'react';
import { useT3kPlayerContext } from '../../context/T3kPlayerContext';
import { Model, T3kPlayerProps } from '../../types';
import { initVisualizer, setupVisualizer } from '../../utils/visualizer';
import { Play } from '../ui/Play';
import { Skip } from '../ui/Skip';
import { Select } from '../ui/Select';
import { ToggleSimple } from '../ui/ToggleSimple';
import { Pause } from '../ui/Pause';

const PlayerFC: React.FC<T3kPlayerProps> = ({ models, irs, inputs }) => {
  const {
    audioElement,
    audioContext,
    outputGainNode,
    audioWorkletNode,
    loadProfile,
    setAudioSource,
    toggleBypass,
    isProfileLoaded,
    // cleanup,
    resetProfile,
    loadIr,
    removeIr,
    isIrLoaded,
  } = useT3kPlayerContext();

  const [model, setModel] = useState<Model>(models[0]);
  const modelUrl = model.model_url;
  const modelOptions = models.map((model) => ({
    label: model.name,
    value: model.model_url,
  }));

  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [bypassed, setBypassed] = useState(false);
  const visualizerRef = useRef<HTMLCanvasElement>(null);
  const [audioSrc, setAudioSrc] = useState<string>(inputs[0].input_url);
  const audioOptions = inputs.map((input) => ({
    label: input.name,
    value: input.input_url,
  }));
  const [ir, setIr] = useState<string | undefined>(irs[0].ir_url);
  const irOptions = irs.map((ir) => ({
    label: ir.name,
    value: ir.ir_url,
  }));

  // Add ref to track visualizer node
  const visualizerNodeRef = useRef<AnalyserNode | null>(null);
  const canvasWrapperRef = useRef<HTMLDivElement>(null);

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

  useEffect(() => {
    if (audioElement && audioContext && outputGainNode && visualizerRef.current) {
      // Disconnect old visualizer if it exists
      if (visualizerNodeRef.current) {
        visualizerNodeRef.current.disconnect();
      }

      // Create and store new visualizer
      const visualizer = setupVisualizer(visualizerRef.current, audioContext);
      visualizerNodeRef.current = visualizer;

      // Connect output to visualizer
      outputGainNode.connect(visualizer);

      audioElement.addEventListener('timeupdate', () => {
        setCurrentTime(audioElement.currentTime);
      });

      audioElement.addEventListener('ended', () => {
        setIsPlaying(false);
      });

      // Cleanup function
      return () => {
        if (visualizerNodeRef.current) {
          visualizerNodeRef.current.disconnect();
        }
      };
    }
  }, [audioElement, audioContext, outputGainNode]);

  useEffect(() => {
    if (audioElement) {
      const handleLoadedMetadata = () => {
        setDuration(audioElement.duration);
      };

      // Set initial duration if already loaded
      if (audioElement.duration) {
        setDuration(audioElement.duration);
      }

      audioElement.addEventListener('loadedmetadata', handleLoadedMetadata);

      return () => {
        audioElement.removeEventListener('loadedmetadata', handleLoadedMetadata);
      };
    }
  }, [audioElement, audioSrc]);

  useEffect(() => {
    resetProfile();
    setAudioSource(audioSrc);
  }, []);

  const togglePlay = async () => {
    if (audioElement) {
      try {
        if (isPlaying) {
          audioElement.pause();
        } else {
          // Ensure audio context is running (especially important for iOS)
          if (audioContext) {
            await audioContext.resume();
          }

          if (!isProfileLoaded) {
            await loadProfile(modelUrl);
          }

          await audioElement.play();
        }
        setIsPlaying(!isPlaying);
      } catch (error) {
        console.error('Error in togglePlay:', error);
      }
    }
  };

  useEffect(() => {
    if (ir && !isIrLoaded && audioContext && outputGainNode && audioWorkletNode) {
      const irSample = irs.find((sample) => sample.ir_url === ir);
      loadIr(ir, irSample?.mix, irSample?.gain);
    }
  }, [ir, isIrLoaded, audioContext, outputGainNode, audioWorkletNode, loadIr]);

  const handleBypassToggle = () => {
    toggleBypass();
    setBypassed(!bypassed);
  };

  const formatTime = (time: number) => {
    const minutes = Math.floor(time / 60);
    const seconds = Math.floor(time % 60);
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
  };

  const handleInputChange = async (value: string | number) => {
    const newSrc = String(value);
    const wasPlaying = isPlaying;

    setAudioSrc(newSrc);
    setAudioSource(newSrc);

    if (wasPlaying && audioElement) {
      audioElement.addEventListener(
        'loadeddata',
        () => {
          audioElement.play();
        },
        { once: true }
      );
    }
  };

  const handleIrChange = async (value: string | number) => {
    const newSrc = String(value);
    if (newSrc !== '') {
      setIr(newSrc);
      const irSample = irs.find((sample) => sample.ir_url === newSrc);
      await loadIr(newSrc, irSample?.mix, irSample?.gain);
    } else {
      setIr(undefined);
      removeIr();
    }
  };

  const handleModelChange = async (value: string | number) => {
    const model = models.find((model) => model.model_url === String(value));
    if (model) {
      setModel(model);
      await loadProfile(model.model_url!);
    }
  };

  return (
    <div className="bg-zinc-900 border border-zinc-700 text-white p-4 lg:p-8 pt-0 lg:pt-2 rounded-xl w-full">
      <div className="flex items-center gap-4 overflow-hidden">
        <button
          onClick={togglePlay}
          className="p-0 focus:outline-none"
          aria-label={isPlaying ? 'Pause' : 'Play'}
        >
          {isPlaying ? <Pause /> : <Play />}
        </button>
        <button
          onClick={() => {
            if (audioElement) {
              audioElement.currentTime = 0;
              setCurrentTime(0);
            }
          }}
          className="p-0 focus:outline-none"
          aria-label="Skip to start"
        >
          <Skip opacity={currentTime > 0 ? 1 : 0.6} />
        </button>
        <div className="flex text-sm font-mono gap-2 text-zinc-400">
          <span>{formatTime(currentTime)}</span>
          <span> / </span>
          <span>{formatTime(duration)}</span>
        </div>
        <div ref={canvasWrapperRef} className="flex-1">
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
      <div className="flex flex-col gap-2">
        <div className={'flex flex-row items-center gap-4 flex-wrap'}>
          <div className={'flex-1 min-w-[0px]'}>
            <div
              className={`flex w-full flex-1 min-w-[0px] ${bypassed ? 'opacity-50 touch-none cursor-not-allowed grayscale' : ''}`}
            >
              <Select
                  options={modelOptions}
                  label={'Model'}
                  onChange={handleModelChange}
                  defaultOption={model.model_url!}
                  disabled={bypassed}
                />
            </div>
          </div>
          <div className={'flex items-center pt-[24px] flex-shrink-0'}>
            <ToggleSimple
              label={''}
              onChange={handleBypassToggle}
              isChecked={!bypassed}
              ariaLabel="Bypass"
            />
          </div>
        </div>
        <div className={'flex flex-col sm:flex-row items-center gap-2 sm:gap-6'}>
          <div className={'w-full sm:w-1/2'}>
            <Select
              options={audioOptions}
              label={'Input'}
              onChange={handleInputChange}
              defaultOption={audioSrc}
            />
          </div>
          <div
            className={`w-full sm:w-1/2 ${bypassed ? 'opacity-50 touch-none cursor-not-allowed grayscale' : ''}`}
          >
            <Select
                options={irOptions}
                label={'IR'}
                onChange={handleIrChange}
                defaultOption={ir}
                disabled={bypassed}
              />
          </div>
        </div>
      </div>
    </div>
  );
};

const Player = memo(
  ({ models, irs, inputs }: T3kPlayerProps) => {
    return (
      <PlayerFC
        models={models}
        irs={irs}
        inputs={inputs}
      />
    );
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
