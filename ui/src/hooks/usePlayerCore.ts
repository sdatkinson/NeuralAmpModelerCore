import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useT3kPlayerContext } from '../context/T3kPlayerContext';
import { Input, IR, Model, SourceMode } from '../types';
import { initVisualizer, setupVisualizer } from '../utils/visualizer';
import { formatTime, getDefault } from '../utils/player';
import { useSourceMode } from './useSourceMode';
import { useToast } from './useToast';

// ---------- Options interface ----------

interface UsePlayerCoreOptions {
  id?: string;
  previewMode?: string;
  disabled?: boolean;

  // Data arrays — Player passes non-null arrays, AcordianPlayer passes nullable
  models: Model[] | null;
  irs: IR[] | null;
  inputs: Input[] | null;

  // For AcordianPlayer: lazy data fetch that returns resolved selections
  resolveData?: () => Promise<{ model: Model; ir: IR; input: Input }>;

  // Callbacks
  onPlay?: (data: { model: Model; ir: IR; input: Input }) => void;
  onModelChange?: (model: Model) => void;
  onInputChange?: (input: Input) => void;
  onIrChange?: (ir: IR) => void;
}

// ---------- Return interface ----------

export interface UsePlayerCoreReturn {
  // State
  selectedModel: Model | null;
  selectedInput: Input | null;
  selectedIr: IR | null;
  currentTime: number;
  duration: number;
  isLoading: boolean;
  bypassed: boolean;
  isThisPlayerActive: boolean;

  // Source mode
  sourceMode: SourceMode;
  showPlaybackPausedMessage: boolean;
  toastMessage: string | null;

  // Derived
  isLiveConfigured: boolean;
  currentDeviceId: string | null;
  bypassedStyles: string;
  modelOptions: Array<{ label: string; value: string }>;
  audioOptions: Array<{ label: string; value: string }>;
  irOptions: Array<{ label: string; value: string }>;
  liveDeviceOptions: Array<{ label: string; value: string }>;

  // Handlers
  togglePlay: () => Promise<void>;
  handleSkipToStart: () => void;
  handleBypassToggle: () => void;
  handleModelChange: (value: string | number) => Promise<void>;
  handleInputChange: (value: string | number) => Promise<void>;
  handleIrChange: (value: string | number) => Promise<void>;
  handleSourceModeChange: (mode: SourceMode) => Promise<void>;
  handleLiveDeviceChange: (deviceId: string) => Promise<void>;
  openSettingsDialog: () => void | Promise<void>;

  // Refs
  visualizerRef: React.RefObject<HTMLCanvasElement>;
  canvasWrapperRef: React.RefObject<HTMLDivElement>;

  // Context passthrough
  audioInputDevices: ReturnType<typeof useT3kPlayerContext>['audioInputDevices'];
  inputModeType: string;
}

// ---------- Hook ----------

export function usePlayerCore(options: UsePlayerCoreOptions): UsePlayerCoreReturn {
  const {
    id,
    disabled = false,
    models,
    irs,
    inputs,
    resolveData,
    onPlay,
    onModelChange,
    onInputChange,
    onIrChange,
  } = options;

  // --- Context ---
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
    reconnectLiveInput,
    stopLiveInput,
    setPlaying,
    openSettingsDialog: openDialog,
  } = useT3kPlayerContext();

  // --- Source mode hook (per-player) ---
  const {
    sourceMode,
    showPlaybackPausedMessage,
    liveDeviceOptions,
    handleSourceModeChange,
    handleLiveDeviceChange,
  } = useSourceMode({ playerId: id });

  // Toast messages
  const toastMessage = useToast();

  // --- Derived from context ---
  const isLiveConfigured = audioState.liveInputConfig !== null;
  const currentDeviceId = audioState.liveInputConfig?.deviceId ?? null;
  const isThisPlayerActive = audioState.activePlayerId === id;

  // --- Selection state ---
  const [selectedModel, setSelectedModel] = useState<Model | null>(
    () => (models ? getDefault(models) : null)
  );
  const [selectedInput, setSelectedInput] = useState<Input | null>(
    () => (inputs ? getDefault(inputs) : null)
  );
  const [selectedIr, setSelectedIr] = useState<IR | null>(
    () => (irs ? getDefault(irs) : null)
  );

  // Initialize selections when data arrives (null → array transition for AcordianPlayer)
  useEffect(() => {
    if (models && !selectedModel) setSelectedModel(getDefault(models));
  }, [models, selectedModel]);
  useEffect(() => {
    if (inputs && !selectedInput) setSelectedInput(getDefault(inputs));
  }, [inputs, selectedInput]);
  useEffect(() => {
    if (irs && !selectedIr) setSelectedIr(getDefault(irs));
  }, [irs, selectedIr]);

  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [isLoading, setIsLoading] = useState(false);
  const [bypassed, setBypassed] = useState(false);

  // Sync local bypass state from context when this player is active
  useEffect(() => {
    if (isThisPlayerActive) {
      setBypassed(audioState.isBypassed);
    }
  }, [isThisPlayerActive, audioState.isBypassed]);

  // --- Refs ---
  const visualizerRef = useRef<HTMLCanvasElement>(null);
  const visualizerNodeRef = useRef<AnalyserNode | null>(null);
  const canvasWrapperRef = useRef<HTMLDivElement>(null);

  // --- Memoized options ---
  const modelOptions = useMemo(
    () => (models ?? []).map(m => ({ label: m.name, value: m.url })),
    [models]
  );
  const audioOptions = useMemo(
    () => (inputs ?? []).map(i => ({ label: i.name, value: i.url })),
    [inputs]
  );
  const irOptions = useMemo(
    () => (irs ?? []).map(ir => ({ label: ir.name, value: ir.url })),
    [irs]
  );

  // --- Helpers ---
  const ensureSelections = useCallback(async (): Promise<{
    model: Model; ir: IR; input: Input;
  }> => {
    if (selectedModel && selectedIr && selectedInput) {
      return { model: selectedModel, ir: selectedIr, input: selectedInput };
    }
    if (resolveData) {
      const data = await resolveData();
      setSelectedModel(data.model);
      setSelectedIr(data.ir);
      setSelectedInput(data.input);
      return data;
    }
    throw new Error('No data available and no resolveData provided');
  }, [selectedModel, selectedIr, selectedInput, resolveData]);

  const syncPlayerToEngine = useCallback(async (model: Model, ir: IR, bypassState: boolean) => {
    await syncEngineSettings({
      modelUrl: model.url,
      ir: { url: ir.url, mix: ir.mix, gain: ir.gain },
      bypassed: bypassState,
    });
  }, [syncEngineSettings]);

  // --- Effects ---

  // Cleanup on unmount
  useEffect(() => {
    return () => { cleanup(); };
  }, [cleanup]);

  // Resize handler
  const handleResize = useCallback(() => {
    if (canvasWrapperRef.current && visualizerRef.current) {
      visualizerRef.current.width = canvasWrapperRef.current.clientWidth;
      initVisualizer({ canvas: visualizerRef.current });
    }
  }, []);

  useEffect(() => {
    handleResize();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [handleResize]);

  // Audio element event listeners — only when THIS player is active
  useEffect(() => {
    if (!isThisPlayerActive) return;
    const audioElement = getAudioNodes().audioElement;
    if (!audioElement) return;

    const handleTimeUpdate = () => setCurrentTime(audioElement.currentTime);
    const handleEnded = () => setPlaying(false);
    const handleLoadedMetadata = () => setDuration(audioElement.duration);

    if (audioElement.duration) setDuration(audioElement.duration);

    audioElement.addEventListener('timeupdate', handleTimeUpdate);
    audioElement.addEventListener('ended', handleEnded);
    audioElement.addEventListener('loadedmetadata', handleLoadedMetadata);

    return () => {
      audioElement.removeEventListener('timeupdate', handleTimeUpdate);
      audioElement.removeEventListener('ended', handleEnded);
      audioElement.removeEventListener('loadedmetadata', handleLoadedMetadata);
    };
  }, [getAudioNodes, isThisPlayerActive, setPlaying]);

  // Visualizer setup — only when THIS player is active
  useEffect(() => {
    if (!isThisPlayerActive) return;
    if (audioState.initState !== 'ready' || !visualizerRef.current) return;

    const audioContext = getAudioNodes().audioContext;
    if (!audioContext) return;

    const { analyser, stop } = setupVisualizer(visualizerRef.current, audioContext);
    visualizerNodeRef.current = analyser;
    const disconnect = connectVisualizerNode(analyser);

    return () => {
      stop();
      disconnect();
      visualizerNodeRef.current = null;
    };
  }, [audioState.initState, getAudioNodes, connectVisualizerNode, isThisPlayerActive]);

  // --- Event handlers ---

  const openSettingsDialog = useCallback(async () => {
    const { model, ir } = await ensureSelections();
    openDialog({
      sourceMode,
      playerId: id,
      selectedModel: model,
      selectedIr: ir,
    });
  }, [ensureSelections, openDialog, sourceMode, id]);

  const togglePlay = useCallback(async () => {
    if (!id) return;
    const wantsLiveInput = sourceMode === 'live';
    const isActive = audioState.activePlayerId === id;

    // === LIVE MODE ===
    if (wantsLiveInput) {
      const { model, ir } = await ensureSelections();
      const { audioContext } = getAudioNodes();

      await reconnectLiveInput();

      if (audioContext && audioContext.state === 'suspended') {
        await audioContext.resume();
      }

      await syncPlayerToEngine(model, ir, bypassed);

      if (isActive) {
        setPlaying(false);
      } else {
        setPlaying(true, id);
      }
      return;
    }

    // === PREVIEW MODE ===
    const { mediaStream } = getAudioNodes();
    if (mediaStream?.active) {
      stopLiveInput();
    }

    setIsLoading(true);

    try {
      const { model, ir, input } = await ensureSelections();

      if (audioState.initState !== 'ready') {
        await init({ audioUrl: input.url });
      }

      if (isActive) {
        setPlaying(false);
      } else {
        if (audioState.initState === 'ready') {
          const { audioContext } = getAudioNodes();
          if (audioContext) await audioContext.resume();
        }

        if (!audioState.audioUrl || audioState.audioUrl !== input.url) {
          await loadAudio(input.url);
        }

        await syncPlayerToEngine(model, ir, bypassed);
        setPlaying(true, id);
        onPlay?.({ model, ir, input });
      }
    } catch (error) {
      console.error('Error in togglePlay:', error);
      setPlaying(false);
    } finally {
      setIsLoading(false);
    }
  }, [
    id,
    audioState.activePlayerId,
    audioState.initState,
    audioState.audioUrl,
    sourceMode,
    bypassed,
    ensureSelections,
    getAudioNodes,
    init,
    loadAudio,
    reconnectLiveInput,
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
    if (isThisPlayerActive) {
      setBypass(newBypassed);
    }
  }, [setBypass, bypassed, isThisPlayerActive]);

  const handleModelChange = useCallback(
    async (value: string | number) => {
      const model = models?.find(m => m.url === String(value));
      if (model) {
        setSelectedModel(model);
        try {
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
      const wasThisPlayerPlaying = audioState.activePlayerId === id;
      const input = inputs?.find(i => i.url === String(value));
      if (!input) return;

      setSelectedInput(input);

      try {
        if (
          audioState.initState === 'ready' &&
          (!audioState.audioUrl || audioState.audioUrl !== input.url)
        ) {
          await loadAudio(input.url);
        }

        if (wasThisPlayerPlaying && id) {
          setPlaying(true, id);
        }

        onInputChange?.(input);
      } catch (error) {
        console.error('Error loading audio:', error);
      }
    },
    [id, audioState.activePlayerId, audioState.audioUrl, audioState.initState, inputs, loadAudio, onInputChange, setPlaying]
  );

  const handleIrChange = useCallback(
    async (value: string | number) => {
      const ir = irs?.find(i => i.url === String(value));
      if (!ir) return;

      setSelectedIr(ir);

      try {
        if (isThisPlayerActive) {
          if (audioState.initState === 'ready' && ir.url) {
            await loadIr({ url: ir.url, wetAmount: ir.mix, gainAmount: ir.gain });
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

  const bypassedStyles = bypassed
    ? 'opacity-50 touch-none cursor-not-allowed grayscale'
    : '';

  return {
    selectedModel,
    selectedInput,
    selectedIr,
    currentTime,
    duration,
    isLoading,
    bypassed,
    isThisPlayerActive,

    sourceMode,
    showPlaybackPausedMessage,
    toastMessage,

    isLiveConfigured,
    currentDeviceId,
    bypassedStyles,
    modelOptions,
    audioOptions,
    irOptions,
    liveDeviceOptions,

    togglePlay,
    handleSkipToStart,
    handleBypassToggle,
    handleModelChange,
    handleInputChange,
    handleIrChange,
    handleSourceModeChange,
    handleLiveDeviceChange,
    openSettingsDialog,

    visualizerRef,
    canvasWrapperRef,

    audioInputDevices,
    inputModeType: audioState.inputMode.type,
  };
}

export { formatTime };
