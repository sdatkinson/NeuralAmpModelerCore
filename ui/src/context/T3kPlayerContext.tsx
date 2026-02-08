import React, {
  createContext,
  useContext,
  useRef,
  useState,
  useEffect,
  ReactNode,
  useCallback,
  useMemo,
} from 'react';
import { useModule } from '../hooks/useModule';
import { readModel } from '../utils/readModel';
import { DEFAULT_AUDIO_SRC, DEFAULT_MODELS } from '../constants';
import { InputMode, AudioInputDevice, MicrophonePermissionState, AudioInputDeviceState, AudioOutputDeviceState, ChannelSelection, SettingsSnapshot, SettingsDialogState, SourceMode, Model, IR, AudioNodes, AudioState, IrConfig, EMPTY_AUDIO_NODES } from '../types';
import { dbToLinear } from '../utils/metering';
import { cleanupLiveInputNodes, cleanupOutputWorkaroundRouting, teardownLiveInput, applyOutputDeviceRouting } from '../utils/audioNodes';
import { useAudioDevicesAndPermissions } from '../hooks/useAudioDevicesAndPermissions';

/**
 * Audio Initialization Flow
 * ========================
 *
 * The WASM-based audio system has a specific initialization sequence:
 *
 * 1. init() loads the WASM script and sets up the wasmAudioWorkletCreated callback
 * 2. The WASM only creates AudioContext/AudioWorkletNode when setDsp is called (via loadModel)
 * 3. When WASM creates the audio nodes, it calls our callback with the node references
 * 4. waitForAudioReady() resolves when the callback has executed
 *
 * The init() function handles this entire sequence internally:
 * - Loads the WASM script
 * - Loads a default model to trigger audio initialization
 * - Waits for the callback to execute
 * - Returns only when audio is fully ready
 *
 * After await init(), all audio nodes are guaranteed to exist.
 */

interface T3kPlayerContextType {
  // State
  audioState: AudioState;
  microphonePermission: MicrophonePermissionState;
  audioInputDevices: AudioInputDeviceState;
  audioOutputDevices: AudioOutputDeviceState;

  // Getters
  getAudioNodes: () => AudioNodes;
  isAudioReady: () => boolean;

  // Actions
  init: (options?: { audioUrl?: string; modelUrl?: string }) => Promise<void>;
  loadModel: (modelUrl: string) => Promise<void>;
  loadAudio: (src: string) => Promise<void>;
  loadIr: (config: IrConfig) => Promise<void>;
  removeIr: () => void;
  setBypass: (bypassed: boolean) => void;
  cleanup: () => void;
  connectVisualizerNode: (analyserNode: AnalyserNode) => () => void;

  // Engine sync
  syncEngineSettings: (config: {
    modelUrl: string;
    ir: { url: string | null; mix?: number; gain?: number };
    bypassed: boolean;
  }) => Promise<void>;

  // Input mode actions
  startLiveInput: (deviceId?: string, options?: { initialChannel?: ChannelSelection; initialChannelGains?: Record<ChannelSelection, number> }) => Promise<void>;
  reconnectLiveInput: () => Promise<void>;
  stopLiveInput: () => void;
  selectLiveInputChannel: (channel: ChannelSelection) => void;
  setLiveInputGain: (gainDb: number) => void;

  // Playback control
  setPlaying: {
    (playing: true, playerId: string): void;
    (playing: false): void;
  };

  // Microphone permission actions
  requestMicrophonePermission: () => Promise<string | null>;  // Returns preferred device ID

  // Audio input device actions
  refreshAudioDevices: () => Promise<{ inputDevices: AudioInputDevice[]; preferredDeviceId: string | null }>;

  // Audio output device actions
  setOutputDevice: (deviceId: string | null) => Promise<void>;
  refreshAudioOutputDevices: () => Promise<void>;

  // Settings dialog (global, single instance)
  settingsDialog: SettingsDialogState;
  openSettingsDialog: (config: { sourceMode: SourceMode; playerId?: string; selectedModel: Model; selectedIr: IR }) => void;
  closeSettingsDialog: (options: { saved: boolean }) => void;
}

// Context
const T3kPlayerContext = createContext<T3kPlayerContextType | null>(null);

// Provider Component
export function T3kPlayerContextProvider({
  children,
}: {
  children: ReactNode;
}) {
  // Consolidated state
  const [audioState, setAudioState] = useState<AudioState>({
    initState: 'uninitialized',
    isPlaying: false,
    activePlayerId: null,
    isBypassed: false,
    modelUrl: null,
    irUrl: null,
    audioUrl: null,
    inputMode: { type: 'preview' },
    liveInputConfig: null,
  });

  // Settings dialog state (global, single instance)
  const [settingsDialog, setSettingsDialog] = useState<SettingsDialogState>({
    isOpen: false,
    sourceMode: 'preview',
    snapshot: null,
    hadExistingConfig: false,
  });
  const settingsSnapshotRef = useRef<{ snapshot: SettingsSnapshot | null; hadExistingConfig: boolean }>({
    snapshot: null,
    hadExistingConfig: false,
  });

  // Refs for non-render-triggering data
  const audioNodesRef = useRef<AudioNodes>({ ...EMPTY_AUDIO_NODES });

  const isInitializingRef = useRef<boolean>(false);
  const isInitializedRef = useRef<boolean>(false);
  const modulePromise = useModule();

  // Getter for audio nodes
  const getAudioNodes = useCallback((): AudioNodes => {
    return audioNodesRef.current;
  }, []);

  // Check if audio system is ready for operations
  // Use this for conditional logic or UI state
  const isAudioReady = useCallback((): boolean => {
    if (!isInitializedRef.current) return false;
    const nodes = audioNodesRef.current;
    return !!(nodes.audioContext && nodes.audioWorkletNode);
  }, []);

  // Audio device management, permissions, and hot-plug detection
  const {
    microphonePermission,
    audioInputDevices,
    audioOutputDevices,
    requestMicrophonePermission,
    refreshAudioDevices,
    refreshAudioOutputDevices,
    setOutputDevice,
    handleLiveInputUnavailable,
  } = useAudioDevicesAndPermissions({
    applyOutputRouting: useCallback(async (deviceId: string | null) => {
      const nodes = audioNodesRef.current;
      if (!nodes.audioContext || !nodes.outputMeterNode) return;
      await applyOutputDeviceRouting(nodes, deviceId);
    }, []),
    teardownLiveInput: useCallback(() => {
      teardownLiveInput(audioNodesRef.current, { muteOutput: false });
    }, []),
    onLiveInputLost: useCallback(() => {
      setAudioState(prev => ({
        ...prev,
        inputMode: { type: 'preview' },
        liveInputConfig: null,
        isPlaying: false,
        activePlayerId: null,
      }));
    }, []),
    liveInputConfig: audioState.liveInputConfig,
  });


  /**
   * Initialize the audio system fully.
   *
   * This handles the complete initialization sequence:
   * 1. Loads the audio element for preview playback
   * 2. Loads the WASM script
   * 3. Loads a model to trigger WASM audio initialization
   * 4. Waits for the WASM callback to create audio nodes
   *
   * After this function resolves, all audio nodes are guaranteed to exist.
   *
   * @param options.audioUrl - URL for preview audio (uses default if not provided)
   * @param options.modelUrl - URL for NAM model (uses default if not provided)
   */
  const init = useCallback(
    async (options?: { audioUrl?: string; modelUrl?: string }): Promise<void> => {
      if (isInitializedRef.current || isInitializingRef.current) return;

      isInitializingRef.current = true;
      setAudioState(prev => ({ ...prev, initState: 'initializing' }));

      try {
        // Check browser support
        if (
          typeof window === 'undefined' ||
          !window.AudioContext ||
          !window.AudioWorklet
        ) {
          throw new Error('AudioWorklet not supported in this browser');
        }

        // Create and setup audio element
        const audio = new Audio();
        audio.crossOrigin = 'anonymous';
        audio.src = options?.audioUrl || DEFAULT_AUDIO_SRC;
        getAudioNodes().audioElement = audio;
        document.body.appendChild(audio);

        await new Promise<void>((resolve, reject) => {
          const handleLoad = () => resolve();
          const handleError = () =>
            reject(new Error('Failed to load default audio'));
          audio.addEventListener('loadeddata', handleLoad, { once: true });
          audio.addEventListener('error', handleError, { once: true });
          audio.load();
        });

        // Safety net: reset playback state when audio file ends naturally.
        // Player.tsx also listens for 'ended', but this catches edge cases where
        // the player's listener might not fire (e.g., component unmounted).
        audio.addEventListener('ended', () => {
          const nodes = getAudioNodes();
          if (nodes.outputGainNode && nodes.audioContext) {
            nodes.outputGainNode.gain.setTargetAtTime(0, nodes.audioContext.currentTime, 0.01);
          }
          setAudioState(prev => ({ ...prev, isPlaying: false, activePlayerId: null }));
        });

        // Promise that resolves when WASM callback fires
        // The callback is set on window because the WASM module expects it there
        const audioReady = new Promise<void>(resolve => {
          // @ts-ignore
          window.wasmAudioWorkletCreated = (
            node1: AudioWorkletNode,
            node2: AudioContext
          ) => {
            const audioWorkletNode = node1;
            const context = node2;
            const nodes = getAudioNodes();

            // Store nodes
            nodes.audioWorkletNode = audioWorkletNode;
            nodes.audioContext = context;

            // Create gain nodes
            nodes.inputGainNode = new GainNode(context, { gain: 1 });
            nodes.outputGainNode = new GainNode(context, { gain: 1 });
            nodes.bypassNode = new GainNode(context, { gain: 0 });

            // Create metering nodes
            const meterConfig = { fftSize: 2048 };
            nodes.inputMeterNode = new AnalyserNode(context, meterConfig);
            nodes.outputMeterNode = new AnalyserNode(context, meterConfig);

            // Create source from audio element
            nodes.sourceNode = context.createMediaElementSource(nodes.audioElement!);

            // Connect audio graph
            nodes.sourceNode.connect(nodes.inputGainNode);
            nodes.inputGainNode.connect(nodes.inputMeterNode!);
            nodes.inputMeterNode!.connect(nodes.bypassNode);
            nodes.bypassNode.connect(nodes.outputGainNode);
            nodes.inputMeterNode!.connect(audioWorkletNode);
            audioWorkletNode.connect(nodes.outputGainNode);
            nodes.outputGainNode.connect(nodes.outputMeterNode!);
            nodes.outputMeterNode!.connect(context.destination);

            context.resume();
            resolve();
          };
        });

        // Load WASM module script (callback is already set up above)
        await new Promise<void>((resolve, reject) => {
          const script = document.createElement('script');
          script.src = '/t3k-wasm-module.js';
          script.async = true;
          script.onload = () => resolve();
          script.onerror = () => reject(new Error('Failed to load audio module'));
          document.body.appendChild(script);
        });

        // Load a model to trigger WASM audio initialization
        // The WASM only creates AudioContext/AudioWorklet when setDsp is called
        const modelUrl = options?.modelUrl ||
          DEFAULT_MODELS.find(m => m.default)?.url ||
          DEFAULT_MODELS[0].url;
        await loadModelInternal(modelUrl);

        // Wait for the WASM callback to execute and create audio nodes
        await audioReady;

        // Now fully initialized
        isInitializedRef.current = true;
        setAudioState(prev => ({
          ...prev,
          initState: 'ready',
        }));
      } catch (error) {
        setAudioState(prev => ({ ...prev, initState: 'uninitialized' }));
        console.error('Error initializing audio system:', error);
        throw error;
      } finally {
        isInitializingRef.current = false;
      }
    },
    [getAudioNodes]
  );

  // Internal model loading function used during init (bypasses isInitializingRef check)
  const loadModelInternal = async (modelUrl: string): Promise<void> => {
    // Fetch and process model file
    const response = await fetch(modelUrl);
    if (!response.ok) {
      throw new Error(`Failed to fetch model: ${response.statusText}`);
    }

    const blob = await response.blob();
    const file = new File([blob], 'profile.nam', { type: '.nam' });
    const jsonStr = (await readModel(file)) as string;

    if (!jsonStr) {
      throw new Error('Failed to read model - empty response');
    }

    if (!modulePromise) {
      throw new Error('WASM module not available');
    }

    const module = await modulePromise();

    if (!module?._malloc || !module?.stringToUTF8 || !module?.ccall) {
      throw new Error('WASM module missing required functions');
    }

    // Allocate memory and load model
    // Use UTF-8 byte length: jsonStr.length counts characters, but stringToUTF8 writes
    // UTF-8 bytes. Special chars (e.g. "ö") need 2+ bytes, so we must size correctly.
    const byteLength = new TextEncoder().encode(jsonStr).length + 1;
    const ptr = module._malloc(byteLength);
    module.stringToUTF8(jsonStr, ptr, byteLength);

    try {
      const context = getAudioNodes().audioContext;

      // Suspend context during model loading (if it exists)
      if (context?.state === 'running') {
        await context.suspend();
      }

      // Load DSP - this triggers WASM to create AudioContext/AudioWorklet
      await module.ccall('setDsp', null, ['number'], [ptr], {
        async: true,
      });
      module._free(ptr);

      // Resume context (if it exists)
      if (context?.state === 'suspended') {
        await context.resume();
      }

      setAudioState(prev => ({ ...prev, modelUrl }));
    } catch (error) {
      module._free(ptr);
      throw error;
    }
  };

  // Load model (public API with guards)
  // Uses refs for guards to avoid stale closure issues when called
  // immediately after init() within the same event handler
  const loadModel = useCallback(
    async (modelUrl: string): Promise<void> => {
      if (isInitializingRef.current) {
        throw new Error('Audio system is initializing');
      }
      if (!isInitializedRef.current) {
        throw new Error('Audio system not initialized. Call init() first.');
      }
      await loadModelInternal(modelUrl);
    },
    []
  );

  // Load audio
  const loadAudio = useCallback(
    async (src: string): Promise<void> => {
      const audioElement = getAudioNodes().audioElement;

      if (!audioElement) {
        throw new Error('Audio element not initialized');
      }

      try {
        audioElement.src = src;
        await new Promise<void>((resolve, reject) => {
          const handleLoad = () => {
            audioElement.removeEventListener('loadeddata', handleLoad);
            audioElement.removeEventListener('error', handleError);
            resolve();
          };
          const handleError = () => {
            audioElement.removeEventListener('loadeddata', handleLoad);
            audioElement.removeEventListener('error', handleError);
            reject(new Error('Failed to load audio'));
          };

          audioElement.addEventListener('loadeddata', handleLoad, {
            once: true,
          });
          audioElement.addEventListener('error', handleError, { once: true });
          audioElement.load();
        });

        setAudioState(prev => ({ ...prev, audioUrl: src }));
      } catch (error) {
        console.error('Error loading audio:', error);
        throw error;
      }
    },
    [getAudioNodes]
  );

  // Load IR with configuration
  // NOTE: This function uses a polling pattern to wait for audio nodes. This is defensive
  // code for edge cases where loadIr might be called before audio nodes are fully available.
  // In normal usage, callers should ensure init() has completed before calling loadIr().
  // TODO: Consider removing polling and requiring callers to use isAudioReady() guard.
  const loadIr = useCallback(
    async ({ url, wetAmount = 1, gainAmount = 1 }: IrConfig): Promise<void> => {
      // Poll for audio nodes with timeout (defensive - see note above)
      const pollForNodes = async (): Promise<{
        audioContext: AudioContext;
        audioWorkletNode: AudioWorkletNode;
        outputGainNode: GainNode;
      }> => {
        const timeout = 5000; // 5 seconds
        const interval = 50; // 50ms between checks
        const startTime = Date.now();

        while (Date.now() - startTime < timeout) {
          const nodes = getAudioNodes();
          const { audioContext, audioWorkletNode, outputGainNode } = nodes;

          if (audioContext && audioWorkletNode && outputGainNode) {
            return { audioContext, audioWorkletNode, outputGainNode };
          }
          // Wait before next check
          await new Promise(resolve => setTimeout(resolve, interval));
        }

        throw new Error(
          'Audio nodes not initialized after 5 seconds - timeout exceeded'
        );
      };

      const { audioContext, audioWorkletNode, outputGainNode } =
        await pollForNodes();
      const nodes = getAudioNodes();

      try {
        // Create or update gain nodes
        if (!nodes.irWetGain) {
          nodes.irWetGain = new GainNode(audioContext, { gain: wetAmount });
        } else {
          nodes.irWetGain.gain.setValueAtTime(
            wetAmount,
            audioContext.currentTime
          );
        }

        if (!nodes.irDryGain) {
          nodes.irDryGain = new GainNode(audioContext, { gain: 1 - wetAmount });
        } else {
          nodes.irDryGain.gain.setValueAtTime(
            1 - wetAmount,
            audioContext.currentTime
          );
        }

        if (!nodes.irGain) {
          nodes.irGain = new GainNode(audioContext, { gain: gainAmount });
        } else {
          nodes.irGain.gain.setValueAtTime(
            gainAmount,
            audioContext.currentTime
          );
        }

        // Fetch and decode IR
        const response = await fetch(url);
        if (!response.ok) {
          throw new Error(`Failed to fetch IR: ${response.statusText}`);
        }

        const arrayBuffer = await response.arrayBuffer();
        const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

        // Create new convolver
        const newIrNode = new ConvolverNode(audioContext);
        newIrNode.buffer = audioBuffer;

        // Disconnect existing connections
        audioWorkletNode.disconnect();
        nodes.irNode?.disconnect();
        nodes.irWetGain?.disconnect();
        nodes.irDryGain?.disconnect();
        nodes.irGain?.disconnect();

        // Setup parallel wet/dry signal paths
        // Wet path: worklet -> IR -> gain -> wet gain -> output
        audioWorkletNode.connect(newIrNode);
        newIrNode.connect(nodes.irGain);
        nodes.irGain.connect(nodes.irWetGain);
        nodes.irWetGain.connect(outputGainNode);

        // Dry path: worklet -> dry gain -> output
        audioWorkletNode.connect(nodes.irDryGain);
        nodes.irDryGain.connect(outputGainNode);

        nodes.irNode = newIrNode;
        setAudioState(prev => ({ ...prev, irUrl: url }));
      } catch (error) {
        console.error('Error loading IR:', error);
        throw error;
      }
    },
    [getAudioNodes]
  );

  // Remove IR
  const removeIr = useCallback((): void => {
    const nodes = getAudioNodes();
    const { audioWorkletNode, outputGainNode } = nodes;

    if (!audioWorkletNode || !outputGainNode) return;

    // Disconnect all IR nodes
    audioWorkletNode.disconnect();
    nodes.irNode?.disconnect();
    nodes.irWetGain?.disconnect();
    nodes.irDryGain?.disconnect();
    nodes.irGain?.disconnect();

    // Reset IR nodes
    nodes.irNode = null;
    nodes.irWetGain = null;
    nodes.irDryGain = null;
    nodes.irGain = null;

    // Reconnect direct path
    audioWorkletNode.connect(outputGainNode);
    setAudioState(prev => ({ ...prev, irUrl: null }));
  }, [getAudioNodes]);

  // Set bypass to a specific state (idempotent — reads audio node to determine if action is needed)
  const setBypass = useCallback((bypassed: boolean): void => {
    if (!isAudioReady()) {
      console.warn('Cannot set bypass: audio not initialized');
      return;
    }

    const nodes = getAudioNodes();
    const {
      audioWorkletNode,
      audioContext,
      bypassNode,
      outputGainNode,
      irNode,
      irWetGain,
      irDryGain,
      irGain,
    } = nodes;

    if (!audioWorkletNode || !audioContext || !bypassNode) {
      console.warn('Cannot set bypass: required nodes not available');
      return;
    }

    const currentlyBypassed = bypassNode.gain.value > 0.5;
    if (currentlyBypassed === bypassed) return; // already in desired state

    try {
      if (bypassed) {
        // Enable bypass
        if (irNode && irWetGain && irDryGain && irGain && outputGainNode) {
          // Disconnect IR paths
          audioWorkletNode.disconnect(irNode);
          audioWorkletNode.disconnect(irDryGain);
          irNode.disconnect(irGain);
          irGain.disconnect(irWetGain);
          irWetGain.disconnect(outputGainNode);
          irDryGain.disconnect(outputGainNode);
        } else if (outputGainNode) {
          // Disconnect direct path
          audioWorkletNode.disconnect(outputGainNode);
        }
        bypassNode.gain.setTargetAtTime(1, audioContext.currentTime, 0.002);
      } else {
        // Disable bypass
        if (irNode && irWetGain && irDryGain && irGain && outputGainNode) {
          // Reconnect IR paths
          audioWorkletNode.connect(irNode);
          irNode.connect(irGain);
          irGain.connect(irWetGain);
          irWetGain.connect(outputGainNode);
          audioWorkletNode.connect(irDryGain);
          irDryGain.connect(outputGainNode);
        } else if (outputGainNode) {
          // Reconnect direct path
          audioWorkletNode.connect(outputGainNode);
        }
        bypassNode.gain.setTargetAtTime(0, audioContext.currentTime, 0.002);
      }

      setAudioState(prev => ({ ...prev, isBypassed: bypassed }));
    } catch (error) {
      console.error('Error setting bypass:', error);
    }
  }, [isAudioReady, getAudioNodes]);

  // Sync engine settings (model, IR, bypass) to match a player's preferences
  const syncEngineSettings = useCallback(async (config: {
    modelUrl: string;
    ir: { url: string | null; mix?: number; gain?: number };
    bypassed: boolean;
  }): Promise<void> => {
    const currentModelUrl = audioState.modelUrl;
    const currentIrUrl = audioState.irUrl;

    if (currentModelUrl !== config.modelUrl) {
      await loadModel(config.modelUrl);
    }
    if (config.ir.url) {
      if (currentIrUrl !== config.ir.url) {
        await loadIr({ url: config.ir.url, wetAmount: config.ir.mix, gainAmount: config.ir.gain });
      }
    } else if (currentIrUrl) {
      removeIr();
    }
    setBypass(config.bypassed);
  }, [audioState.modelUrl, audioState.irUrl, loadModel, loadIr, removeIr, setBypass]);

  // Connect visualizer
  const connectVisualizerNode = useCallback(
    (analyserNode: AnalyserNode): (() => void) => {
      const { outputGainNode } = getAudioNodes();

      if (!outputGainNode) {
        return () => {};
      }

      outputGainNode.connect(analyserNode);

      return () => {
        try {
          outputGainNode.disconnect(analyserNode);
        } catch {
          // Node already disconnected
        }
      };
    },
    [getAudioNodes]
  );

  // Cleanup
  const cleanup = useCallback((): void => {
    const nodes = getAudioNodes();
    const { audioElement, audioContext } = nodes;

    // Stop file playback
    if (audioElement) {
      audioElement.pause();
      audioElement.currentTime = 0;
      audioElement.remove();
    }

    // Clean up all live input nodes
    cleanupLiveInputNodes(nodes);

    cleanupOutputWorkaroundRouting(nodes);

    removeIr();

    // Disconnect all core audio nodes
    const coreNodes: (AudioNode | null)[] = [
      nodes.inputGainNode,
      nodes.outputGainNode,
      nodes.bypassNode,
      nodes.inputMeterNode,
      nodes.outputMeterNode,
      nodes.sourceNode,
      nodes.audioWorkletNode,
    ];
    for (const node of coreNodes) {
      try { node?.disconnect(); } catch { /* already disconnected */ }
    }

    // Close AudioContext
    if (audioContext && audioContext.state !== 'closed') {
      audioContext.close();
    }

    // Null out all node references
    nodes.audioContext = null;
    nodes.audioElement = null;
    nodes.audioWorkletNode = null;
    nodes.inputGainNode = null;
    nodes.outputGainNode = null;
    nodes.bypassNode = null;
    nodes.sourceNode = null;
    nodes.inputMeterNode = null;
    nodes.outputMeterNode = null;
    nodes.irNode = null;
    nodes.irWetGain = null;
    nodes.irDryGain = null;
    nodes.irGain = null;

    isInitializedRef.current = false;

    setAudioState(prev => ({
      ...prev,
      initState: 'uninitialized',
      isPlaying: false,
      activePlayerId: null,
      modelUrl: null,
      audioUrl: null,
      isBypassed: false,
      inputMode: { type: 'preview' },
      liveInputConfig: null,
    }));
  }, [getAudioNodes, removeIr]);

  // Start live audio input from microphone/audio interface
  // Optional initialChannel and initialChannelGains allow restoring a previous configuration
  // without needing to call selectLiveInputChannel separately (which would have stale closure issues)
  const startLiveInput = useCallback(async (
    deviceId?: string,
    options?: { initialChannel?: ChannelSelection; initialChannelGains?: Record<ChannelSelection, number> }
  ): Promise<void> => {
    const requestedChannel = options?.initialChannel ?? 'first';
    const initialChannelGains = options?.initialChannelGains ?? { first: 0, second: 0 };

    if (!isAudioReady()) {
      throw new Error('Audio system not initialized. Call init() first.');
    }
    const nodes = getAudioNodes();
    const { audioContext, inputMeterNode, sourceNode, audioElement } = nodes;

    if (!audioContext || !inputMeterNode) {
      throw new Error('Required audio nodes not available.');
    }

    // Take over audio engine: stop any current playback and reset ownership
    if (audioElement) {
      audioElement.pause();
    }
    if (nodes.outputGainNode && audioContext) {
      nodes.outputGainNode.gain.setTargetAtTime(0, audioContext.currentTime, 0.01);
    }

    setAudioState(prev => ({
      ...prev,
      inputMode: { type: 'connecting' },
      isPlaying: false,
      activePlayerId: null,
    }));

    try {
      // Clean up any existing live input nodes before setting up new ones
      cleanupLiveInputNodes(nodes);

      // Request microphone/audio interface access with settings optimized for instruments
      const baseAudioConstraints: MediaTrackConstraints = {
        channelCount: { ideal: 2 },  // Request stereo if available
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
      };

      let stream: MediaStream;
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          audio: {
            ...baseAudioConstraints,
            deviceId: deviceId ? { exact: deviceId } : undefined,
          },
        });
      } catch (error) {
        if (deviceId && error instanceof DOMException && error.name === 'OverconstrainedError') {
          console.warn(`Device ${deviceId} not available, falling back to default`);
          stream = await navigator.mediaDevices.getUserMedia({
            audio: baseAudioConstraints,
          });
        } else {
          throw error;
        }
      }

      nodes.mediaStream = stream;

      // Get actual channel count and device ID from the track
      const track = stream.getAudioTracks()[0];
      const settings = track.getSettings();
      const channelCount = settings.channelCount ?? 1;
      const actualDeviceId = settings.deviceId ?? deviceId ?? '';

      // Clamp channel selection to what the device actually supports
      const initialChannel = (channelCount > 1) ? requestedChannel : 'first';
      const initialChannelIndex = initialChannel === 'first' ? 0 : 1;

      // Listen for track ending (device disconnected, permission revoked, etc.)
      // This is the most reliable signal — Chrome kills the track immediately on permission reset,
      // but may not fire the Permissions API 'change' event.
      track.addEventListener('ended', () => {
        console.warn('Audio track ended unexpectedly');
        handleLiveInputUnavailable('device-disconnected');
      });

      // Create source node from the live stream
      const liveSource = audioContext.createMediaStreamSource(stream);
      nodes.liveSourceNode = liveSource;

      // Disconnect file source from inputGainNode
      sourceNode?.disconnect();

      // Create dedicated gain node for live input (applied before channel splitting)
      nodes.liveInputGainNode = new GainNode(audioContext, { gain: 1 });

      // Create channel splitter for multi-channel input
      nodes.channelSplitterNode = audioContext.createChannelSplitter(2);
      nodes.channelMergerNode = audioContext.createChannelMerger(1);

      // Create per-channel meters (post-gain, since gain is applied before splitting)
      const meterConfig = { fftSize: 2048 };
      nodes.channel0PreviewMeter = new AnalyserNode(audioContext, meterConfig);
      nodes.channel1PreviewMeter = new AnalyserNode(audioContext, meterConfig);

      // Connect: liveSource → liveInputGainNode → channelSplitter
      // This applies gain BEFORE channel splitting, so meters show post-gain levels
      liveSource.connect(nodes.liveInputGainNode);
      nodes.liveInputGainNode.connect(nodes.channelSplitterNode);

      // Connect each channel to its meter (these are now post-gain)
      nodes.channelSplitterNode.connect(nodes.channel0PreviewMeter, 0);
      if (channelCount > 1) {
        nodes.channelSplitterNode.connect(nodes.channel1PreviewMeter, 1);
      }

      // Route selected channel to processing chain
      // Connect: channelSplitter → channelMerger → inputMeterNode (bypassing inputGainNode)
      nodes.channelSplitterNode.connect(nodes.channelMergerNode, initialChannelIndex, 0);
      nodes.channelMergerNode.connect(inputMeterNode);

      // Apply initial gain for the selected channel
      const initialGainDb = initialChannelGains[initialChannel] ?? 0;
      const initialLinearGain = dbToLinear(initialGainDb);
      nodes.liveInputGainNode.gain.setValueAtTime(initialLinearGain, audioContext.currentTime);

      setAudioState(prev => ({
        ...prev,
        inputMode: { type: 'live' },
        liveInputConfig: {
          deviceId: actualDeviceId,
          channelCount: channelCount,
          selectedChannel: initialChannel,
          channelGains: initialChannelGains,
        },
      }));

      // Resume audio context if suspended
      if (audioContext.state === 'suspended') {
        await audioContext.resume();
      }

      // Re-apply output device routing after changing input sources
      const selectedOutputId = audioOutputDevices.selectedDeviceId;
      await applyOutputDeviceRouting(nodes, selectedOutputId);
    } catch (error) {
      // Clean up any partially-created live input nodes
      cleanupLiveInputNodes(nodes);

      setAudioState(prev => ({
        ...prev,
        inputMode: { type: 'preview' },
      }));
      console.error('Error starting live input:', error);
      throw error;
    }
  }, [isAudioReady, getAudioNodes, audioOutputDevices.selectedDeviceId, handleLiveInputUnavailable]);

  // Reconnect live input using the saved liveInputConfig.
  // No-op if no config exists or live input is already active.
  const reconnectLiveInput = useCallback(async (): Promise<void> => {
    if (audioState.inputMode.type === 'live') return;
    if (!audioState.liveInputConfig) return;
    await startLiveInput(audioState.liveInputConfig.deviceId, {
      initialChannel: audioState.liveInputConfig.selectedChannel,
      initialChannelGains: audioState.liveInputConfig.channelGains,
    });
  }, [audioState.inputMode.type, audioState.liveInputConfig, startLiveInput]);

  // Stop live audio input and prepare for preview mode takeover
  // Note: This preserves liveInputConfig so players can still see/use the configured device.
  // Use clearLiveInputConfig() to fully disconnect and clear the saved configuration.
  const stopLiveInput = useCallback((): void => {
    teardownLiveInput(getAudioNodes(), { muteOutput: true });

    // Reset ownership - no player owns the audio engine until they press play
    setAudioState(prev => ({
      ...prev,
      inputMode: { type: 'preview' },
      isPlaying: false,
      activePlayerId: null,
    }));
  }, [getAudioNodes]);

  // Clear the saved live input configuration (used when user explicitly disconnects in settings)
  // This stops live input if active AND clears the saved config so it won't auto-reconnect
  const clearLiveInputConfig = (): void => {
    const nodes = getAudioNodes();

    // Stop live input if currently active
    if (nodes.mediaStream) {
      teardownLiveInput(nodes, { muteOutput: true });
    }

    // Clear both inputMode and liveInputConfig
    setAudioState(prev => ({
      ...prev,
      inputMode: { type: 'preview' },
      liveInputConfig: null,
      isPlaying: false,
      activePlayerId: null,
    }));
  };

  // Select which channel to route from live input to processing
  const selectLiveInputChannel = useCallback((channel: ChannelSelection): void => {
    const nodes = getAudioNodes();
    const { channelSplitterNode, channelMergerNode, inputGainNode, liveInputGainNode, audioContext } = nodes;

    if (!channelSplitterNode || !channelMergerNode || !inputGainNode) {
      console.warn('Cannot select channel: live input not active');
      return;
    }

    const newChannelIndex = channel === 'first' ? 0 : 1;

    // Disconnect all splitter → merger connections, then connect only the new channel.
    // This avoids relying on React state to track the current connection, which can be
    // stale due to closure capture. Since connect() is additive, a missed disconnect
    // would result in both channels being heard simultaneously.
    channelSplitterNode.disconnect(channelMergerNode);
    channelSplitterNode.connect(channelMergerNode, newChannelIndex, 0);

    // Apply gain for the new channel (side effect outside state updater)
    if (liveInputGainNode && audioContext && audioState.liveInputConfig) {
      const gainDb = audioState.liveInputConfig.channelGains?.[channel] ?? 0;
      liveInputGainNode.gain.setTargetAtTime(dbToLinear(gainDb), audioContext.currentTime, 0.02);
    }

    // Pure state update
    setAudioState(prev => {
      if (!prev.liveInputConfig) return prev;
      return {
        ...prev,
        liveInputConfig: {
          ...prev.liveInputConfig,
          selectedChannel: channel,
        },
      };
    });
  }, [getAudioNodes, audioState.liveInputConfig]);

  // Set live input gain for the currently selected channel (persists to liveInputConfig)
  const setLiveInputGain = useCallback((gainDb: number): void => {
    const nodes = getAudioNodes();
    const { liveInputGainNode, audioContext } = nodes;

    // Apply gain to the live input gain node (before channel splitting)
    if (liveInputGainNode && audioContext) {
      const linearGain = dbToLinear(gainDb);
      liveInputGainNode.gain.setTargetAtTime(
        linearGain,
        audioContext.currentTime,
        0.02 // 20ms smoothing
      );
    }

    // Persist to liveInputConfig for the currently selected channel
    setAudioState(prev => {
      if (!prev.liveInputConfig) return prev;
      const selectedChannel = prev.liveInputConfig.selectedChannel ?? 'first';
      return {
        ...prev,
        liveInputConfig: {
          ...prev.liveInputConfig,
          channelGains: {
            ...prev.liveInputConfig.channelGains,
            [selectedChannel]: gainDb,
          },
        },
      };
    });
  }, [getAudioNodes]);

  // Open the global settings dialog with a snapshot of current state
  const openSettingsDialog = useCallback((config: {
    sourceMode: SourceMode;
    playerId?: string;
    selectedModel: Model;
    selectedIr: IR;
  }): void => {
    const hadExistingConfig = audioState.liveInputConfig !== null;
    const snapshot: SettingsSnapshot = {
      outputDeviceId: audioOutputDevices.selectedDeviceId,
      inputMode: { ...audioState.inputMode },
      liveInputConfig: audioState.liveInputConfig ? { ...audioState.liveInputConfig } : null,
      isPlaying: audioState.isPlaying,
      isBypassed: audioState.isBypassed,
      activePlayerId: audioState.activePlayerId,
    };

    settingsSnapshotRef.current = { snapshot, hadExistingConfig };

    setSettingsDialog({
      isOpen: true,
      sourceMode: config.sourceMode,
      playerId: config.playerId,
      selectedModel: config.selectedModel,
      selectedIr: config.selectedIr,
      snapshot,
      hadExistingConfig,
    });
  }, [audioState.liveInputConfig, audioState.inputMode, audioState.isPlaying, audioState.isBypassed, audioState.activePlayerId, audioOutputDevices.selectedDeviceId]);

  // Set playing state (controls audioElement in preview mode, outputGainNode in live mode)
  // playerId identifies which player is taking control (required when playing=true)
  const setPlaying = useCallback((playing: boolean, playerId?: string): void => {
    // Guard: require initialization when starting playback
    // (stopping is always allowed to handle edge cases)
    if (playing && !isAudioReady()) {
      console.warn('Cannot start playback: audio not initialized');
      return;
    }

    const nodes = getAudioNodes();
    const { audioElement, outputGainNode, audioContext, mediaStream } = nodes;

    // Check actual node state, not React state (which can be stale due to async updates)
    const isActuallyLiveMode = mediaStream !== null && mediaStream.active;

    // Update audio nodes based on actual input mode
    if (isActuallyLiveMode) {
      // Live mode: control output gain only (no audio element involved)
      if (outputGainNode && audioContext) {
        // Resume context if suspended
        if (playing && audioContext.state === 'suspended') {
          audioContext.resume().catch(error => {
            console.error('Failed to resume audio context:', error);
            setAudioState(prev => ({ ...prev, isPlaying: false, activePlayerId: null }));
          });
        }
        outputGainNode.gain.setTargetAtTime(
          playing ? 1 : 0,
          audioContext.currentTime,
          0.01 // 10ms for quick response
        );
      }
    } else {
      // Preview mode: control audio element playback
      if (audioElement) {
        if (playing) {
          audioContext?.resume().catch(error => {
            console.error('Failed to resume audio context:', error);
          });
          audioElement.play().catch(error => {
            console.error('Playback failed:', error);
            setAudioState(prev => ({ ...prev, isPlaying: false, activePlayerId: null }));
          });
        } else {
          audioElement.pause();
        }
      }
      // Also set output gain for preview mode
      if (outputGainNode && audioContext) {
        outputGainNode.gain.setTargetAtTime(
          playing ? 1 : 0,
          audioContext.currentTime,
          0.01
        );
      }
    }

    setAudioState(prev => ({
      ...prev,
      isPlaying: playing,
      activePlayerId: playing ? (playerId ?? prev.activePlayerId) : null,
    }));
  }, [isAudioReady, getAudioNodes, audioState.inputMode.type]);

  // Close the global settings dialog, restoring snapshot on cancel
  const closeSettingsDialog = useCallback(async (options: { saved: boolean }): Promise<void> => {
    const { snapshot, hadExistingConfig } = settingsSnapshotRef.current;

    if (!options.saved && snapshot) {

      // If we opened without a live config but one was created, clear it
      if (!hadExistingConfig && audioState.liveInputConfig !== null) {
        clearLiveInputConfig();
      } else if (hadExistingConfig && snapshot.liveInputConfig) {
        // Only reconnect live input if it was actually active (not just configured)
        if (snapshot.inputMode.type === 'live') {
          await startLiveInput(snapshot.liveInputConfig.deviceId, {
            initialChannel: snapshot.liveInputConfig.selectedChannel,
            initialChannelGains: snapshot.liveInputConfig.channelGains,
          });

          // Restore playback and bypass state
          if (snapshot.isPlaying && snapshot.activePlayerId) {
            setPlaying(true, snapshot.activePlayerId);
          } else {
            setPlaying(false);
          }
          setBypass(snapshot.isBypassed);
        } else {
          // Was in preview mode — just restore the config without connecting
          setAudioState(prev => ({
            ...prev,
            liveInputConfig: snapshot.liveInputConfig,
          }));
        }
      }

      // Restore output device
      await setOutputDevice(snapshot.outputDeviceId);
    }

    setSettingsDialog(prev => ({
      ...prev,
      isOpen: false,
      snapshot: null,
    }));
  }, [audioState.liveInputConfig, startLiveInput, setPlaying, setOutputDevice, setBypass]);

  // Full teardown on provider unmount (handles HMR/routing teardown)
  useEffect(() => {
    const nodesRef = audioNodesRef;
    const initRef = isInitializedRef;
    return () => {
      const nodes = nodesRef.current;
      if (nodes.audioContext && nodes.audioContext.state !== 'closed') {
        nodes.audioContext.close();
      }
      initRef.current = false;
    };
  }, []);

  // Memoize context value
  const contextValue = useMemo<T3kPlayerContextType>(
    () => ({
      audioState,
      microphonePermission,
      audioInputDevices,
      audioOutputDevices,
      getAudioNodes,
      isAudioReady,
      init,
      loadModel,
      loadAudio,
      loadIr,
      removeIr,
      setBypass,
      syncEngineSettings,
      cleanup,
      connectVisualizerNode,
      startLiveInput,
      reconnectLiveInput,
      stopLiveInput,
      selectLiveInputChannel,
      setLiveInputGain,
      setOutputDevice,
      setPlaying,
      requestMicrophonePermission,
      refreshAudioDevices,
      refreshAudioOutputDevices,
      settingsDialog,
      openSettingsDialog,
      closeSettingsDialog,
    }),
    [
      audioState,
      microphonePermission,
      audioInputDevices,
      audioOutputDevices,
      getAudioNodes,
      isAudioReady,
      init,
      loadModel,
      loadAudio,
      loadIr,
      removeIr,
      setBypass,
      syncEngineSettings,
      cleanup,
      connectVisualizerNode,
      startLiveInput,
      reconnectLiveInput,
      stopLiveInput,
      selectLiveInputChannel,
      setLiveInputGain,
      setOutputDevice,
      setPlaying,
      requestMicrophonePermission,
      refreshAudioDevices,
      refreshAudioOutputDevices,
      settingsDialog,
      openSettingsDialog,
      closeSettingsDialog,
    ]
  );

  return (
    <T3kPlayerContext.Provider value={contextValue}>
      {children}
    </T3kPlayerContext.Provider>
  );
}

// SSR-safe defaults (no-op implementations for server-side rendering)
function createNoopContext(): T3kPlayerContextType {
  return {
    audioState: {
      initState: 'uninitialized',
      isPlaying: false,
      activePlayerId: null,
      isBypassed: false,
      modelUrl: null,
      irUrl: null,
      audioUrl: null,
      inputMode: { type: 'preview' },
      liveInputConfig: null,
    },
    microphonePermission: { status: 'idle', error: null },
    audioInputDevices: { devices: [], isLoading: false, error: null, preferredDeviceId: null },
    audioOutputDevices: { devices: [], selectedDeviceId: null },
    getAudioNodes: () => ({ ...EMPTY_AUDIO_NODES }),
    isAudioReady: () => false,
    init: async () => {},
    loadModel: async () => {},
    loadAudio: async () => {},
    loadIr: async () => {},
    removeIr: () => {},
    setBypass: () => {},
    syncEngineSettings: async () => {},
    cleanup: () => {},
    connectVisualizerNode: () => () => {},
    startLiveInput: async () => {},
    reconnectLiveInput: async () => {},
    stopLiveInput: () => {},
    selectLiveInputChannel: () => {},
    setLiveInputGain: () => {},
    setOutputDevice: async () => {},
    setPlaying: (() => {}) as T3kPlayerContextType['setPlaying'],
    requestMicrophonePermission: async () => null,
    refreshAudioDevices: async () => ({ inputDevices: [], preferredDeviceId: null }),
    refreshAudioOutputDevices: async () => {},
    settingsDialog: { isOpen: false, sourceMode: 'preview', snapshot: null, hadExistingConfig: false },
    openSettingsDialog: () => {},
    closeSettingsDialog: () => {},
  };
}

// Custom hook — useContext is called unconditionally to satisfy Rules of Hooks
export function useT3kPlayerContext(): T3kPlayerContextType {
  const context = useContext(T3kPlayerContext);

  if (typeof window === 'undefined') {
    return createNoopContext();
  }

  if (!context) {
    throw new Error(
      'useT3kPlayerContext must be used within a T3kPlayerContextProvider'
    );
  }

  return context;
}
