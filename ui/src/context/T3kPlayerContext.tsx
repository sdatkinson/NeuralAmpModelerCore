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
import { InputMode, AudioInputDevice, MicrophonePermissionStatus, MicrophonePermissionState, AudioInputDeviceState, ChannelSelection } from '../types';
import { dbToLinear } from '../utils/metering';

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

// Types
interface AudioNodes {
  audioContext: AudioContext | null;
  audioElement: HTMLAudioElement | null;
  audioWorkletNode: AudioWorkletNode | null;
  inputGainNode: GainNode | null;
  outputGainNode: GainNode | null;
  bypassNode: GainNode | null;
  irNode: ConvolverNode | null;
  irWetGain: GainNode | null;
  irDryGain: GainNode | null;
  irGain: GainNode | null;
  sourceNode: MediaElementAudioSourceNode | null;
  // Live input nodes
  liveSourceNode: MediaStreamAudioSourceNode | null;
  liveInputGainNode: GainNode | null;
  mediaStream: MediaStream | null;
  // Metering nodes
  inputMeterNode: AnalyserNode | null;
  outputMeterNode: AnalyserNode | null;
  // Output protection
  limiterNode: DynamicsCompressorNode | null;
  // Channel selection (for multi-channel interfaces)
  channelSplitterNode: ChannelSplitterNode | null;
  channelMergerNode: ChannelMergerNode | null;
  channel0PreviewMeter: AnalyserNode | null;
  channel1PreviewMeter: AnalyserNode | null;
}

// Explicit initialization states for visibility into the init process
type AudioInitState = 'uninitialized' | 'initializing' | 'ready';

// Snapshot of live input configuration for restore operations
interface LiveConfigSnapshot {
  deviceId: string;
  channel: ChannelSelection;
  channelGains: Record<ChannelSelection, number>;
  wasPlaying: boolean;
  wasBypassed: boolean;
}

interface AudioState {
  initState: AudioInitState;
  isPlaying: boolean;  // Whether audio is playing (preview) or monitoring (live)
  isBypassed: boolean;
  isLiveConnecting: boolean;  // Whether live input is being connected/switched
  modelUrl: string | null;
  irUrl: string | null;
  audioUrl: string | null;
  // Input mode state
  inputMode: InputMode;
  // Snapshot of live config for restoring (used by mode switching and dialog cancel)
  liveConfigSnapshot: LiveConfigSnapshot | null;
}

interface IrConfig {
  url: string;
  wetAmount?: number;
  gainAmount?: number;
}

interface T3kPlayerContextType {
  // State
  audioState: AudioState;
  microphonePermission: MicrophonePermissionState;
  audioInputDevices: AudioInputDeviceState;

  // Getters
  getAudioNodes: () => AudioNodes;

  // Actions
  init: (options?: { audioUrl?: string; modelUrl?: string }) => Promise<void>;
  loadModel: (modelUrl: string) => Promise<void>;
  loadAudio: (src: string) => Promise<void>;
  loadIr: (config: IrConfig) => Promise<void>;
  removeIr: () => void;
  toggleBypass: () => void;
  cleanup: () => void;
  connectVisualizerNode: (analyserNode: AnalyserNode) => () => void;

  // Input mode actions
  enumerateInputDevices: () => Promise<AudioInputDevice[]>;
  startLiveInput: (deviceId?: string) => Promise<void>;
  stopLiveInput: () => void;
  setInputMode: (mode: InputMode) => void;
  isLiveInputActive: () => boolean;
  selectLiveInputChannel: (channel: ChannelSelection) => void;
  setLiveInputGain: (gainDb: number) => void;

  // Playback control
  setPlaying: (playing: boolean) => void;

  // Live config snapshot (for mode switching and dialog cancel)
  saveLiveSnapshot: () => void;
  restoreLiveSnapshot: (options?: { includePlaybackState?: boolean }) => Promise<void>;
  clearLiveSnapshot: () => void;

  // Microphone permission actions
  requestMicrophonePermission: () => Promise<string | null>;  // Returns preferred device ID

  // Audio input device actions
  refreshAudioInputDevices: () => Promise<void>;
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
    isBypassed: false,
    isLiveConnecting: false,
    modelUrl: null,
    irUrl: null,
    audioUrl: null,
    inputMode: { type: 'preview' },
    liveConfigSnapshot: null,
  });

  // Microphone permission state (permission concerns only)
  const [microphonePermission, setMicrophonePermission] = useState<MicrophonePermissionState>({
    status: 'idle',
    error: null,
  });

  // Audio input device state (device concerns only)
  const [audioInputDevices, setAudioInputDevices] = useState<AudioInputDeviceState>({
    devices: [],
    isLoading: false,
    error: null,
    preferredDeviceId: null,
  });

  // Refs for non-render-triggering data
  const audioNodesRef = useRef<AudioNodes>({
    audioContext: null,
    audioElement: null,
    audioWorkletNode: null,
    inputGainNode: null,
    outputGainNode: null,
    bypassNode: null,
    irNode: null,
    irWetGain: null,
    irDryGain: null,
    irGain: null,
    sourceNode: null,
    liveSourceNode: null,
    liveInputGainNode: null,
    mediaStream: null,
    // Metering nodes
    inputMeterNode: null,
    outputMeterNode: null,
    // Output protection
    limiterNode: null,
    // Channel selection
    channelSplitterNode: null,
    channelMergerNode: null,
    channel0PreviewMeter: null,
    channel1PreviewMeter: null,
  });

  const isInitializingRef = useRef<boolean>(false);
  const isInitializedRef = useRef<boolean>(false);
  const modulePromise = useModule();

  // Getter for audio nodes
  const getAudioNodes = useCallback((): AudioNodes => {
    return audioNodesRef.current;
  }, []);

  // Query browser's microphone permission state
  const queryBrowserPermission = useCallback(async (): Promise<MicrophonePermissionStatus> => {
    try {
      // Note: 'microphone' requires casting as it's not in all TypeScript definitions
      const result = await navigator.permissions.query({ name: 'microphone' as PermissionName });
      if (result.state === 'granted') return 'granted';
      if (result.state === 'denied') return 'denied';
      return 'idle'; // 'prompt' state means user hasn't decided yet
    } catch {
      // Permissions API not supported (e.g., Firefox for microphone)
      return 'idle';
    }
  }, []);

  // Query permission state on mount and listen for changes
  useEffect(() => {
    if (typeof window === 'undefined') return;

    let mounted = true;

    const checkPermission = async () => {
      const status = await queryBrowserPermission();
      if (mounted) {
        setMicrophonePermission(prev => ({
          ...prev,
          status,
          // Clear error if permission was granted
          error: status === 'granted' ? null : prev.error,
        }));
      }
    };

    checkPermission();

    // Listen for permission changes
    const setupPermissionListener = async () => {
      try {
        const result = await navigator.permissions.query({ name: 'microphone' as PermissionName });

        const handleChange = () => {
          if (!mounted) return;
          if (result.state === 'granted') {
            setMicrophonePermission(prev => ({
              ...prev,
              status: 'granted',
              error: null,
            }));
          } else if (result.state === 'denied') {
            setMicrophonePermission(prev => ({
              ...prev,
              status: 'denied',
              error: 'Microphone access was revoked in browser settings.',
            }));
          } else {
            setMicrophonePermission(prev => ({
              ...prev,
              status: 'idle',
            }));
          }
        };

        result.addEventListener('change', handleChange);
        return () => result.removeEventListener('change', handleChange);
      } catch {
        // Permissions API not supported
        return undefined;
      }
    };

    let cleanupListener: (() => void) | undefined;
    setupPermissionListener().then(cleanup => {
      cleanupListener = cleanup;
    });

    return () => {
      mounted = false;
      cleanupListener?.();
    };
  }, [queryBrowserPermission]);

  // Request microphone permission (permission concerns only)
  // Returns the device ID the user selected in the browser's permission dialog
  const requestMicrophonePermission = useCallback(async (): Promise<string | null> => {
    setMicrophonePermission({
      status: 'pending',
      error: null,
    });

    // Start loading devices in parallel
    setAudioInputDevices(prev => ({
      ...prev,
      isLoading: true,
      error: null,
    }));

    try {
      // Request permission - capture stream to detect device user selected in browser
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

      // Get the device ID the user selected in the browser's permission dialog
      const track = stream.getAudioTracks()[0];
      const selectedDeviceId = track?.getSettings()?.deviceId ?? null;

      // Stop the stream - we only needed it to get permission and device selection
      // startLiveInput will create a new stream with proper settings when user saves
      stream.getTracks().forEach(t => t.stop());

      // Permission granted
      setMicrophonePermission({
        status: 'granted',
        error: null,
      });

      // Enumerate devices after permission is granted
      const allDevices = await navigator.mediaDevices.enumerateDevices();
      const audioInputs = allDevices
        .filter(device => device.kind === 'audioinput')
        .map(device => ({
          deviceId: device.deviceId,
          label: device.label || `Input ${device.deviceId.slice(0, 8)}`,
        }));

      setAudioInputDevices({
        devices: audioInputs,
        isLoading: false,
        error: audioInputs.length === 0 ? 'No audio input devices found.' : null,
        preferredDeviceId: selectedDeviceId,
      });

      return selectedDeviceId;
    } catch (error) {
      // Stop device loading
      setAudioInputDevices(prev => ({
        ...prev,
        isLoading: false,
      }));

      // Determine if this was a permission denial or other error
      if (error instanceof DOMException) {
        if (error.name === 'NotAllowedError' || error.name === 'PermissionDeniedError') {
          setMicrophonePermission({
            status: 'denied',
            error: 'You can enable microphone access in your browser settings.',
          });
        } else if (error.name === 'NotFoundError') {
          setMicrophonePermission({
            status: 'error',
            error: 'No microphone or audio input device was found.',
          });
        } else if (error.name === 'NotReadableError') {
          setMicrophonePermission({
            status: 'error',
            error: 'Your microphone is being used by another application.',
          });
        } else {
          setMicrophonePermission({
            status: 'error',
            error: error.message,
          });
        }
      } else {
        setMicrophonePermission({
          status: 'error',
          error: 'An unexpected error occurred. Please try again.',
        });
      }
      throw error;
    }
  }, []);

  // Refresh audio input devices (device concerns only)
  const refreshAudioInputDevices = useCallback(async (): Promise<void> => {
    // First check permission state
    const status = await queryBrowserPermission();
    
    // Update permission state
    setMicrophonePermission(prev => ({
      ...prev,
      status,
      error: status === 'granted' ? null : prev.error,
    }));

    if (status !== 'granted') {
      // Can't enumerate devices without permission
      return;
    }

    setAudioInputDevices(prev => ({
      ...prev,
      isLoading: true,
      error: null,
    }));

    try {
      const allDevices = await navigator.mediaDevices.enumerateDevices();
      const audioInputs = allDevices
        .filter(device => device.kind === 'audioinput')
        .map(device => ({
          deviceId: device.deviceId,
          label: device.label || `Input ${device.deviceId.slice(0, 8)}`,
        }));
      
      setAudioInputDevices(prev => ({
        ...prev,
        devices: audioInputs,
        isLoading: false,
        error: audioInputs.length === 0 ? 'No audio input devices found.' : null,
      }));
    } catch {
      setAudioInputDevices(prev => ({
        ...prev,
        isLoading: false,
        error: 'Failed to enumerate audio devices.',
      }));
    }
  }, [queryBrowserPermission]);

  // Listen for device changes (connect/disconnect)
  useEffect(() => {
    if (typeof window === 'undefined' || !navigator.mediaDevices) return;

    const handleDeviceChange = async () => {
      // Only refresh if we already have permission
      if (microphonePermission.status === 'granted') {
        try {
          const allDevices = await navigator.mediaDevices.enumerateDevices();
          const audioInputs = allDevices
            .filter(device => device.kind === 'audioinput')
            .map(device => ({
              deviceId: device.deviceId,
              label: device.label || `Input ${device.deviceId.slice(0, 8)}`,
            }));

          // Check if currently active device was disconnected
          const currentInputMode = audioState.inputMode;
          if (currentInputMode.type === 'live') {
            const activeDeviceId = currentInputMode.deviceId;
            const activeDeviceStillExists = audioInputs.some(d => d.deviceId === activeDeviceId);

            if (!activeDeviceStillExists) {
              // Active device was disconnected - stop live input and switch to preview
              console.warn('Active audio input device was disconnected');

              // Stop the media stream and clean up
              const nodes = audioNodesRef.current;
              if (nodes.mediaStream) {
                nodes.mediaStream.getTracks().forEach((track: MediaStreamTrack) => track.stop());
                nodes.mediaStream = null;
              }
              if (nodes.liveSourceNode) {
                nodes.liveSourceNode.disconnect();
                nodes.liveSourceNode = null;
              }
              if (nodes.liveInputGainNode) {
                nodes.liveInputGainNode.disconnect();
                nodes.liveInputGainNode = null;
              }
              if (nodes.channelSplitterNode) {
                nodes.channelSplitterNode.disconnect();
                nodes.channelSplitterNode = null;
              }
              if (nodes.channelMergerNode) {
                nodes.channelMergerNode.disconnect();
                nodes.channelMergerNode = null;
              }
              if (nodes.channel0PreviewMeter) {
                nodes.channel0PreviewMeter.disconnect();
                nodes.channel0PreviewMeter = null;
              }
              if (nodes.channel1PreviewMeter) {
                nodes.channel1PreviewMeter.disconnect();
                nodes.channel1PreviewMeter = null;
              }

              // Reconnect file source if available
              if (nodes.sourceNode && nodes.inputGainNode) {
                try {
                  nodes.sourceNode.connect(nodes.inputGainNode);
                } catch {
                  // Source might already be connected
                }
              }

              // Update state to preview mode
              setAudioState(prev => ({
                ...prev,
                inputMode: { type: 'preview' },
              }));

              // Set error message to notify user
              setAudioInputDevices(prev => ({
                ...prev,
                devices: audioInputs,
                isLoading: false,
                error: 'Audio input device was disconnected.',
              }));
              return;
            }
          }

          setAudioInputDevices(prev => ({
            ...prev,
            devices: audioInputs,
            error: audioInputs.length === 0 ? 'All audio devices have been disconnected.' : null,
          }));
        } catch {
          setAudioInputDevices(prev => ({
            ...prev,
            error: 'Failed to refresh device list.',
          }));
        }
      }
    };

    navigator.mediaDevices.addEventListener('devicechange', handleDeviceChange);
    return () => {
      navigator.mediaDevices.removeEventListener('devicechange', handleDeviceChange);
    };
  }, [microphonePermission.status, audioState.inputMode]);

  // Promise that resolves when audio system is ready (WASM callback has fired)
  const audioReadyPromiseRef = useRef<Promise<void> | null>(null);
  const audioReadyResolveRef = useRef<(() => void) | null>(null);

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

      // Set up the ready promise before anything else
      audioReadyPromiseRef.current = new Promise(resolve => {
        audioReadyResolveRef.current = resolve;
      });

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

        // Set up audio worklet callback BEFORE loading script
        // The WASM module checks if(window.wasmAudioWorkletCreated) during init,
        // so the callback must exist before the script executes
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
          const meterConfig = { fftSize: 256, smoothingTimeConstant: 0.3 };
          nodes.inputMeterNode = new AnalyserNode(context, meterConfig);
          nodes.outputMeterNode = new AnalyserNode(context, meterConfig);

          // Create limiter for output protection
          nodes.limiterNode = new DynamicsCompressorNode(context, {
            threshold: -3,
            knee: 6,
            ratio: 20,
            attack: 0.003,
            release: 0.1
          });

          // Create source from audio element
          nodes.sourceNode = context.createMediaElementSource(nodes.audioElement!);

          // Connect audio graph
          nodes.sourceNode.connect(nodes.inputGainNode);
          nodes.inputGainNode.connect(nodes.inputMeterNode!);
          nodes.inputMeterNode!.connect(nodes.bypassNode);
          nodes.bypassNode.connect(nodes.outputGainNode);
          nodes.inputMeterNode!.connect(audioWorkletNode);
          audioWorkletNode.connect(nodes.outputGainNode);
          nodes.outputGainNode.connect(nodes.limiterNode!);
          nodes.limiterNode!.connect(nodes.outputMeterNode!);
          nodes.outputMeterNode!.connect(context.destination);

          context.resume();

          // Resolve the ready promise
          audioReadyResolveRef.current?.();
          audioReadyResolveRef.current = null;
        };

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
        await audioReadyPromiseRef.current;

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

  // Sync audio nodes with React state when initialization completes
  // This ensures audio nodes reflect the current state even if state was set before init
  useEffect(() => {
    if (audioState.initState !== 'ready') return;

    const nodes = getAudioNodes();
    const { outputGainNode, bypassNode, audioContext } = nodes;

    if (outputGainNode && audioContext) {
      // Sync output gain with isPlaying state
      const targetGain = audioState.isPlaying ? 1 : 0;
      outputGainNode.gain.setTargetAtTime(targetGain, audioContext.currentTime, 0.01);
    }

    if (bypassNode && audioContext) {
      // Sync bypass with isBypassed state
      const targetGain = audioState.isBypassed ? 1 : 0;
      bypassNode.gain.setTargetAtTime(targetGain, audioContext.currentTime, 0.01);
    }
  }, [audioState.initState, audioState.isPlaying, audioState.isBypassed, getAudioNodes]);

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
  const loadIr = useCallback(
    async ({ url, wetAmount = 1, gainAmount = 1 }: IrConfig): Promise<void> => {
      // Poll for audio nodes with timeout
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

  // Toggle bypass
  const toggleBypass = useCallback((): void => {
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

    if (!audioWorkletNode || !bypassNode || !audioContext)
      return console.error('Audio nodes not initialized for bypass');

    const isBypassed = bypassNode.gain.value === 1;

    try {
      if (!isBypassed) {
        // Enable bypass
        if (irNode && irWetGain && irDryGain && irGain) {
          // Disconnect IR paths
          audioWorkletNode.disconnect(irNode);
          audioWorkletNode.disconnect(irDryGain);
          irNode.disconnect(irGain);
          irGain.disconnect(irWetGain);
          irWetGain.disconnect(outputGainNode!);
          irDryGain.disconnect(outputGainNode!);
        } else {
          // Disconnect direct path
          audioWorkletNode.disconnect(outputGainNode!);
        }
        bypassNode.gain.setValueAtTime(1, audioContext.currentTime);
      } else {
        // Disable bypass
        if (irNode && irWetGain && irDryGain && irGain) {
          // Reconnect IR paths
          audioWorkletNode.connect(irNode);
          irNode.connect(irGain);
          irGain.connect(irWetGain);
          irWetGain.connect(outputGainNode!);
          audioWorkletNode.connect(irDryGain);
          irDryGain.connect(outputGainNode!);
        } else {
          // Reconnect direct path
          audioWorkletNode.connect(outputGainNode!);
        }
        bypassNode.gain.setValueAtTime(0, audioContext.currentTime);
      }

      setAudioState(prev => ({ ...prev, isBypassed: !isBypassed }));
    } catch (error) {
      console.error('Error toggling bypass:', error);
    }
  }, [getAudioNodes]);

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
    const { audioElement, mediaStream, liveSourceNode } = nodes;

    // Stop file playback
    if (audioElement) {
      audioElement.pause();
      audioElement.currentTime = 0;
    }

    // Stop live input
    if (mediaStream) {
      mediaStream.getTracks().forEach(track => track.stop());
      nodes.mediaStream = null;
    }
    if (liveSourceNode) {
      liveSourceNode.disconnect();
      nodes.liveSourceNode = null;
    }

    removeIr();

    setAudioState(prev => ({
      ...prev,
      modelUrl: null,
      audioUrl: null,
      isBypassed: false,
      inputMode: { type: 'preview' },
    }));
  }, [getAudioNodes, removeIr]);

  // Enumerate available audio input devices
  const enumerateInputDevices = useCallback(async (): Promise<AudioInputDevice[]> => {
    try {
      // Request permission first to get labeled devices
      await navigator.mediaDevices.getUserMedia({ audio: true });
      
      const devices = await navigator.mediaDevices.enumerateDevices();
      const audioInputs = devices
        .filter(device => device.kind === 'audioinput')
        .map(device => ({
          deviceId: device.deviceId,
          label: device.label || `Input ${device.deviceId.slice(0, 8)}`,
        }));
      
      return audioInputs;
    } catch (error) {
      console.error('Error enumerating audio devices:', error);
      throw error;
    }
  }, []);

  // Start live audio input from microphone/audio interface
  const startLiveInput = useCallback(async (deviceId?: string): Promise<void> => {
    const nodes = getAudioNodes();
    const { audioContext, inputMeterNode, sourceNode, audioElement } = nodes;

    if (!audioContext || !inputMeterNode) {
      throw new Error('Audio system not initialized. Call init() first.');
    }

    setAudioState(prev => ({ ...prev, isLiveConnecting: true }));

    try {
      // Stop any existing live input
      if (nodes.mediaStream) {
        nodes.mediaStream.getTracks().forEach(track => track.stop());
        nodes.liveSourceNode?.disconnect();
        nodes.liveInputGainNode?.disconnect();
        nodes.channelSplitterNode?.disconnect();
        nodes.channelMergerNode?.disconnect();
      }

      // Stop file playback if active
      if (audioElement) {
        audioElement.pause();
      }

      // Request microphone/audio interface access with settings optimized for instruments
      // Request stereo to support multi-channel interfaces
      const constraints: MediaStreamConstraints = {
        audio: {
          deviceId: deviceId ? { exact: deviceId } : undefined,
          channelCount: { ideal: 2 },  // Request stereo if available
          // Disable processing that would color the guitar/mic signal
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
        },
      };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      nodes.mediaStream = stream;

      // Get actual channel count from the track
      const track = stream.getAudioTracks()[0];
      const settings = track.getSettings();
      const channelCount = settings.channelCount ?? 1;

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
      const meterConfig = { fftSize: 256, smoothingTimeConstant: 0.3 };
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
      nodes.channelSplitterNode.connect(nodes.channelMergerNode, 0, 0);
      nodes.channelMergerNode.connect(inputMeterNode);

      setAudioState(prev => ({
        ...prev,
        isLiveConnecting: false,
        inputMode: {
          type: 'live',
          deviceId: deviceId,
          channelCount: channelCount,
          selectedChannel: 'first',
          channelGains: { first: 0, second: 0 },
        },
      }));

      // Resume audio context if suspended
      if (audioContext.state === 'suspended') {
        await audioContext.resume();
      }
    } catch (error) {
      setAudioState(prev => ({ ...prev, isLiveConnecting: false }));
      console.error('Error starting live input:', error);
      throw error;
    }
  }, [getAudioNodes]);

  // Stop live audio input
  const stopLiveInput = useCallback((): void => {
    const nodes = getAudioNodes();
    const { mediaStream, liveSourceNode, sourceNode, inputGainNode } = nodes;

    if (mediaStream) {
      // Stop all tracks in the stream
      mediaStream.getTracks().forEach(track => track.stop());
      nodes.mediaStream = null;
    }

    if (liveSourceNode) {
      liveSourceNode.disconnect();
      nodes.liveSourceNode = null;
    }

    // Clean up live input gain node
    if (nodes.liveInputGainNode) {
      nodes.liveInputGainNode.disconnect();
      nodes.liveInputGainNode = null;
    }

    // Clean up channel selection nodes
    if (nodes.channelSplitterNode) {
      nodes.channelSplitterNode.disconnect();
      nodes.channelSplitterNode = null;
    }
    if (nodes.channelMergerNode) {
      nodes.channelMergerNode.disconnect();
      nodes.channelMergerNode = null;
    }
    if (nodes.channel0PreviewMeter) {
      nodes.channel0PreviewMeter.disconnect();
      nodes.channel0PreviewMeter = null;
    }
    if (nodes.channel1PreviewMeter) {
      nodes.channel1PreviewMeter.disconnect();
      nodes.channel1PreviewMeter = null;
    }

    // Reconnect file source if available
    if (sourceNode && inputGainNode) {
      try {
        sourceNode.connect(inputGainNode);
      } catch {
        // Source might already be connected
      }
    }

    setAudioState(prev => ({
      ...prev,
      inputMode: { type: 'preview' },
    }));
  }, [getAudioNodes]);

  // Set input mode (switches between preview and live)
  const setInputMode = useCallback((mode: InputMode): void => {
    const nodes = getAudioNodes();
    
    if (mode.type === 'live' && !nodes.mediaStream) {
      // If switching to live but no stream, caller should use startLiveInput
      console.warn('No live input stream. Call startLiveInput() to enable live input.');
      return;
    }

    if (mode.type === 'preview') {
      stopLiveInput();
    }

    setAudioState(prev => ({ ...prev, inputMode: mode }));
  }, [getAudioNodes, stopLiveInput]);

  // Check if live input stream is actually active (hardware state)
  const isLiveInputActive = useCallback((): boolean => {
    const { mediaStream } = getAudioNodes();
    return mediaStream !== null && mediaStream.active;
  }, [getAudioNodes]);

  // Select which channel to route from live input to processing
  const selectLiveInputChannel = useCallback((channel: ChannelSelection): void => {
    const nodes = getAudioNodes();
    const { channelSplitterNode, channelMergerNode, inputGainNode, liveInputGainNode, audioContext } = nodes;

    if (!channelSplitterNode || !channelMergerNode || !inputGainNode) {
      console.warn('Cannot select channel: live input not active');
      return;
    }

    // Get the currently selected channel from state
    const currentChannel = audioState.inputMode.type === 'live'
      ? audioState.inputMode.selectedChannel
      : undefined;

    // Convert channel names to indices
    const newChannelIndex = channel === 'first' ? 0 : 1;
    const currentChannelIndex = currentChannel === 'first' ? 0 : currentChannel === 'second' ? 1 : undefined;

    // Disconnect current channel if different
    if (currentChannelIndex !== undefined && currentChannelIndex !== newChannelIndex) {
      channelSplitterNode.disconnect(channelMergerNode, currentChannelIndex, 0);
    }

    // Connect new channel (only if not already connected)
    if (currentChannelIndex !== newChannelIndex) {
      channelSplitterNode.connect(channelMergerNode, newChannelIndex, 0);
    }

    // Apply the gain setting for the new channel
    if (liveInputGainNode && audioContext && audioState.inputMode.type === 'live') {
      const channelGains = audioState.inputMode.channelGains;
      const gainDb = channelGains?.[channel] ?? 0;
      const linearGain = dbToLinear(gainDb);
      liveInputGainNode.gain.setTargetAtTime(
        linearGain,
        audioContext.currentTime,
        0.02
      );
    }

    // Update state
    setAudioState(prev => {
      if (prev.inputMode.type !== 'live') return prev;
      return {
        ...prev,
        inputMode: {
          ...prev.inputMode,
          selectedChannel: channel,
        },
      };
    });
  }, [getAudioNodes, audioState.inputMode]);

  // Set live input gain for the currently selected channel (persists to state)
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

    // Persist to state for the currently selected channel
    setAudioState(prev => {
      if (prev.inputMode.type !== 'live') return prev;
      const selectedChannel = prev.inputMode.selectedChannel ?? 'first';
      return {
        ...prev,
        inputMode: {
          ...prev.inputMode,
          channelGains: {
            ...prev.inputMode.channelGains,
            [selectedChannel]: gainDb,
          } as Record<ChannelSelection, number>,
        },
      };
    });
  }, [getAudioNodes]);

  // Set playing state (controls audioElement in preview mode, outputGainNode in live mode)
  const setPlaying = useCallback((playing: boolean): void => {
    const nodes = getAudioNodes();
    const { audioElement, outputGainNode, audioContext } = nodes;

    // Update audio nodes based on current input mode
    if (audioState.inputMode.type === 'live') {
      // Live mode: control output gain (mute/unmute)
      if (outputGainNode && audioContext) {
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
          audioContext?.resume();
          audioElement.play().catch(console.error);
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

    setAudioState(prev => ({ ...prev, isPlaying: playing }));
  }, [getAudioNodes, audioState.inputMode.type]);

  // Save current live config to snapshot (for restoring later)
  const saveLiveSnapshot = useCallback((): void => {
    setAudioState(prev => {
      if (prev.inputMode.type !== 'live' || !prev.inputMode.deviceId) {
        return prev;
      }
      return {
        ...prev,
        liveConfigSnapshot: {
          deviceId: prev.inputMode.deviceId,
          channel: prev.inputMode.selectedChannel ?? 'first',
          channelGains: prev.inputMode.channelGains ?? { first: 0, second: 0 },
          wasPlaying: prev.isPlaying,
          wasBypassed: prev.isBypassed,
        },
      };
    });
  }, []);

  // Restore live config from snapshot (handles async reconnection)
  // Options:
  //   includePlaybackState: if true, also restore playing and bypass state
  //                         (use true for dialog cancel, false for source mode switching)
  const restoreLiveSnapshot = useCallback(async (options?: { includePlaybackState?: boolean }): Promise<void> => {
    const snapshot = audioState.liveConfigSnapshot;
    if (!snapshot) return;

    const { deviceId, channel, channelGains, wasPlaying, wasBypassed } = snapshot;
    const includePlaybackState = options?.includePlaybackState ?? false;

    // Reconnect to device (async)
    await startLiveInput(deviceId);

    // Restore channel gains to state
    setAudioState(prev => {
      if (prev.inputMode.type !== 'live') return prev;
      return {
        ...prev,
        inputMode: {
          ...prev.inputMode,
          channelGains,
        },
      };
    });

    // Select channel and apply gain (using snapshot values directly to avoid race condition)
    selectLiveInputChannel(channel);
    setLiveInputGain(channelGains[channel] ?? 0);

    // Only restore playback/bypass state if requested (e.g., dialog cancel)
    if (includePlaybackState) {
      setPlaying(wasPlaying);
      if (audioState.isBypassed !== wasBypassed) {
        toggleBypass();
      }
    }
  }, [audioState.liveConfigSnapshot, audioState.isBypassed, startLiveInput, selectLiveInputChannel, setPlaying, toggleBypass]);

  // Clear the snapshot (after successful connect or when no longer needed)
  const clearLiveSnapshot = useCallback((): void => {
    setAudioState(prev => ({ ...prev, liveConfigSnapshot: null }));
  }, []);

  // Memoize context value
  const contextValue = useMemo<T3kPlayerContextType>(
    () => ({
      audioState,
      microphonePermission,
      audioInputDevices,
      getAudioNodes,
      init,
      loadModel,
      loadAudio,
      loadIr,
      removeIr,
      toggleBypass,
      cleanup,
      connectVisualizerNode,
      enumerateInputDevices,
      startLiveInput,
      stopLiveInput,
      setInputMode,
      isLiveInputActive,
      selectLiveInputChannel,
      setLiveInputGain,
      setPlaying,
      saveLiveSnapshot,
      restoreLiveSnapshot,
      clearLiveSnapshot,
      requestMicrophonePermission,
      refreshAudioInputDevices,
    }),
    [
      audioState,
      microphonePermission,
      audioInputDevices,
      getAudioNodes,
      init,
      loadModel,
      loadAudio,
      loadIr,
      removeIr,
      toggleBypass,
      cleanup,
      connectVisualizerNode,
      enumerateInputDevices,
      startLiveInput,
      stopLiveInput,
      setInputMode,
      isLiveInputActive,
      selectLiveInputChannel,
      setLiveInputGain,
      setPlaying,
      saveLiveSnapshot,
      restoreLiveSnapshot,
      clearLiveSnapshot,
      requestMicrophonePermission,
      refreshAudioInputDevices,
    ]
  );

  return (
    <T3kPlayerContext.Provider value={contextValue}>
      {children}
    </T3kPlayerContext.Provider>
  );
}

// Custom hook with SSR support
export const useT3kPlayerContext = () => {
  if (typeof window === 'undefined') {
    // Return SSR-safe defaults
    return {
      audioState: {
        initState: 'uninitialized' as AudioInitState,
        isPlaying: false,
        isBypassed: false,
        isLiveConnecting: false,
        modelUrl: null,
        irUrl: null,
        audioUrl: null,
        inputMode: { type: 'preview' } as InputMode,
        liveConfigSnapshot: null,
      },
      microphonePermission: {
        status: 'idle' as const,
        error: null,
      },
      audioInputDevices: {
        devices: [],
        isLoading: false,
        error: null,
        preferredDeviceId: null,
      },
      getAudioNodes: () => ({
        audioContext: null,
        audioElement: null,
        audioWorkletNode: null,
        inputGainNode: null,
        outputGainNode: null,
        bypassNode: null,
        irNode: null,
        irWetGain: null,
        irDryGain: null,
        irGain: null,
        sourceNode: null,
        liveSourceNode: null,
        liveInputGainNode: null,
        mediaStream: null,
        inputMeterNode: null,
        outputMeterNode: null,
        limiterNode: null,
        channelSplitterNode: null,
        channelMergerNode: null,
        channel0PreviewMeter: null,
        channel1PreviewMeter: null,
      }),
      init: async () => {},
      loadModel: async () => {},
      loadAudio: async () => {},
      loadIr: async () => {},
      removeIr: () => {},
      toggleBypass: () => {},
      cleanup: () => {},
      connectVisualizerNode: () => () => {},
      enumerateInputDevices: async () => [],
      startLiveInput: async () => {},
      stopLiveInput: () => {},
      setInputMode: () => {},
      isLiveInputActive: () => false,
      selectLiveInputChannel: () => {},
      setLiveInputGain: () => {},
      setPlaying: () => {},
      saveLiveSnapshot: () => {},
      restoreLiveSnapshot: async () => {},
      clearLiveSnapshot: () => {},
      requestMicrophonePermission: async () => null,
      refreshAudioInputDevices: async () => {},
    };
  }

  const context = useContext(T3kPlayerContext);

  if (!context) {
    throw new Error(
      'useT3kPlayerContext must be used within a T3kPlayerContextProvider'
    );
  }

  return context;
};
