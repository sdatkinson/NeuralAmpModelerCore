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
import { InputMode, LiveInputConfig, AudioInputDevice, AudioOutputDevice, MicrophonePermissionStatus, MicrophonePermissionState, AudioInputDeviceState, AudioOutputDeviceState, ChannelSelection, SettingsSnapshot, SettingsDialogState, SourceMode, Model, IR } from '../types';
import { dbToLinear } from '../utils/metering';
import { showToast } from '../hooks/useToast';
import { isFirefox, isSafari, needsMediaStreamWorkaround } from '../utils/browser';
import { mapDevices } from '../utils/devices';

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
  // Channel selection (for multi-channel interfaces)
  channelSplitterNode: ChannelSplitterNode | null;
  channelMergerNode: ChannelMergerNode | null;
  channel0PreviewMeter: AnalyserNode | null;
  channel1PreviewMeter: AnalyserNode | null;
  // Firefox audio output workaround
  firefoxOutputDestination: MediaStreamAudioDestinationNode | null;
  firefoxOutputElement: HTMLAudioElement | null;
}

// Explicit initialization states for visibility into the init process
type AudioInitState = 'uninitialized' | 'initializing' | 'ready';

interface AudioState {
  initState: AudioInitState;
  isPlaying: boolean;  // Whether audio is playing (preview) or monitoring (live)
  activePlayerId: string | null;  // Which player is currently controlling playback
  isBypassed: boolean;
  modelUrl: string | null;
  irUrl: string | null;
  audioUrl: string | null;
  // What audio source is currently active (connected to audio engine)
  inputMode: InputMode;
  // Configured live input settings (persists even when preview is active)
  // This allows UI to show configured device while another player uses file playback
  liveInputConfig: LiveInputConfig | null;
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
  clearLiveInputConfig: () => void;  // Explicitly clear saved config (used when user disconnects in settings)
  setInputMode: (mode: InputMode) => void;
  isLiveInputActive: () => boolean;
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

  // Microphone permission state (permission concerns only)
  const [microphonePermission, setMicrophonePermission] = useState<MicrophonePermissionState>({
    status: 'idle',
    error: null,
  });
  const microphonePermissionRef = useRef(microphonePermission);
  microphonePermissionRef.current = microphonePermission;

  // Audio input device state (device concerns only)
  const [audioInputDevices, setAudioInputDevices] = useState<AudioInputDeviceState>({
    devices: [],
    isLoading: false,
    error: null,
    preferredDeviceId: null,
  });

  // Audio output device state
  const [audioOutputDevices, setAudioOutputDevices] = useState<AudioOutputDeviceState>({
    devices: [],
    selectedDeviceId: null,  // null = system default
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
    // Channel selection
    channelSplitterNode: null,
    channelMergerNode: null,
    channel0PreviewMeter: null,
    channel1PreviewMeter: null,
    // Firefox audio output workaround
    firefoxOutputDestination: null,
    firefoxOutputElement: null,
  });

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

  // Helper: Clean up all live input nodes (mediaStream, source, gain, splitter, merger, meters)
  // This is used by startLiveInput, stopLiveInput, cleanup, and event handlers
  const cleanupLiveInputNodes = (nodes: AudioNodes): void => {
    if (nodes.mediaStream) {
      nodes.mediaStream.getTracks().forEach(track => track.stop());
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
  };

  // Helper: Clean up Firefox output routing and reconnect outputMeterNode to default destination
  // This is used when transitioning away from Firefox's HTMLAudioElement workaround
  const cleanupFirefoxOutputRouting = (nodes: AudioNodes): void => {
    // Track if Firefox routing was active (need to know for proper cleanup)
    const hadFirefoxRouting = nodes.firefoxOutputElement !== null || nodes.firefoxOutputDestination !== null;

    if (nodes.firefoxOutputElement) {
      nodes.firefoxOutputElement.pause();
      nodes.firefoxOutputElement.srcObject = null;
      nodes.firefoxOutputElement = null;
    }
    if (nodes.firefoxOutputDestination) {
      nodes.firefoxOutputDestination.disconnect();
      nodes.firefoxOutputDestination = null;
    }

    // Reconnect outputMeterNode to destination
    // If Firefox routing was active, outputMeterNode was connected to firefoxOutputDestination,
    // so we need to disconnect it first to avoid having multiple output connections
    if (nodes.outputMeterNode && nodes.audioContext) {
      if (hadFirefoxRouting) {
        // Firefox routing was active - disconnect from old destination before reconnecting
        try {
          nodes.outputMeterNode.disconnect();
        } catch {
          // May not have any connections
        }
        nodes.outputMeterNode.connect(nodes.audioContext.destination);
      } else {
        // No Firefox routing was active - just try to connect (may already be connected)
        try {
          nodes.outputMeterNode.connect(nodes.audioContext.destination);
        } catch {
          // Already connected
        }
      }
    }
  };

  // Helper: Tear down live input audio nodes and restore the preview signal path.
  // Used by stopLiveInput, clearLiveInputConfig, and handleLiveInputUnavailable.
  const teardownLiveInput = (nodes: AudioNodes, options: { muteOutput: boolean }): void => {
    cleanupLiveInputNodes(nodes);
    cleanupFirefoxOutputRouting(nodes);

    // Reconnect file source to restore preview path
    if (nodes.sourceNode && nodes.inputGainNode) {
      try {
        nodes.sourceNode.connect(nodes.inputGainNode);
      } catch {
        // Source might already be connected
      }
    }

    if (options.muteOutput && nodes.outputGainNode && nodes.audioContext) {
      nodes.outputGainNode.gain.setTargetAtTime(0, nodes.audioContext.currentTime, 0.01);
    }
  };

  // Helper: Handle live input becoming unavailable (device disconnected or permission revoked)
  // Cleans up audio nodes, reverts to preview mode, and sets an error message
  type LiveInputUnavailableReason = 'device-disconnected' | 'permission-revoked';

  const handleLiveInputUnavailable = useCallback((reason: LiveInputUnavailableReason): void => {
    teardownLiveInput(audioNodesRef.current, { muteOutput: false });

    // Revert to preview mode and clear live config
    setAudioState(prev => ({
      ...prev,
      inputMode: { type: 'preview' },
      liveInputConfig: null,
      isPlaying: false,
      activePlayerId: null,
    }));

    // Set appropriate error message (persistent, shown in live input UI)
    const errorMessages: Record<LiveInputUnavailableReason, string> = {
      'device-disconnected': 'Audio input device was disconnected.',
      'permission-revoked': 'Microphone permission was revoked. Please re-enable access in your browser settings.',
    };

    setAudioInputDevices(prev => ({
      ...prev,
      error: errorMessages[reason],
    }));

    // Immediate feedback via toast (visible regardless of which player/mode is shown)
    const toastMessages: Record<LiveInputUnavailableReason, string> = {
      'device-disconnected': 'Audio input device disconnected',
      'permission-revoked': 'Microphone permission revoked',
    };
    showToast(toastMessages[reason]);
  }, []);

  // Helper: Apply output device routing with browser-specific handling
  // Firefox and Safari require MediaStreamDestination + HTMLAudioElement workaround
  // Chrome can use AudioContext.setSinkId directly
  const applyOutputDeviceRouting = async (
    nodes: AudioNodes,
    deviceId: string | null
  ): Promise<void> => {
    const { audioContext, outputMeterNode } = nodes;
    if (!audioContext || !outputMeterNode) return;

    if (needsMediaStreamWorkaround) {
      // Firefox/Safari: Must route through MediaStreamDestination + HTMLAudioElement
      // Always clean up existing Firefox audio elements first
      if (nodes.firefoxOutputElement) {
        nodes.firefoxOutputElement.pause();
        nodes.firefoxOutputElement.srcObject = null;
        nodes.firefoxOutputElement = null;
      }
      if (nodes.firefoxOutputDestination) {
        nodes.firefoxOutputDestination.disconnect();
        nodes.firefoxOutputDestination = null;
      }

      // Disconnect ALL outputs from outputMeterNode to ensure clean state
      try {
        outputMeterNode.disconnect();
      } catch {
        // May not have any connections
      }

      if (deviceId) {
        // Specific device selected - route through HTMLAudioElement with setSinkId
        const mediaStreamDestination = audioContext.createMediaStreamDestination();
        nodes.firefoxOutputDestination = mediaStreamDestination;
        outputMeterNode.connect(mediaStreamDestination);

        const outputElement = new Audio();
        outputElement.srcObject = mediaStreamDestination.stream;
        nodes.firefoxOutputElement = outputElement;

        const elementWithSinkId = outputElement as HTMLAudioElement & { setSinkId?: (sinkId: string) => Promise<void> };
        if (typeof elementWithSinkId.setSinkId === 'function') {
          await elementWithSinkId.setSinkId(deviceId);
        }

        await outputElement.play();
      } else {
        // System default selected - reconnect to normal destination
        outputMeterNode.connect(audioContext.destination);
      }
    } else {
      // Chrome: Use AudioContext.setSinkId directly
      const contextWithSinkId = audioContext as AudioContext & { setSinkId?: (sinkId: string) => Promise<void> };
      if (typeof contextWithSinkId.setSinkId === 'function') {
        try {
          await contextWithSinkId.setSinkId(deviceId ?? '');
        } catch (e) {
          console.warn('[Audio] Failed to set AudioContext sink:', e);
        }
      }
    }
  };

  // Query browser's microphone permission state
  const queryBrowserPermission = useCallback(async (): Promise<MicrophonePermissionStatus> => {
    try {
      // Note: 'microphone' requires casting as it's not in all TypeScript definitions
      const result = await navigator.permissions.query({ name: 'microphone' as PermissionName });
      if (result.state === 'granted') return 'granted';
      if (result.state === 'denied') return 'blocked'; // Permanently blocked in browser settings
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
            // Browser 'denied' state means permanently blocked
            setMicrophonePermission(prev => ({
              ...prev,
              status: 'blocked',
              error: 'Microphone access is blocked. Please enable it in your browser settings.',
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
    const previousStatus = microphonePermissionRef.current.status;
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
      const audioInputs = mapDevices(allDevices, 'audioinput');

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
          // Check if permanently blocked via Permissions API
          const permissionState = await queryBrowserPermission();
          // Firefox doesn't support Permissions API for microphone, so queryBrowserPermission
          // returns 'idle'. If we already tried once (status is 'denied') and get denied again,
          // Firefox won't re-prompt — treat as blocked.
          const alreadyDenied = previousStatus === 'denied';
          if (permissionState === 'blocked' || (permissionState === 'idle' && alreadyDenied)) {
            setMicrophonePermission({
              status: 'blocked',
              error: 'Microphone access is blocked. Please enable it in your browser settings.',
            });
          } else {
            // First denial or Permissions API not supported - can retry once
            setMicrophonePermission({
              status: 'denied',
              error: 'Microphone access was denied. Please try again.',
            });
          }
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
  const refreshAudioDevices = useCallback(async (): Promise<{ inputDevices: AudioInputDevice[]; preferredDeviceId: string | null }> => {
    // First check permission state via Permissions API
    const permApiStatus = await queryBrowserPermission();

    // If Permissions API says granted or blocked, trust it
    if (permApiStatus === 'granted' || permApiStatus === 'blocked') {
      setMicrophonePermission(prev => ({
        ...prev,
        status: permApiStatus,
        error: permApiStatus === 'granted' ? null : prev.error,
      }));

      if (permApiStatus === 'blocked') {
        return { inputDevices: [], preferredDeviceId: null };
      }
    }

    // For Firefox (or when Permissions API returns 'idle'), we need to check differently:
    // Try enumerating devices - if we get labels, we have permission
    setAudioInputDevices(prev => ({
      ...prev,
      isLoading: true,
      error: null,
    }));

    try {
      let allDevices = await navigator.mediaDevices.enumerateDevices();
      let audioInputs = allDevices.filter(device => device.kind === 'audioinput');

      // Check if we have device labels (indicates permission was granted)
      // Firefox returns empty labels until getUserMedia is called in this session
      const hasLabels = audioInputs.some(device => device.label && device.label.length > 0);

      if (!hasLabels) {
        // No labels - need to call getUserMedia to get proper device info
        // This happens on Firefox even when Permissions API says 'granted'
        try {
          const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
          stream.getTracks().forEach(t => t.stop());

          // Re-enumerate after getting permission
          allDevices = await navigator.mediaDevices.enumerateDevices();
          audioInputs = allDevices.filter(device => device.kind === 'audioinput');

          setMicrophonePermission({
            status: 'granted',
            error: null,
          });
        } catch (error) {
          // getUserMedia failed - user denied or no devices
          if (error instanceof DOMException &&
              (error.name === 'NotAllowedError' || error.name === 'PermissionDeniedError')) {
            setMicrophonePermission(prev => ({
              ...prev,
              status: 'denied',
              error: null,
            }));
            setAudioInputDevices(prev => ({
              ...prev,
              isLoading: false,
            }));
            return { inputDevices: [], preferredDeviceId: null };
          }
          // Other error - continue with what we have
        }
      } else if (hasLabels) {
        // We have labels, so permission is granted
        setMicrophonePermission(prev => ({
          ...prev,
          status: 'granted',
          error: null,
        }));
      }

      const mappedInputDevices = mapDevices(allDevices, 'audioinput');

      // Also enumerate output devices
      const mappedOutputDevices = mapDevices(allDevices, 'audiooutput');

      setAudioInputDevices(prev => ({
        ...prev,
        devices: mappedInputDevices,
        isLoading: false,
        error: mappedInputDevices.length === 0 ? 'No audio input devices found.' : null,
      }));

      setAudioOutputDevices(prev => ({
        ...prev,
        devices: mappedOutputDevices,
      }));

      return { inputDevices: mappedInputDevices, preferredDeviceId: audioInputDevices.preferredDeviceId };
    } catch {
      setAudioInputDevices(prev => ({
        ...prev,
        isLoading: false,
        error: 'Failed to enumerate audio devices.',
      }));
      return { inputDevices: [], preferredDeviceId: null };
    }
  }, [queryBrowserPermission, audioInputDevices.preferredDeviceId]);

  // Refresh audio output devices only (no microphone permission required)
  // Use this when you only need output device list without triggering permission prompts
  const refreshAudioOutputDevices = useCallback(async (): Promise<void> => {
    try {
      const allDevices = await navigator.mediaDevices.enumerateDevices();
      const mappedOutputDevices = mapDevices(allDevices, 'audiooutput');

      setAudioOutputDevices(prev => ({
        ...prev,
        devices: mappedOutputDevices,
      }));
    } catch (error) {
      console.error('Error enumerating output devices:', error);
    }
  }, []);

  // Listen for device changes (connect/disconnect)
  useEffect(() => {
    if (typeof window === 'undefined' || !navigator.mediaDevices) return;

    const handleDeviceChange = async () => {
      // Only refresh if we already have permission
      if (microphonePermission.status === 'granted') {
        try {
          const allDevices = await navigator.mediaDevices.enumerateDevices();
          const audioInputs = mapDevices(allDevices, 'audioinput');

          // Check if configured device was disconnected
          const configuredDeviceId = audioState.liveInputConfig?.deviceId;
          if (configuredDeviceId) {
            const configuredDeviceStillExists = audioInputs.some(d => d.deviceId === configuredDeviceId);

            if (!configuredDeviceStillExists) {
              console.warn('Configured audio input device was disconnected');
              handleLiveInputUnavailable('device-disconnected');
              // Also update the devices list
              setAudioInputDevices(prev => ({ ...prev, devices: audioInputs }));
              return;
            }
          }

          setAudioInputDevices(prev => ({
            ...prev,
            devices: audioInputs,
            error: audioInputs.length === 0 ? 'All audio devices have been disconnected.' : null,
          }));

          // Update output device list and check if selected was disconnected
          const audioOutputs = mapDevices(allDevices, 'audiooutput');

          const currentOutputId = audioOutputDevices.selectedDeviceId;
          const selectedOutputStillExists = currentOutputId === null ||
            audioOutputs.some(d => d.deviceId === currentOutputId);

          setAudioOutputDevices(prev => ({
            ...prev,
            devices: audioOutputs,
          }));

          // If selected output was disconnected, fall back to system default
          if (!selectedOutputStillExists) {
            console.warn('Selected audio output device was disconnected, falling back to system default');
            setOutputDevice(null);
            showToast('Output switched to default');
          }
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
  }, [microphonePermission.status, audioState.liveInputConfig, audioOutputDevices.selectedDeviceId, handleLiveInputUnavailable]);


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
            const meterConfig = { fftSize: 256, smoothingTimeConstant: 0.3 };
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

    const currentlyBypassed = bypassNode.gain.value === 1;
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
        bypassNode.gain.setValueAtTime(1, audioContext.currentTime);
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
        bypassNode.gain.setValueAtTime(0, audioContext.currentTime);
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
    const { audioElement } = nodes;

    // Stop file playback
    if (audioElement) {
      audioElement.pause();
      audioElement.currentTime = 0;
    }

    // Clean up all live input nodes
    cleanupLiveInputNodes(nodes);

    // Clean up Firefox output routing and reconnect to default destination
    cleanupFirefoxOutputRouting(nodes);

    removeIr();

    setAudioState(prev => ({
      ...prev,
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
    const initialChannel = options?.initialChannel ?? 'first';
    const initialChannelGains = options?.initialChannelGains ?? { first: 0, second: 0 };
    const initialChannelIndex = initialChannel === 'first' ? 0 : 1;

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

      // Re-apply output device routing (handles Firefox workaround automatically)
      // This is needed after changing input sources to maintain proper audio routing
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
  const clearLiveInputConfig = useCallback((): void => {
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

  // Set output device for audio playback
  const setOutputDevice = useCallback(async (deviceId: string | null): Promise<void> => {
    const nodes = getAudioNodes();
    const { audioContext, outputMeterNode } = nodes;
    const previousDeviceId = audioOutputDevices.selectedDeviceId;

    // Update state optimistically
    setAudioOutputDevices(prev => ({
      ...prev,
      selectedDeviceId: deviceId,
    }));

    // If no audio context or output meter node yet, just save the preference
    if (!audioContext || !outputMeterNode) {
      return;
    }

    // Apply output routing (handles Firefox workaround automatically)
    try {
      await applyOutputDeviceRouting(nodes, deviceId);
    } catch (error) {
      console.warn('[Audio] Failed to set output device, reverting:', error);
      setAudioOutputDevices(prev => ({ ...prev, selectedDeviceId: previousDeviceId }));
    }
  }, [getAudioNodes, audioOutputDevices.selectedDeviceId]);

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
        // Resume context if suspended (required for Firefox)
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
  }, [audioState.liveInputConfig, clearLiveInputConfig, startLiveInput, setPlaying, setOutputDevice, setBypass]);

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
      clearLiveInputConfig,
      setInputMode,
      isLiveInputActive,
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
      clearLiveInputConfig,
      setInputMode,
      isLiveInputActive,
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

// Custom hook with SSR support
export const useT3kPlayerContext = () => {
  if (typeof window === 'undefined') {
    // Return SSR-safe defaults
    return {
      audioState: {
        initState: 'uninitialized' as AudioInitState,
        isPlaying: false,
        activePlayerId: null,
        isBypassed: false,
        modelUrl: null,
        irUrl: null,
        audioUrl: null,
        inputMode: { type: 'preview' } as InputMode,
        liveInputConfig: null,
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
      audioOutputDevices: {
        devices: [],
        selectedDeviceId: null,
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
        channelSplitterNode: null,
        channelMergerNode: null,
        channel0PreviewMeter: null,
        channel1PreviewMeter: null,
        firefoxOutputDestination: null,
        firefoxOutputElement: null,
      }),
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
      clearLiveInputConfig: () => {},
      setInputMode: () => {},
      isLiveInputActive: () => false,
      selectLiveInputChannel: () => {},
      setLiveInputGain: () => {},
      setOutputDevice: async () => {},
      setPlaying: (() => {}) as T3kPlayerContextType['setPlaying'],
      requestMicrophonePermission: async () => null,
      refreshAudioDevices: async () => ({ inputDevices: [], preferredDeviceId: null }),
      refreshAudioOutputDevices: async () => {},
      settingsDialog: {
        isOpen: false,
        sourceMode: 'preview',
        snapshot: null,
        hadExistingConfig: false,
      } as SettingsDialogState,
      openSettingsDialog: () => {},
      closeSettingsDialog: () => {},
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
