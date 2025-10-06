import React, {
  createContext,
  useContext,
  useRef,
  useState,
  ReactNode,
  useCallback,
  useMemo,
} from 'react';
import { useModule } from '../hooks/useModule';
import { readModel } from '../utils/readModel';
import { DEFAULT_AUDIO_SRC } from '../constants';

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
}

interface AudioState {
  isInitialized: boolean;
  isBypassed: boolean;
  modelUrl: string | null;
  irUrl: string | null;
  audioUrl: string | null;
}

interface IrConfig {
  url: string;
  wetAmount?: number;
  gainAmount?: number;
}

interface T3kPlayerContextType {
  // State
  audioState: AudioState;

  // Getters
  getAudioNodes: () => AudioNodes;

  // Actions
  init: ({ audioUrl }: { audioUrl: string }) => Promise<void>;
  loadModel: (modelUrl: string) => Promise<void>;
  loadAudio: (src: string) => Promise<void>;
  loadIr: (config: IrConfig) => Promise<void>;
  removeIr: () => void;
  toggleBypass: () => void;
  cleanup: () => void;
  connectVisualizerNode: (analyserNode: AnalyserNode) => () => void;
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
    isInitialized: false,
    isBypassed: false,
    modelUrl: null,
    irUrl: null,
    audioUrl: null,
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
  });

  const isInitializingRef = useRef<boolean>(false);
  const modulePromise = useModule();

  // Getter for audio nodes
  const getAudioNodes = useCallback((): AudioNodes => {
    return audioNodesRef.current;
  }, []);

  // Initialize audio system
  const init = useCallback(
    async ({ audioUrl }: { audioUrl: string }): Promise<void> => {
      if (audioState.isInitialized || isInitializingRef.current) return;

      isInitializingRef.current = true;

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
        audio.src = audioUrl || DEFAULT_AUDIO_SRC;
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

        // Load WASM module script
        await new Promise<void>((resolve, reject) => {
          const script = document.createElement('script');
          script.src = '/t3k-wasm-module.js';
          script.async = true;
          script.onload = () => resolve();
          script.onerror = error => {
            console.error('Failed to load t3k-wasm-module.js:', error);
            reject(new Error('Failed to load audio module'));
          };
          document.body.appendChild(script);
        });

        // Setup audio worklet callback
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

          // Create source from audio element
          nodes.sourceNode = context.createMediaElementSource(
            nodes.audioElement!
          );

          // Connect audio graph
          nodes.sourceNode.connect(nodes.inputGainNode);
          nodes.inputGainNode.connect(nodes.bypassNode);
          nodes.bypassNode.connect(nodes.outputGainNode);
          nodes.inputGainNode.connect(audioWorkletNode);
          audioWorkletNode.connect(nodes.outputGainNode);
          nodes.outputGainNode.connect(context.destination);

          context.resume();
          setAudioState(prev => ({ ...prev, isInitialized: true }));
        };
      } catch (error) {
        console.error('Error initializing audio system:', error);
        throw error;
      } finally {
        isInitializingRef.current = false;
      }
    },
    [audioState.isInitialized, getAudioNodes]
  );

  // Load model
  const loadModel = useCallback(
    async (modelUrl: string): Promise<void> => {
      if (isInitializingRef.current) {
        throw new Error('Audio system is initializing');
      }

      try {
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
        const ptr = module._malloc(jsonStr.length + 1);
        module.stringToUTF8(jsonStr, ptr, jsonStr.length + 1);

        try {
          const context = getAudioNodes().audioContext;

          // Suspend context during model loading
          if (context?.state === 'running') {
            await context.suspend();
          }

          // Load DSP
          await module.ccall('setDsp', null, ['number'], [ptr], {
            async: true,
          });
          module._free(ptr);

          // Resume context
          if (context?.state === 'suspended') {
            await context.resume();
          }

          setAudioState(prev => ({ ...prev, modelUrl }));
        } catch (error) {
          module._free(ptr);
          throw error;
        }
      } catch (error) {
        console.error('Error loading model:', error);
        throw error;
      }
    },
    [modulePromise, getAudioNodes]
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
    const { audioElement } = getAudioNodes();

    if (audioElement) {
      audioElement.pause();
      audioElement.currentTime = 0;
    }

    removeIr();

    setAudioState(prev => ({
      ...prev,
      modelUrl: null,
      audioUrl: null,
      isBypassed: false,
    }));
  }, [getAudioNodes, removeIr]);

  // Memoize context value
  const contextValue = useMemo<T3kPlayerContextType>(
    () => ({
      audioState,
      getAudioNodes,
      init,
      loadModel,
      loadAudio,
      loadIr,
      removeIr,
      toggleBypass,
      cleanup,
      connectVisualizerNode,
    }),
    [
      audioState,
      getAudioNodes,
      init,
      loadModel,
      loadAudio,
      loadIr,
      removeIr,
      toggleBypass,
      cleanup,
      connectVisualizerNode,
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
        isInitialized: false,
        isBypassed: false,
        modelUrl: null,
        irUrl: null,
        audioUrl: null,
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
      }),
      init: async () => {},
      loadModel: async () => {},
      loadAudio: async () => {},
      loadIr: async () => {},
      removeIr: () => {},
      toggleBypass: () => {},
      cleanup: () => {},
      connectVisualizerNode: () => () => {},
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
