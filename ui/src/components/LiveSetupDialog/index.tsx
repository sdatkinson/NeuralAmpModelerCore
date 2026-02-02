import React, { useState, useCallback, useEffect, useRef } from 'react';
import { ArrowLeft } from 'lucide-react';
import { Dialog } from '../ui/Dialog';
import { Button } from '../ui/Button';
import { LivePermissionContent } from './content/LivePermissionContent';
import { LiveDeviceChannelContent } from './content/LiveDeviceChannelContent';
import { useT3kPlayerContext } from '../../context/T3kPlayerContext';
import { ChannelSelection } from '../../types';
import { dbToLinear } from '../../utils/metering';

interface LiveSetupDialogProps {
  isOpen: boolean;
  onClose: () => void;
  onConnect?: (deviceId: string, channel: ChannelSelection) => void;
}

type SetupStep = 'permission' | 'device-channel-select';

export const LiveSetupDialog: React.FC<LiveSetupDialogProps> = ({
  isOpen,
  onClose,
  onConnect,
}) => {
  const {
    microphonePermission,
    audioInputDevices,
    audioState,
    init,
    requestMicrophonePermission,
    refreshAudioInputDevices,
    startLiveInput,
    stopLiveInput,
    selectLiveInputChannel,
    toggleBypass,
    getAudioNodes,
  } = useT3kPlayerContext();

  // Local state: only what's needed for dialog UI flow
  const [step, setStep] = useState<SetupStep>('permission');
  const [isInitializing, setIsInitializing] = useState(true);
  const [isConnectingDevice, setIsConnectingDevice] = useState(false);
  const [isMonitoring, setIsMonitoring] = useState(false); // Local monitoring state for preview
  const [inputGain, setInputGain] = useState(0); // Input gain in dB
  const connectedRef = useRef(false); // Track if user clicked Connect

  // Derived from context (single source of truth)
  const liveInputMode = audioState.inputMode.type === 'live' ? audioState.inputMode : null;
  const currentDeviceId = liveInputMode?.deviceId ?? null;
  const currentChannel = liveInputMode?.selectedChannel ?? 'first';
  const channelCount = liveInputMode?.channelCount ?? 1;

  // Dialog open/close lifecycle
  useEffect(() => {
    if (isOpen) {
      connectedRef.current = false;
      setIsMonitoring(false); // Reset monitoring state
      setIsInitializing(true);
      refreshAudioInputDevices().then(() => {
        setIsInitializing(false);
        setStep(microphonePermission.status === 'granted' ? 'device-channel-select' : 'permission');
      });
    } else if (!connectedRef.current) {
      // Dialog closed without Connect - stop preview and restore bypass state
      stopLiveInput();
      if (audioState.isBypassed) toggleBypass();
    }
  }, [isOpen]);

  // Helper to ensure audio system is initialized before starting live input
  const initAndStartLiveInput = useCallback(async (deviceId: string) => {
    setIsConnectingDevice(true);
    try {
      if (audioState.initState !== 'ready') {
        // init() fully initializes the audio system:
        // loads WASM, loads default model, waits for audio nodes to be created
        await init();
      }
      await startLiveInput(deviceId);

      // Enable bypass mode for preview (disconnects NAM worklet from output)
      // This way we only hear dry signal (or silence) during setup
      if (!audioState.isBypassed) {
        toggleBypass();
      }

      // Set bypass gain based on current monitoring state (default: 0 = silence)
      const nodes = getAudioNodes();
      if (nodes.bypassNode && nodes.audioContext) {
        nodes.bypassNode.gain.setValueAtTime(isMonitoring ? 1 : 0, nodes.audioContext.currentTime);
      }
    } finally {
      setIsConnectingDevice(false);
    }
  }, [audioState.initState, audioState.isBypassed, init, startLiveInput, toggleBypass, getAudioNodes, isMonitoring]);

  // NOTE: We intentionally do NOT auto-start live input here.
  // AudioContext creation requires a user gesture (browser autoplay policy).
  // The user must explicitly click a device in the dropdown to trigger initialization.

  // Event handlers - directly call context methods
  const handleDeviceChange = useCallback((deviceId: string) => {
    initAndStartLiveInput(deviceId);
  }, [initAndStartLiveInput]);

  const handleChannelChange = useCallback((channel: ChannelSelection) => {
    selectLiveInputChannel(channel);
  }, [selectLiveInputChannel]);

  const handleMonitoringChange = useCallback((enabled: boolean) => {
    // During preview, we stay in bypass mode (worklet disconnected)
    // Monitor ON (checked) → bypass gain = 1 → hear dry signal
    // Monitor OFF (unchecked) → bypass gain = 0 → silence
    setIsMonitoring(enabled);
    const nodes = getAudioNodes();
    if (nodes.bypassNode && nodes.audioContext) {
      nodes.bypassNode.gain.setValueAtTime(enabled ? 1 : 0, nodes.audioContext.currentTime);
    }
  }, [getAudioNodes]);

  const handleInputGainChange = useCallback((gainDb: number) => {
    setInputGain(gainDb);
    const nodes = getAudioNodes();
    if (nodes.inputGainNode && nodes.audioContext) {
      const linearGain = dbToLinear(gainDb);
      nodes.inputGainNode.gain.setTargetAtTime(
        linearGain,
        nodes.audioContext.currentTime,
        0.02 // 20ms smoothing
      );
    }
  }, [getAudioNodes]);

  const handleRequestPermission = useCallback(async () => {
    try {
      await requestMicrophonePermission();
      setStep('device-channel-select');
    } catch {
      // Error handling is done in context
    }
  }, [requestMicrophonePermission]);

  const handleConnect = () => {
    if (currentDeviceId) {
      connectedRef.current = true;

      // Exit bypass mode so the NAM processing is connected to output
      // (during preview we were in bypass mode to hear dry signal)
      if (audioState.isBypassed) {
        toggleBypass();
      }

      // Reset bypass gain to 1 in case monitoring was off (gain was 0)
      const nodes = getAudioNodes();
      if (nodes.bypassNode && nodes.audioContext) {
        nodes.bypassNode.gain.setValueAtTime(1, nodes.audioContext.currentTime);
      }

      onConnect?.(currentDeviceId, currentChannel);
    }
    handleClose();
  };

  const handleBack = () => {
    handleClose();
  };

  const handleClose = () => {
    onClose();
  };

  const isPending = microphonePermission.status === 'pending';

  // Header with back button
  const header = (
    <div className='flex items-center gap-3 p-4'>
      <button
        onClick={handleBack}
        className='p-1 hover:bg-zinc-800 rounded-md transition-colors'
        aria-label='Back'
        disabled={isPending}
      >
        <ArrowLeft size={20} className='text-zinc-400' />
      </button>
      <h2 className='text-lg font-semibold'>Live Input Setup</h2>
    </div>
  );

  // Footer only for device/channel selection step
  const footer = !isInitializing && step === 'device-channel-select' ? (
    <div className='flex justify-end gap-3 p-4'>
      <Button variant='ghost' onClick={handleClose}>
        Cancel
      </Button>
      <Button
        variant='primary'
        onClick={handleConnect}
        disabled={!currentDeviceId}
      >
        Connect
      </Button>
    </div>
  ) : null;

  // Skeleton content while checking permission state
  const skeletonContent = (
    <div className='flex flex-col gap-4'>
      {/* Text skeleton */}
      <div className='flex flex-col gap-2'>
        <div className='h-4 bg-zinc-800 rounded animate-pulse w-full' />
        <div className='h-4 bg-zinc-800 rounded animate-pulse w-3/4' />
      </div>
      {/* Button skeleton */}
      <div className='h-12 bg-zinc-800 rounded animate-pulse w-full' />
    </div>
  );

  return (
    <Dialog
      isOpen={isOpen}
      onClose={handleClose}
      header={header}
      footer={footer}
      closeOnBackdropClick={!isPending && !isInitializing}
    >
      {isInitializing ? (
        skeletonContent
      ) : step === 'permission' ? (
        <LivePermissionContent
          status={microphonePermission.status}
          errorMessage={microphonePermission.error}
          onRequestPermission={handleRequestPermission}
        />
      ) : (
        <LiveDeviceChannelContent
          devices={audioInputDevices.devices}
          selectedDeviceId={currentDeviceId ?? ''}
          selectedChannel={currentChannel}
          channelCount={channelCount}
          isMonitoring={isMonitoring}
          inputGain={inputGain}
          onDeviceChange={handleDeviceChange}
          onChannelChange={handleChannelChange}
          onMonitoringChange={handleMonitoringChange}
          onInputGainChange={handleInputGainChange}
          isLoading={audioInputDevices.isLoading}
          isConnecting={isConnectingDevice}
          error={audioInputDevices.error}
          onRefresh={refreshAudioInputDevices}
          channel0Meter={getAudioNodes().channel0PreviewMeter}
          channel1Meter={getAudioNodes().channel1PreviewMeter}
        />
      )}
    </Dialog>
  );
};

export default LiveSetupDialog;
