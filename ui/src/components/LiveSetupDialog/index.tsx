import React, { useState, useCallback, useEffect, useRef } from 'react';
import { ArrowLeft } from 'lucide-react';
import { Dialog } from '../ui/Dialog';
import { Button } from '../ui/Button';
import { LivePermissionContent } from './content/LivePermissionContent';
import { LiveDeviceChannelContent } from './content/LiveDeviceChannelContent';
import { useT3kPlayerContext } from '../../context/T3kPlayerContext';
import { ChannelSelection } from '../../types';

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
    setLiveInputGain,
    setPlaying,
    toggleBypass,
    getAudioNodes,
    saveLiveSnapshot,
    restoreLiveSnapshot,
    clearLiveSnapshot,
  } = useT3kPlayerContext();

  // Local state: only UI flow
  const [step, setStep] = useState<SetupStep>('permission');
  const [isInitializing, setIsInitializing] = useState(true);
  const connectedRef = useRef(false);

  // Track if there was an existing connection when dialog opened
  const hadExistingConnectionRef = useRef(false);

  // Derived from context (single source of truth)
  const liveInputMode = audioState.inputMode.type === 'live' ? audioState.inputMode : null;
  const currentDeviceId = liveInputMode?.deviceId ?? null;
  const currentChannel = liveInputMode?.selectedChannel ?? 'first';
  const channelCount = liveInputMode?.channelCount ?? 1;
  const currentInputGain = liveInputMode?.channelGains?.[currentChannel] ?? 0;

  // Dialog open/close lifecycle
  useEffect(() => {
    if (isOpen) {
      connectedRef.current = false;
      setIsInitializing(true);

      // Check if there's an existing live connection
      const existingLiveMode = audioState.inputMode.type === 'live' ? audioState.inputMode : null;
      hadExistingConnectionRef.current = existingLiveMode !== null && existingLiveMode.deviceId !== undefined;

      // Save snapshot for cancel/restore
      if (hadExistingConnectionRef.current) {
        saveLiveSnapshot();
      }

      refreshAudioInputDevices().then(() => {
        setIsInitializing(false);
        setStep(microphonePermission.status === 'granted' ? 'device-channel-select' : 'permission');
      });
    } else if (!connectedRef.current) {
      // Dialog closed without Save - restore previous state including playback/bypass
      if (hadExistingConnectionRef.current) {
        restoreLiveSnapshot({ includePlaybackState: true });
      } else {
        // No existing connection: stop live input
        stopLiveInput();
      }
      hadExistingConnectionRef.current = false;
    }
  }, [isOpen]);

  // Helper to ensure audio system is initialized before starting live input
  const initAndStartLiveInput = useCallback(async (deviceId: string) => {
    if (audioState.initState !== 'ready') {
      await init();
    }
    await startLiveInput(deviceId);
  }, [audioState.initState, init, startLiveInput]);

  // Event handlers
  const handleDeviceChange = useCallback((deviceId: string) => {
    initAndStartLiveInput(deviceId);
  }, [initAndStartLiveInput]);

  const handleChannelChange = useCallback((channel: ChannelSelection) => {
    selectLiveInputChannel(channel);
  }, [selectLiveInputChannel]);

  const handleMonitoringChange = useCallback((enabled: boolean) => {
    // Use context's setPlaying - this updates both the audio node and React state
    setPlaying(enabled);
  }, [setPlaying]);

  const handleWetSignalToggle = useCallback(() => {
    // Uses the same bypass state as the player
    toggleBypass();
  }, [toggleBypass]);

  const handleInputGainChange = useCallback((gainDb: number) => {
    setLiveInputGain(gainDb);
  }, [setLiveInputGain]);

  const handleRequestPermission = useCallback(async () => {
    try {
      const preferredDeviceId = await requestMicrophonePermission();
      setStep('device-channel-select');

      // Auto-connect to the device user selected in browser's permission dialog
      if (preferredDeviceId) {
        initAndStartLiveInput(preferredDeviceId);
      }
    } catch {
      // Error handling is done in context
    }
  }, [requestMicrophonePermission, initAndStartLiveInput]);

  const handleConnect = () => {
    if (currentDeviceId) {
      connectedRef.current = true;
      clearLiveSnapshot();  // Commit changes, no need to restore
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
        Save
      </Button>
    </div>
  ) : null;

  // Skeleton content while checking permission state
  const skeletonContent = (
    <div className='flex flex-col gap-4'>
      <div className='flex flex-col gap-2'>
        <div className='h-4 bg-zinc-800 rounded animate-pulse w-full' />
        <div className='h-4 bg-zinc-800 rounded animate-pulse w-3/4' />
      </div>
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
          isMonitoring={audioState.isPlaying}
          isWetSignalEnabled={!audioState.isBypassed}
          inputGain={currentInputGain}
          onDeviceChange={handleDeviceChange}
          onChannelChange={handleChannelChange}
          onMonitoringChange={handleMonitoringChange}
          onWetSignalToggle={handleWetSignalToggle}
          onInputGainChange={handleInputGainChange}
          isLoading={audioInputDevices.isLoading}
          isConnecting={audioState.isLiveConnecting}
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
