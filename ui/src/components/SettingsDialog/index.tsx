import React, { useState, useCallback, useEffect, useRef } from 'react';
import { ArrowLeft } from 'lucide-react';
import { Dialog } from '../ui/Dialog';
import { Button } from '../ui/Button';
import { PermissionContent } from './content/PermissionContent';
import { DeviceChannelContent } from './content/DeviceChannelContent';
import { useT3kPlayerContext } from '../../context/T3kPlayerContext';
import { ChannelSelection } from '../../types';

type SourceMode = 'preview' | 'live';

interface SettingsDialogProps {
  isOpen: boolean;
  onClose: () => void;
  sourceMode: SourceMode;
  onConnect?: (deviceId: string, channel: ChannelSelection) => void;
}

type SetupStep = 'permission' | 'device-channel-select';

export const SettingsDialog: React.FC<SettingsDialogProps> = ({
  isOpen,
  onClose,
  sourceMode,
  onConnect,
}) => {
  const {
    microphonePermission,
    audioInputDevices,
    audioOutputDevices,
    audioState,
    init,
    requestMicrophonePermission,
    refreshAudioDevices,
    startLiveInput,
    stopLiveInput,
    selectLiveInputChannel,
    setLiveInputGain,
    setOutputDevice,
    setPlaying,
    toggleBypass,
    getAudioNodes,
    saveSettingsSnapshot,
    restoreSettingsSnapshot,
    clearSettingsSnapshot,
  } = useT3kPlayerContext();

  // Local state: only UI flow
  const [step, setStep] = useState<SetupStep>('permission');
  const [isInitializing, setIsInitializing] = useState(true);
  const savedRef = useRef(false);

  // Track the mode when dialog opened (to avoid stale closure issues)
  const openedInLiveModeRef = useRef(false);
  // Track if there was an existing live connection when dialog opened
  const hadExistingConnectionRef = useRef(false);

  // Derived from context (single source of truth)
  const liveInputMode = audioState.inputMode.type === 'live' ? audioState.inputMode : null;
  const currentDeviceId = liveInputMode?.deviceId ?? null;
  const currentChannel = liveInputMode?.selectedChannel ?? 'first';
  const channelCount = liveInputMode?.channelCount ?? 1;
  const currentInputGain = liveInputMode?.channelGains?.[currentChannel] ?? 0;

  const isLiveMode = sourceMode === 'live';
  const isPreviewMode = sourceMode === 'preview';

  // Dialog open/close lifecycle
  useEffect(() => {
    if (isOpen) {
      savedRef.current = false;
      setIsInitializing(true);

      // Capture mode at open time to avoid stale closure
      openedInLiveModeRef.current = sourceMode === 'live';

      // Check if there's an existing live connection (only relevant in live mode)
      const existingLiveMode = audioState.inputMode.type === 'live' ? audioState.inputMode : null;
      hadExistingConnectionRef.current = existingLiveMode !== null && existingLiveMode.deviceId !== undefined;

      // Save snapshot for cancel/restore
      saveSettingsSnapshot({ includeLiveSettings: openedInLiveModeRef.current && hadExistingConnectionRef.current });

      // If permission not granted, show permission step immediately (no loading needed)
      // Only refresh devices if we already have permission
      if (microphonePermission.status === 'granted') {
        refreshAudioDevices().then(({ inputDevices, preferredDeviceId }) => {
          setIsInitializing(false);
          setStep('device-channel-select');

          // In live mode, auto-connect if no device is currently selected
          if (sourceMode === 'live' && !hadExistingConnectionRef.current && inputDevices.length > 0) {
            const deviceToConnect = preferredDeviceId ?? inputDevices[0].deviceId;
            initAndStartLiveInput(deviceToConnect);
          }
        });
      } else {
        setIsInitializing(false);
        setStep('permission');
      }
    } else if (!savedRef.current) {
      // Dialog closed without Save - restore using the mode captured at open time
      restoreSettingsSnapshot({
        includePlaybackState: true,
        includeLiveSettings: openedInLiveModeRef.current
      });
      hadExistingConnectionRef.current = false;
    }
  }, [isOpen]);

  // React to permission changes while dialog is open
  useEffect(() => {
    if (!isOpen || isInitializing) return;

    // If permission is revoked while on device selection, go back to permission step
    if (microphonePermission.status !== 'granted' && step === 'device-channel-select') {
      setStep('permission');
    }
  }, [isOpen, isInitializing, microphonePermission.status, step]);

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

  // Output device change - applied immediately (snapshot handles revert on cancel)
  const handleOutputDeviceChange = useCallback((deviceId: string | null) => {
    setOutputDevice(deviceId);
  }, [setOutputDevice]);

  const handleRequestPermission = useCallback(async () => {
    try {
      const preferredDeviceId = await requestMicrophonePermission();

      // Refresh all devices (input + output) now that we have permission
      await refreshAudioDevices();

      setStep('device-channel-select');

      // In live mode, auto-connect to the device user selected in browser's permission dialog
      if (isLiveMode && preferredDeviceId) {
        initAndStartLiveInput(preferredDeviceId);
      }
    } catch {
      // Error handling is done in context
    }
  }, [requestMicrophonePermission, refreshAudioDevices, initAndStartLiveInput, isLiveMode]);

  const handleSave = () => {
    savedRef.current = true;

    // In live mode with a device selected, call onConnect
    if (isLiveMode && currentDeviceId) {
      onConnect?.(currentDeviceId, currentChannel);
      clearSettingsSnapshot();
    }
    // In preview mode: don't clear snapshot - preserve live settings for mode switching

    handleClose();
  };

  const handleBack = () => {
    handleClose();
  };

  const handleClose = () => {
    onClose();
  };

  const isPending = microphonePermission.status === 'pending';

  // Determine dialog title based on mode
  const dialogTitle = isPreviewMode ? 'Settings' : 'Live Input Setup';

  // Determine if Save should be enabled
  // In preview mode: always enabled (user can save output device change)
  // In live mode: enabled only when a device is selected
  const isSaveEnabled = isPreviewMode || currentDeviceId !== null;

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
      <h2 className='text-lg font-semibold'>{dialogTitle}</h2>
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
        onClick={handleSave}
        disabled={!isSaveEnabled}
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
        <PermissionContent
          status={microphonePermission.status}
          errorMessage={microphonePermission.error}
          onRequestPermission={handleRequestPermission}
          sourceMode={sourceMode}
        />
      ) : (
        <DeviceChannelContent
          sourceMode={sourceMode}
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
          onRefresh={refreshAudioDevices}
          channel0Meter={getAudioNodes().channel0PreviewMeter}
          channel1Meter={getAudioNodes().channel1PreviewMeter}
          outputDevices={audioOutputDevices.devices}
          selectedOutputDeviceId={audioOutputDevices.selectedDeviceId}
          onOutputDeviceChange={handleOutputDeviceChange}
        />
      )}
    </Dialog>
  );
};

export default SettingsDialog;
