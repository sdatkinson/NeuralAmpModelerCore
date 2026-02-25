import React, { useState, useCallback } from 'react';
import { ArrowLeft } from 'lucide-react';
import { Dialog } from '../ui/Dialog';
import { Button } from '../ui/Button';
import { PermissionContent } from './content/PermissionContent';
import { DeviceChannelContent } from './content/DeviceChannelContent';
import { useT3kPlayerContext } from '../../context/T3kPlayerContext';
import { Model, IR, SourceMode } from '../../types';

interface SettingsDialogProps {
  isOpen: boolean;
  onSave: () => void;
  onCancel: () => void;
  sourceMode: SourceMode;
  selectedModel: Model;
  selectedIr: IR;
  playerId?: string;
}

type SetupStep = 'permission' | 'device-channel-select';

export const SettingsDialog: React.FC<SettingsDialogProps> = ({
  isOpen,
  onSave,
  onCancel,
  sourceMode,
  selectedModel,
  selectedIr,
  playerId,
}) => {
  const {
    microphonePermission,
    audioInputDevices,
    audioOutputDevices,
    audioState,
    init,
    requestMicrophonePermission,
    refreshAudioInputDevices,
    refreshAudioOutputDevices,
    startLiveInput,
    reconnectLiveInput,
    selectLiveInputChannel,
    setLiveInputGain,
    syncEngineSettings,
    setOutputDevice,
    setPlaying,
    setBypass,
    getAudioNodes,
  } = useT3kPlayerContext();

  // Local state: only UI flow
  const [step, setStep] = useState<SetupStep>('permission');
  const [isInitializing, setIsInitializing] = useState(true);
  const [connectionError, setConnectionError] = useState<string | null>(null);

  // Derived from context (single source of truth)
  const liveInputConfig = audioState.liveInputConfig;
  const currentDeviceId = liveInputConfig?.deviceId ?? null;
  const currentChannel = liveInputConfig?.selectedChannel ?? 'first';
  const channelCount = liveInputConfig?.channelCount ?? 1;
  const currentInputGain = liveInputConfig?.channelGains?.[currentChannel] ?? 0;

  const isLiveMode = sourceMode === 'live';
  const isPreviewMode = sourceMode === 'preview';

  // Derive effective step: if permission isn't granted while on device-channel-select, show permission.
  // Applies in both modes — preview mode needs permission for output device labels on Firefox/Safari.
  const effectiveStep = (
    !isInitializing &&
    microphonePermission.status !== 'granted' &&
    step === 'device-channel-select'
  ) ? 'permission' : step;

  // Helper to ensure audio system is initialized before starting live input
  const initAndStartLiveInput = useCallback(async (deviceId: string) => {
    setConnectionError(null);
    try {
      if (audioState.initState !== 'ready') {
        await init();
      }
      await startLiveInput(deviceId);
    } catch (error) {
      const message = error instanceof DOMException
        ? error.message
        : 'Failed to connect to audio device. Please try again.';
      setConnectionError(message);
    }
  }, [audioState.initState, init, startLiveInput]);

  // Imperative open handler — called once by Dialog's onOpen (ref-guarded against StrictMode)
  const handleOpen = useCallback(() => {
    setIsInitializing(true);

    if (sourceMode === 'preview') {
      if (microphonePermission.status === 'granted') {
        // Permission granted: enumerate input devices first to trigger getUserMedia
        // self-healing (Firefox/Safari require per-session getUserMedia to unlock device labels),
        // then enumerate output devices so they have labels.
        refreshAudioInputDevices()
          .then(() => refreshAudioOutputDevices())
          .then(() => {
            setIsInitializing(false);
            setStep('device-channel-select');
          });
      } else {
        // No permission: enumerate what we can. effectiveStep will redirect to
        // the permission step so the user can unlock device labels.
        refreshAudioOutputDevices().then(() => {
          setIsInitializing(false);
          setStep('device-channel-select');
        });
      }
    } else if (microphonePermission.status === 'granted') {
      // Live mode with permission: enumerate both input and output devices
      Promise.all([refreshAudioInputDevices(), refreshAudioOutputDevices()])
        .then(([{ inputDevices, preferredDeviceId }]) => {
          setIsInitializing(false);
          setStep('device-channel-select');

          // Auto-connect on first open in live mode (no device configured yet)
          if (!audioState.liveInputConfig && inputDevices.length > 0) {
            const deviceToConnect = preferredDeviceId ?? inputDevices[0].deviceId;
            initAndStartLiveInput(deviceToConnect);
          }
        });
    } else {
      // Live mode without permission: show permission step
      setIsInitializing(false);
      setStep('permission');
    }
  }, [sourceMode, microphonePermission.status, refreshAudioInputDevices, refreshAudioOutputDevices, audioState.liveInputConfig, initAndStartLiveInput]);

  // Event handlers
  const handleDeviceChange = useCallback((deviceId: string) => {
    initAndStartLiveInput(deviceId);
  }, [initAndStartLiveInput]);

  const handleMonitoringChange = useCallback(async (enabled: boolean) => {
    if (enabled) {
      // Reconnect live input if it was torn down (e.g. another player was previewing)
      await reconnectLiveInput();

      if (audioState.initState === 'ready') {
        await syncEngineSettings({
          modelUrl: selectedModel.url,
          ir: { url: selectedIr.url, mix: selectedIr.mix ?? 1, gain: selectedIr.gain ?? 1 },
          bypassed: audioState.isBypassed,
        });
      }

      if (playerId) {
        setPlaying(true, playerId);
      }
    } else {
      setPlaying(false);
    }
  }, [reconnectLiveInput, setPlaying, syncEngineSettings, audioState.initState, audioState.isBypassed, selectedModel, selectedIr, playerId]);

  const handleWetSignalToggle = useCallback(() => {
    setBypass(!audioState.isBypassed);
  }, [setBypass, audioState.isBypassed]);

  const handleRequestPermission = useCallback(async () => {
    try {
      const preferredDeviceId = await requestMicrophonePermission();
      // requestMicrophonePermission only handles the permission prompt.
      // Enumerate devices separately — refreshAudioInputDevices will get labels
      // because getUserMedia just ran (self-healing workaround handles Firefox/Safari).
      await Promise.all([refreshAudioInputDevices(), refreshAudioOutputDevices()]);
      setStep('device-channel-select');

      if (isLiveMode && preferredDeviceId) {
        initAndStartLiveInput(preferredDeviceId);
      }
    } catch {
      // Error handling is done in context
    }
  }, [requestMicrophonePermission, refreshAudioInputDevices, refreshAudioOutputDevices, initAndStartLiveInput, isLiveMode]);

  const handleRefreshDevices = useCallback(async () => {
    await Promise.all([refreshAudioInputDevices(), refreshAudioOutputDevices()]);
  }, [refreshAudioInputDevices, refreshAudioOutputDevices]);

  const isPending = microphonePermission.status === 'pending';
  const dialogTitle = isPreviewMode ? 'Settings' : 'Live Input Setup';
  const isSaveEnabled = isPreviewMode || currentDeviceId !== null;

  // Header with back button
  const header = (
    <div className='flex items-center gap-3 p-4'>
      <button
        onClick={onCancel}
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
  const footer = !isInitializing && effectiveStep === 'device-channel-select' ? (
    <div className='flex justify-end gap-3 p-4'>
      <Button variant='ghost' onClick={onCancel}>
        Cancel
      </Button>
      <Button
        variant='primary'
        onClick={onSave}
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
      onClose={onCancel}
      onOpen={handleOpen}
      header={header}
      footer={footer}
      closeOnBackdropClick={!isPending && !isInitializing}
    >
      {isInitializing ? (
        skeletonContent
      ) : effectiveStep === 'permission' ? (
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
          isMonitoring={audioState.isPlaying && audioState.activePlayerId === playerId}
          isWetSignalEnabled={!audioState.isBypassed}
          inputGain={currentInputGain}
          onDeviceChange={handleDeviceChange}
          onChannelChange={selectLiveInputChannel}
          onMonitoringChange={handleMonitoringChange}
          onWetSignalToggle={handleWetSignalToggle}
          onInputGainChange={setLiveInputGain}
          isLoading={audioInputDevices.isLoading}
          isConnecting={audioState.inputMode.type === 'connecting'}
          error={audioInputDevices.error}
          connectionError={connectionError}
          onRefresh={handleRefreshDevices}
          channel0Meter={getAudioNodes().channel0PreviewMeter}
          channel1Meter={getAudioNodes().channel1PreviewMeter}
          outputDevices={audioOutputDevices.devices}
          selectedOutputDeviceId={audioOutputDevices.selectedDeviceId}
          onOutputDeviceChange={setOutputDevice}
        />
      )}
    </Dialog>
  );
};

export default SettingsDialog;
