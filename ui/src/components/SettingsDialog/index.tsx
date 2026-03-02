import React, { useState, useCallback } from 'react';
import { Info, Settings, X } from 'lucide-react';
import { Dialog } from '../ui/Dialog';
import { Button } from '../ui/Button';
import { PermissionContent } from './content/PermissionContent';
import { DeviceContent } from './content/DeviceContent';
import { useT3kPlayerContext } from '../../context/T3kPlayerContext';
import { Model, IR, SourceMode } from '../../types';

interface SettingsDialogProps {
  isOpen: boolean;
  onClose: () => void;
  sourceMode: SourceMode;
  selectedModel: Model;
  selectedIr: IR;
  playerId?: string;
  onMonitoringChange: (enabled: boolean) => void | Promise<void>;
}

type SetupStep = 'permission' | 'device-select';

export const SettingsDialog: React.FC<SettingsDialogProps> = ({
  isOpen,
  onClose,
  sourceMode,
  selectedModel,
  selectedIr,
  playerId,
  onMonitoringChange,
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
    selectLiveInputChannel,
    setOutputDevice,
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

  const isDemoMode = sourceMode === 'demo';

  // Derive effective step: if permission isn't granted while on device-select, show permission.
  // Applies in both modes — demo mode needs permission for output device labels on Firefox/Safari.
  const effectiveStep =
    !isInitializing &&
    microphonePermission.status !== 'granted' &&
    step === 'device-select'
      ? 'permission'
      : step;

  // Helper to ensure audio system is initialized before starting live input
  const initAndStartLiveInput = useCallback(
    async (deviceId: string) => {
      setConnectionError(null);
      try {
        if (audioState.initState !== 'ready') {
          await init();
        }
        await startLiveInput(deviceId);
      } catch (error) {
        const message =
          error instanceof DOMException
            ? error.message
            : 'Failed to connect to audio device. Please try again.';
        setConnectionError(message);
      }
    },
    [audioState.initState, init, startLiveInput]
  );

  // Imperative open handler — called once by Dialog's onOpen (ref-guarded against StrictMode)
  const handleOpen = useCallback(() => {
    setIsInitializing(true);

    if (sourceMode === 'demo') {
      if (microphonePermission.status === 'granted') {
        // Permission granted: enumerate input devices first to trigger getUserMedia
        // self-healing (Firefox/Safari require per-session getUserMedia to unlock device labels),
        // then enumerate output devices so they have labels.
        refreshAudioInputDevices()
          .then(() => refreshAudioOutputDevices())
          .then(() => {
            setIsInitializing(false);
            setStep('device-select');
          });
      } else {
        // No permission: enumerate what we can. effectiveStep will redirect to
        // the permission step so the user can unlock device labels.
        refreshAudioOutputDevices().then(() => {
          setIsInitializing(false);
          setStep('device-select');
        });
      }
    } else if (microphonePermission.status === 'granted') {
      // Live mode with permission: enumerate both input and output devices
      Promise.all([
        refreshAudioInputDevices(),
        refreshAudioOutputDevices(),
      ]).then(([{ inputDevices, preferredDeviceId }]) => {
        setIsInitializing(false);
        setStep('device-select');

        // Auto-connect on first open in live mode (no device configured yet)
        if (!audioState.liveInputConfig && inputDevices.length > 0) {
          const deviceToConnect = preferredDeviceId ?? inputDevices[0].deviceId;
          initAndStartLiveInput(deviceToConnect);
        }
      });
    } else {
      // Play-live mode without permission: show permission step
      setIsInitializing(false);
      setStep('permission');
    }
  }, [
    sourceMode,
    microphonePermission.status,
    refreshAudioInputDevices,
    refreshAudioOutputDevices,
    audioState.liveInputConfig,
    initAndStartLiveInput,
  ]);

  // Event handlers
  const handleDeviceChange = useCallback(
    (deviceId: string) => initAndStartLiveInput(deviceId),
    [initAndStartLiveInput]
  );

  const handleRequestPermission = useCallback(async () => {
    try {
      const preferredDeviceId = await requestMicrophonePermission();
      // requestMicrophonePermission only handles the permission prompt.
      // Enumerate devices separately — refreshAudioInputDevices will get labels
      // because getUserMedia just ran (self-healing workaround handles Firefox/Safari).
      await Promise.all([
        refreshAudioInputDevices(),
        refreshAudioOutputDevices(),
      ]);
      setStep('device-select');

      if (preferredDeviceId) {
        initAndStartLiveInput(preferredDeviceId);
      }
    } catch {
      // Error handling is done in context
    }
  }, [
    requestMicrophonePermission,
    refreshAudioInputDevices,
    refreshAudioOutputDevices,
    initAndStartLiveInput,
  ]);

  const handleRefreshDevices = useCallback(async () => {
    await Promise.all([
      refreshAudioInputDevices(),
      refreshAudioOutputDevices(),
    ]);
  }, [refreshAudioInputDevices, refreshAudioOutputDevices]);

  const isPending = microphonePermission.status === 'pending';
  const dialogTitle =
    effectiveStep !== 'permission' ? 'Settings' : 'Microphone Access';
  const dialogIcon =
    effectiveStep !== 'permission' ? (
      <Settings size={24} />
    ) : (
      <Info size={24} />
    );
  const isSaveEnabled = isDemoMode || currentDeviceId !== null;

  // Header with close button
  const header = (
    <div className='flex items-center justify-between p-4'>
      <div className='flex items-center gap-4'>
        {dialogIcon}
        <h2 className='text-lg font-semibold'>{dialogTitle}</h2>
      </div>
      <button
        onClick={onClose}
        className='p-1'
        aria-label='Close'
        disabled={isPending}
      >
        <X size={24} className='text-zinc-400' />
      </button>
    </div>
  );

  // Footer only for device/channel selection step
  const footer =
    !isInitializing && effectiveStep === 'device-select' ? (
      <div className='flex justify-center p-4'>
        <Button
          variant='primary'
          size='lg'
          fullWidth
          onClick={onClose}
          disabled={!isSaveEnabled}
        >
          Done
        </Button>
      </div>
    ) : null;

  return (
    <Dialog
      isOpen={isOpen}
      onClose={onClose}
      onOpen={handleOpen}
      header={header}
      footer={footer}
      closeOnBackdropClick={!isPending && !isInitializing}
      isLoading={isInitializing || audioInputDevices.isLoading}
    >
      {effectiveStep === 'permission' ? (
        <PermissionContent
          status={microphonePermission.status}
          errorMessage={microphonePermission.error}
          onRequestPermission={handleRequestPermission}
        />
      ) : (
        <DeviceContent
          devices={audioInputDevices.devices}
          selectedDeviceId={currentDeviceId ?? ''}
          selectedChannel={currentChannel}
          channelCount={channelCount}
          isMonitoring={
            audioState.isPlaying && audioState.activePlayerId === playerId
          }
          onDeviceChange={handleDeviceChange}
          onChannelChange={selectLiveInputChannel}
          onMonitoringChange={onMonitoringChange}
          isConnecting={audioState.inputMode.type === 'connecting'}
          error={audioInputDevices.error}
          connectionError={connectionError}
          onRefresh={handleRefreshDevices}
          channel0Meter={getAudioNodes().channel0LiveMeter}
          channel1Meter={getAudioNodes().channel1LiveMeter}
          outputDevices={audioOutputDevices.devices}
          selectedOutputDeviceId={audioOutputDevices.selectedDeviceId}
          onOutputDeviceChange={setOutputDevice}
        />
      )}
    </Dialog>
  );
};

export default SettingsDialog;
