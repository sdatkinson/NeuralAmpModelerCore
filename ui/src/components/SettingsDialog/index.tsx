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
  onSave: () => void;
  onCancel: () => void;
  sourceMode: SourceMode;
  selectedModel: Model;
  selectedIr: IR;
  playerId?: string;
}

type SetupStep = 'permission' | 'device-select';

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
    startPlayInput,
    reconnectPlayInput,
    selectPlayInputChannel,
    syncEngineSettings,
    setOutputDevice,
    setPlaying,
    getAudioNodes,
  } = useT3kPlayerContext();

  // Local state: only UI flow
  const [step, setStep] = useState<SetupStep>('permission');
  const [isInitializing, setIsInitializing] = useState(true);
  const [connectionError, setConnectionError] = useState<string | null>(null);

  // Derived from context (single source of truth)
  const playInputConfig = audioState.playInputConfig;
  const currentDeviceId = playInputConfig?.deviceId ?? null;
  const currentChannel = playInputConfig?.selectedChannel ?? 'first';
  const channelCount = playInputConfig?.channelCount ?? 1;

  const isDemoMode = sourceMode === 'demo';

  // Derive effective step: if permission isn't granted while on device-select, show permission.
  // Applies in both modes — demo mode needs permission for output device labels on Firefox/Safari.
  const effectiveStep =
    !isInitializing &&
    microphonePermission.status !== 'granted' &&
    step === 'device-select'
      ? 'permission'
      : step;

  // Helper to ensure audio system is initialized before starting play input
  const initAndStartPlayInput = useCallback(
    async (deviceId: string) => {
      setConnectionError(null);
      try {
        if (audioState.initState !== 'ready') {
          await init();
        }
        await startPlayInput(deviceId);
      } catch (error) {
        const message =
          error instanceof DOMException
            ? error.message
            : 'Failed to connect to audio device. Please try again.';
        setConnectionError(message);
      }
    },
    [audioState.initState, init, startPlayInput]
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
      // Play mode with permission: enumerate both input and output devices
      Promise.all([
        refreshAudioInputDevices(),
        refreshAudioOutputDevices(),
      ]).then(([{ inputDevices, preferredDeviceId }]) => {
        setIsInitializing(false);
        setStep('device-select');

        // Auto-connect on first open in play mode (no device configured yet)
        if (!audioState.playInputConfig && inputDevices.length > 0) {
          const deviceToConnect = preferredDeviceId ?? inputDevices[0].deviceId;
          initAndStartPlayInput(deviceToConnect);
        }
      });
    } else {
      // Play mode without permission: show permission step
      setIsInitializing(false);
      setStep('permission');
    }
  }, [
    sourceMode,
    microphonePermission.status,
    refreshAudioInputDevices,
    refreshAudioOutputDevices,
    audioState.playInputConfig,
    initAndStartPlayInput,
  ]);

  // Event handlers
  const handleDeviceChange = useCallback(
    (deviceId: string) => {
      initAndStartPlayInput(deviceId);
    },
    [initAndStartPlayInput]
  );

  const handleMonitoringChange = useCallback(
    async (enabled: boolean) => {
      if (enabled) {
        // Reconnect play input if it was torn down (e.g. another player was in demo mode)
        await reconnectPlayInput();

        if (audioState.initState === 'ready') {
          await syncEngineSettings({
            modelUrl: selectedModel.url,
            ir: {
              url: selectedIr.url,
              mix: selectedIr.mix ?? 1,
              gain: selectedIr.gain ?? 1,
            },
            bypassed: audioState.isBypassed,
          });
        }

        if (playerId) {
          setPlaying(true, playerId);
        }
      } else {
        setPlaying(false);
      }
    },
    [
      reconnectPlayInput,
      setPlaying,
      syncEngineSettings,
      audioState.initState,
      audioState.isBypassed,
      selectedModel,
      selectedIr,
      playerId,
    ]
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
        initAndStartPlayInput(preferredDeviceId);
      }
    } catch {
      // Error handling is done in context
    }
  }, [
    requestMicrophonePermission,
    refreshAudioInputDevices,
    refreshAudioOutputDevices,
    initAndStartPlayInput,
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
        onClick={onCancel}
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
          onClick={onSave}
          disabled={!isSaveEnabled}
        >
          Done
        </Button>
      </div>
    ) : null;

  return (
    <Dialog
      isOpen={isOpen}
      onClose={onCancel}
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
          onChannelChange={selectPlayInputChannel}
          onMonitoringChange={handleMonitoringChange}
          isConnecting={audioState.inputMode.type === 'connecting'}
          error={audioInputDevices.error}
          connectionError={connectionError}
          onRefresh={handleRefreshDevices}
          channel0Meter={getAudioNodes().channel0PlayMeter}
          channel1Meter={getAudioNodes().channel1PlayMeter}
          outputDevices={audioOutputDevices.devices}
          selectedOutputDeviceId={audioOutputDevices.selectedDeviceId}
          onOutputDeviceChange={setOutputDevice}
        />
      )}
    </Dialog>
  );
};

export default SettingsDialog;
