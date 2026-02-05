import React, { useState, useCallback, useEffect } from 'react';
import { ArrowLeft } from 'lucide-react';
import { Dialog } from '../ui/Dialog';
import { Button } from '../ui/Button';
import { PermissionContent } from './content/PermissionContent';
import { DeviceChannelContent } from './content/DeviceChannelContent';
import { useT3kPlayerContext } from '../../context/T3kPlayerContext';
import { ChannelSelection, Model, IR, SourceMode } from '../../types';

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
    refreshAudioDevices,
    startLiveInput,
    selectLiveInputChannel,
    setLiveInputGain,
    setOutputDevice,
    setPlaying,
    toggleBypass,
    getAudioNodes,
    loadModel,
    loadIr,
    removeIr,
  } = useT3kPlayerContext();

  // Local state: only UI flow
  const [step, setStep] = useState<SetupStep>('permission');
  const [isInitializing, setIsInitializing] = useState(true);

  // Derived from context (single source of truth)
  const liveInputConfig = audioState.liveInputConfig;
  const currentDeviceId = liveInputConfig?.deviceId ?? null;
  const currentChannel = liveInputConfig?.selectedChannel ?? 'first';
  const channelCount = liveInputConfig?.channelCount ?? 1;
  const currentInputGain = liveInputConfig?.channelGains?.[currentChannel] ?? 0;

  const isLiveMode = sourceMode === 'live';
  const isPreviewMode = sourceMode === 'preview';

  // React to permission changes while dialog is open
  useEffect(() => {
    if (!isOpen || isInitializing) return;

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

  // Imperative open handler — called once by Dialog's onOpen (ref-guarded against StrictMode)
  const handleOpen = useCallback(() => {
    setIsInitializing(true);

    if (microphonePermission.status === 'granted') {
      refreshAudioDevices().then(({ inputDevices, preferredDeviceId }) => {
        setIsInitializing(false);
        setStep('device-channel-select');

        // Auto-connect on first open in live mode (no device configured yet)
        if (sourceMode === 'live' && !audioState.liveInputConfig && inputDevices.length > 0) {
          const deviceToConnect = preferredDeviceId ?? inputDevices[0].deviceId;
          initAndStartLiveInput(deviceToConnect);
        }
      });
    } else {
      setIsInitializing(false);
      setStep('permission');
    }
  }, [microphonePermission.status, refreshAudioDevices, sourceMode, audioState.liveInputConfig, initAndStartLiveInput]);

  // Event handlers
  const handleDeviceChange = useCallback((deviceId: string) => {
    initAndStartLiveInput(deviceId);
  }, [initAndStartLiveInput]);

  const handleChannelChange = useCallback((channel: ChannelSelection) => {
    selectLiveInputChannel(channel);
  }, [selectLiveInputChannel]);

  const handleMonitoringChange = useCallback(async (enabled: boolean) => {
    if (enabled) {
      if (audioState.initState === 'ready') {
        if (audioState.modelUrl !== selectedModel.url) {
          await loadModel(selectedModel.url);
        }
        if (selectedIr.url) {
          if (audioState.irUrl !== selectedIr.url) {
            await loadIr({
              url: selectedIr.url,
              wetAmount: selectedIr.mix ?? 1,
              gainAmount: selectedIr.gain ?? 1,
            });
          }
        } else if (audioState.irUrl !== null) {
          removeIr();
        }
      }
    }
    setPlaying(enabled, playerId);
  }, [setPlaying, loadModel, loadIr, removeIr, audioState.initState, audioState.modelUrl, audioState.irUrl, selectedModel, selectedIr, playerId]);

  const handleWetSignalToggle = useCallback(() => {
    toggleBypass();
  }, [toggleBypass]);

  const handleInputGainChange = useCallback((gainDb: number) => {
    setLiveInputGain(gainDb);
  }, [setLiveInputGain]);

  const handleOutputDeviceChange = useCallback((deviceId: string | null) => {
    setOutputDevice(deviceId);
  }, [setOutputDevice]);

  const handleRequestPermission = useCallback(async () => {
    try {
      const preferredDeviceId = await requestMicrophonePermission();
      await refreshAudioDevices();
      setStep('device-channel-select');

      if (isLiveMode && preferredDeviceId) {
        initAndStartLiveInput(preferredDeviceId);
      }
    } catch {
      // Error handling is done in context
    }
  }, [requestMicrophonePermission, refreshAudioDevices, initAndStartLiveInput, isLiveMode]);

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
  const footer = !isInitializing && step === 'device-channel-select' ? (
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
