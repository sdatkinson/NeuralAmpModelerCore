import { useState, useCallback, useMemo } from 'react';
import { useT3kPlayerContext } from '../context/T3kPlayerContext';
import { SourceMode, LiveInputConfig } from '../types';

interface UseSourceModeOptions {
  playerId?: string;
  onSourceModeChange?: (mode: SourceMode) => void;
}

interface UseSourceModeReturn {
  // State
  sourceMode: SourceMode;
  showPlaybackPausedMessage: boolean;

  // Derived (from context)
  isLiveConfigured: boolean;
  isLiveInputActive: boolean;
  currentDeviceId: string | null;
  liveInputConfig: LiveInputConfig | null;
  liveDeviceOptions: Array<{ label: string; value: string }>;

  // Handlers
  handleSourceModeChange: (mode: SourceMode) => Promise<void>;
  handleLiveDeviceChange: (deviceId: string) => Promise<void>;
}

export function useSourceMode(options: UseSourceModeOptions = {}): UseSourceModeReturn {
  const { playerId, onSourceModeChange } = options;

  const {
    audioState,
    audioInputDevices,
    startLiveInput,
    setPlaying,
  } = useT3kPlayerContext();

  // Local state
  const [sourceMode, setSourceMode] = useState<SourceMode>('preview');
  const [showPlaybackPausedMessage, setShowPlaybackPausedMessage] = useState(false);

  // Derived state for live input
  const liveInputConfig = audioState.liveInputConfig;
  const isLiveConfigured = liveInputConfig !== null;
  const isLiveInputActive = audioState.inputMode.type === 'live';
  const currentDeviceId = liveInputConfig?.deviceId ?? null;

  // Live device options for Select component
  const liveDeviceOptions = useMemo(
    () =>
      audioInputDevices.devices.map(device => ({
        label: device.label,
        value: device.deviceId,
      })),
    [audioInputDevices.devices]
  );

  // Handle source mode change (Preview <-> Live)
  const handleSourceModeChange = useCallback(
    async (newMode: SourceMode) => {
      if (newMode === sourceMode) return;

      if (newMode === 'live') {
        // Switching from Preview to Live
        if (audioState.isPlaying) {
          setPlaying(false);
          setShowPlaybackPausedMessage(true);
          setTimeout(() => setShowPlaybackPausedMessage(false), 3000);
        }

        // If live input is already active (another player started it), just switch mode
        // If not active but we have a config, reconnect using the saved config
        const liveAlreadyActive = audioState.inputMode.type === 'live';
        if (!liveAlreadyActive && audioState.liveInputConfig) {
          await startLiveInput(audioState.liveInputConfig.deviceId, {
            initialChannel: audioState.liveInputConfig.selectedChannel,
            initialChannelGains: audioState.liveInputConfig.channelGains,
          });
        }
      } else {
        // Switching from Live to Preview
        setShowPlaybackPausedMessage(false);

        // Stop this player's monitoring if it was active
        if (audioState.activePlayerId === playerId) {
          setPlaying(false);
        }
      }

      setSourceMode(newMode);
      onSourceModeChange?.(newMode);
    },
    [
      sourceMode,
      audioState.isPlaying,
      audioState.inputMode.type,
      audioState.liveInputConfig,
      audioState.activePlayerId,
      playerId,
      setPlaying,
      startLiveInput,
      onSourceModeChange,
    ]
  );

  // Handle device change
  const handleLiveDeviceChange = useCallback(
    async (deviceId: string) => {
      if (deviceId !== currentDeviceId) {
        await startLiveInput(deviceId);
      }
    },
    [currentDeviceId, startLiveInput]
  );

  return {
    sourceMode,
    showPlaybackPausedMessage,
    isLiveConfigured,
    isLiveInputActive,
    currentDeviceId,
    liveInputConfig,
    liveDeviceOptions,
    handleSourceModeChange,
    handleLiveDeviceChange,
  };
}
