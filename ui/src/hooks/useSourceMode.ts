import { useState, useCallback, useMemo } from 'react';
import { useT3kPlayerContext } from '../context/T3kPlayerContext';
import { SourceMode } from '../types';

interface UseSourceModeOptions {
  playerId?: string;
  onSourceModeChange?: (mode: SourceMode) => void;
}

interface UseSourceModeReturn {
  // State
  sourceMode: SourceMode;
  showPlaybackPausedMessage: boolean;

  // Derived (from context, shared by multiple consumers)
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
    reconnectLiveInput,
    setPlaying,
  } = useT3kPlayerContext();

  // Local state
  const [sourceMode, setSourceMode] = useState<SourceMode>('preview');
  const [showPlaybackPausedMessage, setShowPlaybackPausedMessage] = useState(false);

  // Live device options for Select component (shared by multiple consumers)
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
        await reconnectLiveInput();
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
      audioState.activePlayerId,
      playerId,
      setPlaying,
      reconnectLiveInput,
      onSourceModeChange,
    ]
  );

  // Handle device change
  const handleLiveDeviceChange = useCallback(
    async (deviceId: string) => {
      if (deviceId !== audioState.liveInputConfig?.deviceId) {
        await startLiveInput(deviceId);
      }
    },
    [audioState.liveInputConfig?.deviceId, startLiveInput]
  );

  return {
    sourceMode,
    showPlaybackPausedMessage,
    liveDeviceOptions,
    handleSourceModeChange,
    handleLiveDeviceChange,
  };
}
