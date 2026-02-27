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
  playDeviceOptions: Array<{ label: string; value: string }>;

  // Handlers
  handleSourceModeChange: (mode: SourceMode) => Promise<void>;
  handlePlayDeviceChange: (deviceId: string) => Promise<void>;
}

export function useSourceMode(
  options: UseSourceModeOptions = {}
): UseSourceModeReturn {
  const { playerId, onSourceModeChange } = options;

  const {
    audioState,
    audioInputDevices,
    sourceMode,
    setSourceMode,
    startPlayInput,
    reconnectPlayInput,
    setPlaying,
  } = useT3kPlayerContext();
  const [showPlaybackPausedMessage, setShowPlaybackPausedMessage] =
    useState(false);

  // Play device options for Select component (shared by multiple consumers)
  const playDeviceOptions = useMemo(
    () =>
      audioInputDevices.devices.map(device => ({
        label: device.label,
        value: device.deviceId,
      })),
    [audioInputDevices.devices]
  );

  // Handle source mode change (Demo <-> Play)
  const handleSourceModeChange = useCallback(
    async (newMode: SourceMode) => {
      if (newMode === sourceMode) return;

      if (newMode === 'play') {
        // Switching from Demo to Play
        if (audioState.isPlaying) {
          setPlaying(false);
          setShowPlaybackPausedMessage(true);
          setTimeout(() => setShowPlaybackPausedMessage(false), 3000);
        }

        // If play input is already active (another player started it), just switch mode
        // If not active but we have a config, reconnect using the saved config
        await reconnectPlayInput();
      } else {
        // Switching from Play to Demo
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
      reconnectPlayInput,
      onSourceModeChange,
    ]
  );

  // Handle device change
  const handlePlayDeviceChange = useCallback(
    async (deviceId: string) => {
      if (deviceId !== audioState.playInputConfig?.deviceId) {
        await startPlayInput(deviceId);
      }
    },
    [audioState.playInputConfig?.deviceId, startPlayInput]
  );

  return {
    sourceMode,
    showPlaybackPausedMessage,
    playDeviceOptions,
    handleSourceModeChange,
    handlePlayDeviceChange,
  };
}
