import { useState, useCallback, useEffect, useRef, useMemo } from 'react';
import { useT3kPlayerContext } from '../context/T3kPlayerContext';
import { SourceMode, ChannelSelection, LiveInputConfig } from '../types';

// Snapshot of settings configuration for restore operations (used by dialog cancel)
interface SettingsSnapshot {
  // Live input settings (only present when in live mode)
  deviceId?: string;
  channel?: ChannelSelection;
  channelGains?: Record<ChannelSelection, number>;
  wasPlaying?: boolean;
  wasBypassed?: boolean;
  // Output device (used in both modes)
  outputDeviceId: string | null;
}

interface UseLiveModeOptions {
  playerId?: string;
  onSourceModeChange?: (mode: SourceMode) => void;
}

interface UseLiveModeReturn {
  // State
  sourceMode: SourceMode;
  isSettingsDialogOpen: boolean;
  showPlaybackPausedMessage: boolean;
  showOutputFallbackMessage: boolean;

  // Derived (from context)
  isLiveConfigured: boolean;      // Is there a configured live device? (persists across mode switches)
  isLiveInputActive: boolean;     // Is the audio engine currently using live input?
  currentDeviceId: string | null;
  liveInputConfig: LiveInputConfig | null;
  liveDeviceOptions: Array<{ label: string; value: string }>;

  // Handlers
  handleSourceModeChange: (mode: SourceMode) => Promise<void>;
  handleLiveDeviceChange: (deviceId: string) => Promise<void>;
  openSettingsDialog: () => void;
  closeSettingsDialog: () => void;

  // Snapshot functions (for SettingsDialog)
  saveSettingsSnapshot: (options?: { includeLiveSettings?: boolean }) => void;
  restoreSettingsSnapshot: (options?: { includePlaybackState?: boolean; includeLiveSettings?: boolean; includeOutputDevice?: boolean }) => Promise<void>;
  clearSettingsSnapshot: () => void;
}

export function useLiveMode(options: UseLiveModeOptions = {}): UseLiveModeReturn {
  const { playerId, onSourceModeChange } = options;

  const {
    audioState,
    audioInputDevices,
    audioOutputDevices,
    startLiveInput,
    setPlaying,
    setOutputDevice,
    toggleBypass,
  } = useT3kPlayerContext();

  // Local state
  const [sourceMode, setSourceMode] = useState<SourceMode>('preview');
  const [showPlaybackPausedMessage, setShowPlaybackPausedMessage] = useState(false);
  const [showOutputFallbackMessage, setShowOutputFallbackMessage] = useState(false);
  const [isSettingsDialogOpen, setIsSettingsDialogOpen] = useState(false);

  // Per-player snapshot state (this is the key fix - each player has its own snapshot)
  const [settingsSnapshot, setSettingsSnapshot] = useState<SettingsSnapshot | null>(null);

  // Refs for tracking output device changes
  const prevOutputDeviceIdRef = useRef<string | null>(audioOutputDevices.selectedDeviceId);

  // Derived state for live input
  // liveInputConfig persists even when preview is active, so UI stays stable
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

  // Detect output device fallback (when selected device is disconnected)
  useEffect(() => {
    const prevId = prevOutputDeviceIdRef.current;
    const currentId = audioOutputDevices.selectedDeviceId;

    // Detect transition from non-null to null (fallback to default)
    if (prevId !== null && currentId === null) {
      setShowOutputFallbackMessage(true);
      const timer = setTimeout(() => setShowOutputFallbackMessage(false), 3000);
      return () => clearTimeout(timer);
    }

    prevOutputDeviceIdRef.current = currentId;
  }, [audioOutputDevices.selectedDeviceId]);

  // Cleanup message timers on unmount
  useEffect(() => {
    return () => {
      setShowPlaybackPausedMessage(false);
      setShowOutputFallbackMessage(false);
    };
  }, []);

  // Save current settings to snapshot (for restoring later)
  // Options:
  //   includeLiveSettings: if true, also save live input settings (device, channel, gains)
  //                        (use true when live is configured, false otherwise)
  // Note: When includeLiveSettings is false, existing live settings in the snapshot are preserved
  const saveSettingsSnapshot = useCallback((options?: { includeLiveSettings?: boolean }): void => {
    const includeLive = options?.includeLiveSettings ?? false;

    setSettingsSnapshot(prev => {
      const snapshot: SettingsSnapshot = {
        outputDeviceId: audioOutputDevices.selectedDeviceId,
      };

      // Include live settings if requested and we have a configured device
      if (includeLive && audioState.liveInputConfig) {
        snapshot.deviceId = audioState.liveInputConfig.deviceId;
        snapshot.channel = audioState.liveInputConfig.selectedChannel ?? 'first';
        snapshot.channelGains = audioState.liveInputConfig.channelGains ?? { first: 0, second: 0 };
        snapshot.wasPlaying = audioState.isPlaying;
        snapshot.wasBypassed = audioState.isBypassed;
      } else if (prev) {
        // Preserve existing live settings from previous snapshot
        snapshot.deviceId = prev.deviceId;
        snapshot.channel = prev.channel;
        snapshot.channelGains = prev.channelGains;
        snapshot.wasPlaying = prev.wasPlaying;
        snapshot.wasBypassed = prev.wasBypassed;
      }

      return snapshot;
    });
  }, [audioOutputDevices.selectedDeviceId, audioState.liveInputConfig, audioState.isPlaying, audioState.isBypassed]);

  // Restore settings from snapshot (handles async reconnection for live mode)
  // Options:
  //   includePlaybackState: if true, also restore playing and bypass state
  //                         (use true for dialog cancel, false for source mode switching)
  //   includeLiveSettings: if true, restore live input settings (device, channel, gains)
  //                        (use true when restoring in live mode, false in preview mode)
  //   includeOutputDevice: if true, restore output device (use true for dialog cancel, false for mode switching)
  const restoreSettingsSnapshot = useCallback(async (options?: { includePlaybackState?: boolean; includeLiveSettings?: boolean; includeOutputDevice?: boolean }): Promise<void> => {
    if (!settingsSnapshot) return;

    const { deviceId, channel, channelGains, wasPlaying, wasBypassed, outputDeviceId } = settingsSnapshot;
    const includePlaybackState = options?.includePlaybackState ?? false;
    const includeLiveSettings = options?.includeLiveSettings ?? true;
    const includeOutputDevice = options?.includeOutputDevice ?? true;

    // If snapshot has live settings and we should restore them
    if (includeLiveSettings && deviceId && channel && channelGains) {
      // Reconnect to device with initial channel and gains
      // This avoids stale closure issues that would occur if we called
      // selectLiveInputChannel separately after startLiveInput
      await startLiveInput(deviceId, {
        initialChannel: channel,
        initialChannelGains: channelGains,
      });

      // Only restore playback/bypass state if requested (e.g., dialog cancel)
      if (includePlaybackState) {
        setPlaying(wasPlaying ?? false, playerId);
        if (audioState.isBypassed !== (wasBypassed ?? false)) {
          toggleBypass();
        }
      }
    }

    // Restore output device AFTER startLiveInput to avoid race condition
    // (startLiveInput reads audioOutputDevices.selectedDeviceId which may not have updated yet)
    if (includeOutputDevice) {
      await setOutputDevice(outputDeviceId);
    }
  }, [settingsSnapshot, audioState.isBypassed, setOutputDevice, startLiveInput, setPlaying, toggleBypass, playerId]);

  // Clear the snapshot (after successful connect or when no longer needed)
  const clearSettingsSnapshot = useCallback((): void => {
    setSettingsSnapshot(null);
  }, []);

  // Handle source mode change (Preview <-> Live)
  const handleSourceModeChange = useCallback(
    async (newMode: SourceMode) => {
      if (newMode === sourceMode) return;

      if (newMode === 'live') {
        // Switching from Preview to Live
        if (audioState.isPlaying) {
          // Pause playback but preserve playhead position
          setPlaying(false);
          setShowPlaybackPausedMessage(true);

          // Auto-hide the message after 3 seconds
          const timer = setTimeout(() => {
            setShowPlaybackPausedMessage(false);
          }, 3000);
          // Note: timer cleanup happens on unmount via effect above
        }

        // If live input is already active (another player started it), just switch mode
        // If not active but we have a config, reconnect using the saved config or snapshot
        const liveAlreadyActive = audioState.inputMode.type === 'live';
        if (!liveAlreadyActive) {
          // Try to reconnect using liveInputConfig first (persisted config), then snapshot
          if (audioState.liveInputConfig) {
            await startLiveInput(audioState.liveInputConfig.deviceId, {
              initialChannel: audioState.liveInputConfig.selectedChannel,
              initialChannelGains: audioState.liveInputConfig.channelGains,
            });
          } else if (settingsSnapshot?.deviceId) {
            await restoreSettingsSnapshot({ includeOutputDevice: false });
          }
        }
      } else {
        // Switching from Live to Preview
        setShowPlaybackPausedMessage(false);

        // DON'T stop live input here - other players may still be using it.
        // Live input will be stopped when this player presses play (takes over audio engine)
        // or when user explicitly disconnects in settings.

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
      settingsSnapshot,
      setPlaying,
      startLiveInput,
      restoreSettingsSnapshot,
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

  // Settings dialog handlers
  const openSettingsDialog = useCallback(() => {
    setIsSettingsDialogOpen(true);
  }, []);

  const closeSettingsDialog = useCallback(() => {
    setIsSettingsDialogOpen(false);
  }, []);

  return {
    // State
    sourceMode,
    isSettingsDialogOpen,
    showPlaybackPausedMessage,
    showOutputFallbackMessage,

    // Derived
    isLiveConfigured,
    isLiveInputActive,
    currentDeviceId,
    liveInputConfig,
    liveDeviceOptions,

    // Handlers
    handleSourceModeChange,
    handleLiveDeviceChange,
    openSettingsDialog,
    closeSettingsDialog,

    // Snapshot functions
    saveSettingsSnapshot,
    restoreSettingsSnapshot,
    clearSettingsSnapshot,
  };
}
