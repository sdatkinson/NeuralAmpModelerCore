import { useCallback, useEffect, useState } from 'react';
import {
  AudioInputDevice,
  AudioInputDeviceState,
  AudioOutputDeviceState,
  LiveInputConfig,
  MicrophonePermissionState,
  MicrophonePermissionStatus,
} from '../types';
import { mapDevices } from '../utils/devices';
import { showToast } from './useToast';

type LiveInputUnavailableReason = 'device-disconnected' | 'permission-revoked';

interface UseAudioDevicesAndPermissionsParams {
  /** Callback to apply audio routing when output device changes. */
  applyOutputRouting: (deviceId: string | null) => Promise<void>;
  /** Callback to tear down live input audio nodes and restore preview path. */
  teardownLiveInput: () => void;
  /** Callback to reset audio state when live input is lost (mode, playback, config). */
  onLiveInputLost: () => void;
  /** Current live input config from audio state (for disconnect detection). */
  liveInputConfig: LiveInputConfig | null;
}

interface UseAudioDevicesAndPermissionsReturn {
  microphonePermission: MicrophonePermissionState;
  audioInputDevices: AudioInputDeviceState;
  audioOutputDevices: AudioOutputDeviceState;
  requestMicrophonePermission: () => Promise<string | null>;
  refreshAudioDevices: () => Promise<{ inputDevices: AudioInputDevice[]; preferredDeviceId: string | null }>;
  refreshAudioOutputDevices: () => Promise<void>;
  setOutputDevice: (deviceId: string | null) => Promise<void>;
  handleLiveInputUnavailable: (reason: LiveInputUnavailableReason) => void;
}

/**
 * Manages audio device enumeration, output device selection, microphone permissions,
 * and hot-plug detection. Owns all device and permission state.
 *
 * Audio graph side effects are delegated to the caller via narrow callbacks.
 */
export function useAudioDevicesAndPermissions({
  applyOutputRouting,
  teardownLiveInput,
  onLiveInputLost,
  liveInputConfig,
}: UseAudioDevicesAndPermissionsParams): UseAudioDevicesAndPermissionsReturn {

  // --- State ---

  const [microphonePermission, setMicrophonePermission] = useState<MicrophonePermissionState>({
    status: 'idle',
    error: null,
  });

  const [audioInputDevices, setAudioInputDevices] = useState<AudioInputDeviceState>({
    devices: [],
    isLoading: false,
    error: null,
    preferredDeviceId: null,
  });

  const [audioOutputDevices, setAudioOutputDevices] = useState<AudioOutputDeviceState>({
    devices: [],
    selectedDeviceId: null,
  });

  // --- Permission logic ---

  const queryBrowserPermission = useCallback(async (): Promise<MicrophonePermissionStatus> => {
    try {
      const result = await navigator.permissions.query({ name: 'microphone' as PermissionName });
      if (result.state === 'granted') return 'granted';
      if (result.state === 'denied') return 'blocked';
      return 'idle';
    } catch {
      // Permissions API not supported for microphone (e.g., Firefox)
      return 'idle';
    }
  }, []);

  // Query permission state on mount and listen for changes
  useEffect(() => {
    if (typeof window === 'undefined') return;

    let mounted = true;

    const checkPermission = async () => {
      const status = await queryBrowserPermission();
      if (mounted) {
        setMicrophonePermission(prev => ({
          ...prev,
          status,
          error: status === 'granted' ? null : prev.error,
        }));
      }
    };

    checkPermission();

    const setupPermissionListener = async () => {
      try {
        const result = await navigator.permissions.query({ name: 'microphone' as PermissionName });

        const handleChange = () => {
          if (!mounted) return;
          if (result.state === 'granted') {
            setMicrophonePermission(prev => ({ ...prev, status: 'granted', error: null }));
          } else if (result.state === 'denied') {
            setMicrophonePermission(prev => ({
              ...prev,
              status: 'blocked',
              error: 'Microphone access is blocked. Please enable it in your browser settings.',
            }));
          } else {
            setMicrophonePermission(prev => ({ ...prev, status: 'idle' }));
          }
        };

        result.addEventListener('change', handleChange);
        return () => result.removeEventListener('change', handleChange);
      } catch {
        return undefined;
      }
    };

    let cleanupListener: (() => void) | undefined;
    setupPermissionListener().then(cleanup => {
      cleanupListener = cleanup;
    });

    return () => {
      mounted = false;
      cleanupListener?.();
    };
  }, [queryBrowserPermission]);

  // Request microphone permission
  // Returns the device ID the user selected in the browser's permission dialog
  const requestMicrophonePermission = useCallback(async (): Promise<string | null> => {
    // Capture previous status via functional updater (avoids needing a ref)
    let previousStatus = 'idle' as MicrophonePermissionStatus;
    setMicrophonePermission(prev => {
      previousStatus = prev.status;
      return { status: 'pending', error: null };
    });

    setAudioInputDevices(prev => ({ ...prev, isLoading: true, error: null }));

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const track = stream.getAudioTracks()[0];
      const selectedDeviceId = track?.getSettings()?.deviceId ?? null;
      stream.getTracks().forEach(t => t.stop());

      setMicrophonePermission({ status: 'granted', error: null });

      const allDevices = await navigator.mediaDevices.enumerateDevices();
      const audioInputs = mapDevices(allDevices, 'audioinput');

      setAudioInputDevices({
        devices: audioInputs,
        isLoading: false,
        error: audioInputs.length === 0 ? 'No audio input devices found.' : null,
        preferredDeviceId: selectedDeviceId,
      });

      return selectedDeviceId;
    } catch (error) {
      setAudioInputDevices(prev => ({ ...prev, isLoading: false }));

      if (error instanceof DOMException) {
        if (error.name === 'NotAllowedError' || error.name === 'PermissionDeniedError') {
          const permissionState = await queryBrowserPermission();
          // Firefox doesn't support Permissions API for microphone, so queryBrowserPermission
          // returns 'idle'. If we already tried once (status is 'denied') and get denied again,
          // Firefox won't re-prompt — treat as blocked.
          const alreadyDenied = previousStatus === 'denied';
          if (permissionState === 'blocked' || (permissionState === 'idle' && alreadyDenied)) {
            setMicrophonePermission({
              status: 'blocked',
              error: 'Microphone access is blocked. Please enable it in your browser settings.',
            });
          } else {
            setMicrophonePermission({
              status: 'denied',
              error: 'Microphone access was denied. Please try again.',
            });
          }
        } else if (error.name === 'NotFoundError') {
          setMicrophonePermission({ status: 'error', error: 'No microphone or audio input device was found.' });
        } else if (error.name === 'NotReadableError') {
          setMicrophonePermission({ status: 'error', error: 'Your microphone is being used by another application.' });
        } else {
          setMicrophonePermission({ status: 'error', error: error.message });
        }
      } else {
        setMicrophonePermission({ status: 'error', error: 'An unexpected error occurred. Please try again.' });
      }
      throw error;
    }
  }, [queryBrowserPermission]);

  // --- Device enumeration ---

  const refreshAudioDevices = useCallback(async (): Promise<{ inputDevices: AudioInputDevice[]; preferredDeviceId: string | null }> => {
    const permApiStatus = await queryBrowserPermission();

    if (permApiStatus === 'granted' || permApiStatus === 'blocked') {
      setMicrophonePermission(prev => ({
        ...prev,
        status: permApiStatus,
        error: permApiStatus === 'granted' ? null : prev.error,
      }));
      if (permApiStatus === 'blocked') {
        return { inputDevices: [], preferredDeviceId: null };
      }
    }

    setAudioInputDevices(prev => ({ ...prev, isLoading: true, error: null }));

    try {
      let allDevices = await navigator.mediaDevices.enumerateDevices();
      let audioInputs = allDevices.filter(device => device.kind === 'audioinput');

      // Firefox returns empty labels until getUserMedia is called in this session
      const hasLabels = audioInputs.some(device => device.label && device.label.length > 0);

      if (!hasLabels) {
        try {
          const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
          stream.getTracks().forEach(t => t.stop());
          allDevices = await navigator.mediaDevices.enumerateDevices();
          audioInputs = allDevices.filter(device => device.kind === 'audioinput');
          setMicrophonePermission({ status: 'granted', error: null });
        } catch (error) {
          if (error instanceof DOMException &&
              (error.name === 'NotAllowedError' || error.name === 'PermissionDeniedError')) {
            setMicrophonePermission(prev => ({ ...prev, status: 'denied', error: null }));
            setAudioInputDevices(prev => ({ ...prev, isLoading: false }));
            return { inputDevices: [], preferredDeviceId: null };
          }
        }
      } else {
        setMicrophonePermission(prev => ({ ...prev, status: 'granted', error: null }));
      }

      const mappedInputDevices = mapDevices(allDevices, 'audioinput');
      const mappedOutputDevices = mapDevices(allDevices, 'audiooutput');

      setAudioInputDevices(prev => ({
        ...prev,
        devices: mappedInputDevices,
        isLoading: false,
        error: mappedInputDevices.length === 0 ? 'No audio input devices found.' : null,
      }));

      setAudioOutputDevices(prev => ({ ...prev, devices: mappedOutputDevices }));

      return { inputDevices: mappedInputDevices, preferredDeviceId: audioInputDevices.preferredDeviceId };
    } catch {
      setAudioInputDevices(prev => ({
        ...prev,
        isLoading: false,
        error: 'Failed to enumerate audio devices.',
      }));
      return { inputDevices: [], preferredDeviceId: null };
    }
  }, [queryBrowserPermission, audioInputDevices.preferredDeviceId]);

  const refreshAudioOutputDevices = useCallback(async (): Promise<void> => {
    try {
      const allDevices = await navigator.mediaDevices.enumerateDevices();
      const mappedOutputDevices = mapDevices(allDevices, 'audiooutput');
      setAudioOutputDevices(prev => ({ ...prev, devices: mappedOutputDevices }));
    } catch (error) {
      console.error('Error enumerating output devices:', error);
    }
  }, []);

  // --- Output device selection ---

  const setOutputDevice = useCallback(async (deviceId: string | null): Promise<void> => {
    // Capture previous for rollback via functional updater
    let previousDeviceId: string | null = null;
    setAudioOutputDevices(prev => {
      previousDeviceId = prev.selectedDeviceId;
      return { ...prev, selectedDeviceId: deviceId };
    });

    try {
      await applyOutputRouting(deviceId);
    } catch (error) {
      console.warn('[Audio] Failed to set output device, reverting:', error);
      setAudioOutputDevices(prev => ({ ...prev, selectedDeviceId: previousDeviceId }));
    }
  }, [applyOutputRouting]);

  // --- Live input unavailable handler ---

  const handleLiveInputUnavailable = useCallback((reason: LiveInputUnavailableReason): void => {
    teardownLiveInput();
    onLiveInputLost();

    const errorMessages: Record<LiveInputUnavailableReason, string> = {
      'device-disconnected': 'Audio input device was disconnected.',
      'permission-revoked': 'Microphone permission was revoked. Please re-enable access in your browser settings.',
    };
    setAudioInputDevices(prev => ({ ...prev, error: errorMessages[reason] }));

    const toastMessages: Record<LiveInputUnavailableReason, string> = {
      'device-disconnected': 'Audio input device disconnected',
      'permission-revoked': 'Microphone permission revoked',
    };
    showToast(toastMessages[reason]);
  }, [teardownLiveInput, onLiveInputLost]);

  // --- Hot-plug detection ---

  useEffect(() => {
    if (typeof window === 'undefined' || !navigator.mediaDevices) return;

    const handleDeviceChange = async () => {
      if (microphonePermission.status === 'granted') {
        try {
          const allDevices = await navigator.mediaDevices.enumerateDevices();
          const audioInputs = mapDevices(allDevices, 'audioinput');

          // Check if configured input device was disconnected
          const configuredDeviceId = liveInputConfig?.deviceId;
          if (configuredDeviceId) {
            const configuredDeviceStillExists = audioInputs.some(d => d.deviceId === configuredDeviceId);
            if (!configuredDeviceStillExists) {
              console.warn('Configured audio input device was disconnected');
              handleLiveInputUnavailable('device-disconnected');
              setAudioInputDevices(prev => ({ ...prev, devices: audioInputs }));
              return;
            }
          }

          setAudioInputDevices(prev => ({
            ...prev,
            devices: audioInputs,
            error: audioInputs.length === 0 ? 'All audio devices have been disconnected.' : null,
          }));

          // Check if selected output device was disconnected
          const audioOutputs = mapDevices(allDevices, 'audiooutput');
          let outputDeviceLost = false;
          setAudioOutputDevices(prev => {
            const selectedStillExists = prev.selectedDeviceId === null ||
              audioOutputs.some(d => d.deviceId === prev.selectedDeviceId);
            if (!selectedStillExists) {
              outputDeviceLost = true;
              return { ...prev, devices: audioOutputs, selectedDeviceId: null };
            }
            return { ...prev, devices: audioOutputs };
          });
          if (outputDeviceLost) {
            console.warn('Selected audio output device was disconnected, falling back to system default');
            applyOutputRouting(null).catch(e =>
              console.warn('[Audio] Failed to reset output routing:', e)
            );
            showToast('Output switched to default');
          }
        } catch {
          setAudioInputDevices(prev => ({ ...prev, error: 'Failed to refresh device list.' }));
        }
      }
    };

    navigator.mediaDevices.addEventListener('devicechange', handleDeviceChange);
    return () => {
      navigator.mediaDevices.removeEventListener('devicechange', handleDeviceChange);
    };
  }, [microphonePermission.status, liveInputConfig, handleLiveInputUnavailable, applyOutputRouting]);

  return {
    microphonePermission,
    audioInputDevices,
    audioOutputDevices,
    requestMicrophonePermission,
    refreshAudioDevices,
    refreshAudioOutputDevices,
    setOutputDevice,
    handleLiveInputUnavailable,
  };
}
