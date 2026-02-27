/**
 * Map raw MediaDeviceInfo to simplified { deviceId, label } objects.
 * Provides a fallback label when the browser doesn't expose one.
 */
export function mapDevices(
  allDevices: MediaDeviceInfo[],
  kind: 'audioinput' | 'audiooutput'
): Array<{ deviceId: string; label: string }> {
  const prefix = kind === 'audioinput' ? 'Input' : 'Output';
  return allDevices
    .filter(device => device.kind === kind)
    .map(device => ({
      deviceId: device.deviceId,
      label: device.label || `${prefix} ${device.deviceId.slice(0, 8)}`,
    }));
}
