const ua =
  typeof navigator !== 'undefined' ? navigator.userAgent.toLowerCase() : '';

export const isFirefox = ua.includes('firefox');
export const isSafari =
  ua.includes('safari') &&
  !ua.includes('chrome') &&
  !ua.includes('firefox') &&
  !ua.includes('edg');
export const isEdge = ua.includes('edg');
export const isChrome =
  (ua.includes('chrome') || ua.includes('crios')) && !ua.includes('edg');
export const needsMediaStreamWorkaround = isFirefox || isSafari;

export type BrowserName = 'chrome' | 'safari' | 'firefox' | 'edge' | 'other';

export function getBrowserName(): BrowserName {
  if (isEdge) return 'edge';
  if (isFirefox) return 'firefox';
  if (isSafari) return 'safari';
  if (isChrome) return 'chrome';
  return 'other';
}
