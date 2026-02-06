const ua = typeof navigator !== 'undefined' ? navigator.userAgent.toLowerCase() : '';

export const isFirefox = ua.includes('firefox');
export const isSafari = ua.includes('safari') && !ua.includes('chrome');
export const needsMediaStreamWorkaround = isFirefox || isSafari;
