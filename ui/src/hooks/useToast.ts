import { useSyncExternalStore } from 'react';

const DEFAULT_DURATION_MS = 3000;

let currentMessage: string | null = null;
let timer: ReturnType<typeof setTimeout>;
const listeners = new Set<() => void>();

function notify() {
  listeners.forEach(l => l());
}

/** Show a toast message that auto-dismisses. */
export function showToast(message: string, durationMs = DEFAULT_DURATION_MS) {
  clearTimeout(timer);
  currentMessage = message;
  notify();
  timer = setTimeout(() => {
    currentMessage = null;
    notify();
  }, durationMs);
}

/** Subscribe to the current toast message. Returns the message or null. */
export function useToast(): string | null {
  return useSyncExternalStore(
    (cb) => { listeners.add(cb); return () => listeners.delete(cb); },
    () => currentMessage,
    () => null,
  );
}
