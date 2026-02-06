export const upperCaseFirst = (str: string) =>
  str.charAt(0).toUpperCase() + str.slice(1);

export const getExtension = (url: string) => {
  if (!url) return undefined;
  return url.split('.').pop();
};

export function formatTime(time: number): string {
  const minutes = Math.floor(time / 60);
  const seconds = Math.floor(time % 60);
  return `${minutes}:${seconds.toString().padStart(2, '0')}`;
}

export function getDefault<T extends { default?: boolean }>(items: T[]): T {
  return items.find(item => item.default) || items[0];
}
