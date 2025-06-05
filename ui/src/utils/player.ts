export const upperCaseFirst = (str: string) => str.charAt(0).toUpperCase() + str.slice(1);

export const getExtension = (url: string) => {
  if (!url) return undefined;
  return url.split('.').pop();
};