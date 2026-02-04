export const readModel = (
  file: File
): Promise<string | ArrayBuffer | null> => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();

    reader.onload = function (e: ProgressEvent<FileReader>) {
      const content = e.target?.result;
      resolve(content ?? null);
    };

    reader.onerror = function () {
      reject(new Error(`Failed to read file: ${file.name}`));
    };

    reader.onabort = function () {
      reject(new Error(`File read aborted: ${file.name}`));
    };

    reader.readAsText(file);
  });
};
