export const readModel = (
  file: File
): Promise<string | ArrayBuffer | null | undefined> => {
  return new Promise(resolve => {
    const reader = new FileReader();

    reader.onload = function (e: ProgressEvent<FileReader>) {
      const content = e.target?.result;
      resolve(content);
    };

    reader.readAsText(file);
  });
};
