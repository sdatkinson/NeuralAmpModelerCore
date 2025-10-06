interface ModuleType {
  asm: unknown;
  [key: string]: any; // @todo: Add more specific types
}

declare global {
  interface Window {
    Module?: ModuleType;
  }
}

export const useModule = (): (() => Promise<ModuleType>) => {
  const modulePromise = (): Promise<ModuleType> => {
    return new Promise((resolve, reject) => {
      // console.log('Starting to load WASM module - ' + JSON.stringify({
      //   hasWebAssembly: !!window.WebAssembly,
      //   hasSharedArrayBuffer: typeof SharedArrayBuffer !== 'undefined',
      //   userAgent: navigator.userAgent,
      //   hasWorker: !!window.Worker,
      //   hasAudioWorklet: !!window.AudioWorklet
      // }));

      // Add error handler for worker loading
      window.addEventListener(
        'error',
        e => {
          console.error(
            'Global error: ' +
              JSON.stringify({
                message: e.message,
                filename: e.filename,
                lineno: e.lineno,
                colno: e.colno,
                error: e.error?.toString(),
              })
          );
        },
        false
      );

      // Add error handler for worker
      window.addEventListener('unhandledrejection', e => {
        console.error(
          'Unhandled rejection: ' +
            JSON.stringify({
              reason: e.reason?.toString(),
              promise: 'Promise rejected',
            })
        );
      });

      let attempts = 0;
      const maxAttempts = 100;

      const interval = setInterval(() => {
        attempts++;
        const { Module } = window;

        if (Module) {
          // console.log('Module found - ' + JSON.stringify({
          //   hasModule: true,
          //   keys: Object.keys(Module),
          //   asmKeys: Module.asm ? Object.keys(Module.asm) : 'no asm',
          //   wasmMemory: !!Module.wasmMemory,
          //   wasmTable: !!Module.wasmTable,
          //   malloc: !!Module._malloc,
          //   stringToUTF8: !!Module.stringToUTF8,
          //   ccall: !!Module.ccall,
          //   asm: !!Module.asm,
          //   moduleLoaded: !!Module.moduleLoaded,
          //   calledRun: !!Module.calledRun,
          //   runtimeInitialized: !!Module.runtimeInitialized,
          //   wasmBinary: !!Module.wasmBinary,
          //   mainScriptUrlOrBlob: !!Module.mainScriptUrlOrBlob
          // }));

          if (Module._malloc && Module.stringToUTF8 && Module.ccall) {
            // console.log('WASM Module loaded with required functions');
            resolve(Module);
            clearInterval(interval);
          } else {
            console.log(
              'Module missing functions - ' +
                JSON.stringify({
                  malloc: !!Module._malloc,
                  stringToUTF8: !!Module.stringToUTF8,
                  ccall: !!Module.ccall,
                  runtimeInitialized: !!Module.runtimeInitialized,
                  calledRun: !!Module.calledRun,
                  hasWasmMemory: !!Module.wasmMemory,
                  hasWasmTable: !!Module.wasmTable,
                })
            );
          }
        }

        if (attempts >= maxAttempts) {
          const finalState = window.Module
            ? JSON.stringify({
                keys: Object.keys(window.Module),
                asmKeys: window.Module.asm
                  ? Object.keys(window.Module.asm)
                  : 'no asm',
                wasmMemory: !!window.Module.wasmMemory,
                wasmTable: !!window.Module.wasmTable,
                runtimeInitialized: !!window.Module.runtimeInitialized,
                calledRun: !!window.Module.calledRun,
                wasmBinary: !!window.Module.wasmBinary,
                mainScriptUrlOrBlob: !!window.Module.mainScriptUrlOrBlob,
              })
            : 'no Module';

          console.error('Module load timeout - final state: ' + finalState);
          clearInterval(interval);
          reject(new Error('Failed to load WASM module - timeout'));
        }
      }, 100);

      return () => clearInterval(interval);
    });
  };

  return modulePromise;
};
