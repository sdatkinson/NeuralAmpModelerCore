# Neural Amp Modeler WebAssembly React Component

This is a TONE3000 fork of [Steve Atkinson's Neural Amp Modeler Core](https://github.com/sdatkinson/NeuralAmpModelerCore) DSP library, specifically adapted to run Neural Amp Modeler inference in web browsers using WebAssembly. This enables real-time guitar amp modeling directly in the browser without requiring native plugins.

The original Neural Amp Modeler Core is a C++ DSP library for NAM plugins. This fork extends its capabilities to the web platform, allowing you to run Neural Amp Modeler models in any modern web browser.

![Neural Amp Modeler Wasm](https://raw.githubusercontent.com/tone-3000/neural-amp-modeler-wasm/main/docs/nam-wasm.png)

## Installation

```bash
npm install neural-amp-modeler-wasm
```

## WASM Files Setup

Before using the component, you need to host the WebAssembly files at the root of your project. These files are required for the component to function:

1. Copy the following files from the `build/wasm` directory to your project's public directory:
   - `t3k-wasm-module.js`
   - `t3k-wasm-module.wasm`
   - `t3k-wasm-module.worker.js`
   - `t3k-wasm-module.aw.js`
   - `t3k-wasm-module.ww.js`

2. Make sure these files are accessible at the root of your web server (e.g., `https://your-domain.com/t3k-wasm-module.wasm`)

## Usage

```tsx
import { T3kPlayer, T3kPlayerContextProvider } from 'neural-amp-modeler-wasm';
import 'neural-amp-modeler-wasm/dist/styles.css';

function App() {
  return (
    <T3kPlayerContextProvider>
      <T3kPlayer
        models={[
          {
            name: "Vox AC10",
            model_url: "https://www.tone3000.com/nams/ac10.nam"
          },
          {
            name: "Fender Deluxe Reverb",
            model_url: "https://www.tone3000.com/nams/deluxe.nam"
          }
        ]}
        irs={[
          {
            name: "Celestion",
            ir_url: "https://www.tone3000.com/irs/celestion.wav"
          },
          {
            name: "EMT 140 Plate Reverb",
            ir_url: "https://www.tone3000.com/irs/plate.wav",
            mix: 0.5,  // Optional: wet/dry mix (0-1)
            gain: 1.0  // Optional: gain adjustment
          }
        ]}
        inputs={[
          {
            name: "Mayer - Guitar",
            input_url: "https://www.tone3000.com/samples/Mayer%20-%20Guitar.wav"
          },
          {
            name: "Downtown - Bass",
            input_url: "https://www.tone3000.com/samples/Downtown%20-%20Bass.wav"
          }
        ]}
        isLoading={false}
      />
    </T3kPlayerContextProvider>
  );
}
```

## Component Props

The `T3kPlayer` component accepts the following props:

### models
Array of model objects, each containing:
- `name`: Display name for the model
- `model_url`: URL to the NAM model file

### irs
Array of IR (Impulse Response) objects, each containing:
- `name`: Display name for the IR
- `ir_url`: URL to the IR file
- `mix`: Optional wet/dry mix ratio (0-1)
- `gain`: Optional gain adjustment

### inputs
Array of input audio objects, each containing:
- `name`: Display name for the input
- `input_url`: URL to the audio file

### isLoading
Optional boolean to show loading state

## Requirements

- React 18 or higher
- Modern browser with WebAssembly support
- The WebAssembly files must be hosted at the root of your project

## Features

- WebAssembly-powered audio processing
- Real-time guitar amp modeling
- React-based UI components
- TypeScript support
- Built-in audio player with model switching
- Impulse Response (IR) support
- Multiple input audio support

## Development

This package is part of the [neural-amp-modeler-wasm](https://github.com/tone-3000/neural-amp-modeler-wasm) project, which includes both the WebAssembly compilation code and the React UI components. For more information about the project structure and development setup, please refer to the [main repository](https://github.com/tone-3000/neural-amp-modeler-wasm).

## License

MIT 