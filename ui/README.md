# Neural Amp Modeler WebAssembly React Component

This is a [TONE3000](https://tone3000.com) fork of [Steve Atkinson's Neural Amp Modeler Core](https://github.com/sdatkinson/NeuralAmpModelerCore) DSP library, specifically adapted to run Neural Amp Modeler inference in web browsers using WebAssembly. This enables real-time audio modeling directly in the browser without requiring native plugins.

The original Neural Amp Modeler Core is a C++ DSP library for NAM plugins. This fork extends its capabilities to the web platform, allowing you to run Neural Amp Modeler models in any modern web browser.

![screenshot](https://raw.githubusercontent.com/tone-3000/neural-amp-modeler-wasm/refs/heads/main/ui/public/screenshot.png)

## Installation

```bash
npm install neural-amp-modeler-wasm
```

## WASM Files Setup

Before using the component, you need to host the WebAssembly files at the root of your project. These files are required for the component to function:

1. Copy the following files from the repository to your project's public directory:
   - [t3k-wasm-module.js](https://raw.githubusercontent.com/tone-3000/neural-amp-modeler-wasm/refs/heads/main/ui/public/t3k-wasm-module.js)
   - [t3k-wasm-module.wasm](https://raw.githubusercontent.com/tone-3000/neural-amp-modeler-wasm/refs/heads/main/ui/public/t3k-wasm-module.wasm)
   - [t3k-wasm-module.worker.js](https://raw.githubusercontent.com/tone-3000/neural-amp-modeler-wasm/refs/heads/main/ui/public/t3k-wasm-module.worker.js)
   - [t3k-wasm-module.aw.js](https://raw.githubusercontent.com/tone-3000/neural-amp-modeler-wasm/refs/heads/main/ui/public/t3k-wasm-module.aw.js)
   - [t3k-wasm-module.ww.js](https://raw.githubusercontent.com/tone-3000/neural-amp-modeler-wasm/refs/heads/main/ui/public/t3k-wasm-module.ww.js)

2. Make sure these files are accessible at the root of your web server (e.g., `https://your-domain.com/t3k-wasm-module.wasm`)

## Usage

```tsx
import {
  T3kPlayer,
  T3kPlayerContextProvider,
  PREVIEW_MODE,
} from 'neural-amp-modeler-wasm';
import 'neural-amp-modeler-wasm/dist/styles.css';

function App() {
  return (
    <T3kPlayerContextProvider>
      <T3kPlayer
        models={[
          {
            name: 'Vox AC10',
            url: 'https://raw.githubusercontent.com/tone-3000/neural-amp-modeler-wasm/refs/heads/main/ui/public/models/ac10.nam',
            default: true,
          },
          {
            name: 'Fender Deluxe Reverb',
            url: 'https://raw.githubusercontent.com/tone-3000/neural-amp-modeler-wasm/refs/heads/main/ui/public/models/deluxe.nam',
          },
        ]}
        irs={[
          {
            name: 'None',
            url: '',
          },
          {
            name: 'Celestion',
            url: 'https://raw.githubusercontent.com/tone-3000/neural-amp-modeler-wasm/refs/heads/main/ui/public/irs/celestion.wav',
            default: true,
          },
          {
            name: 'EMT 140 Plate Reverb',
            url: 'https://raw.githubusercontent.com/tone-3000/neural-amp-modeler-wasm/refs/heads/main/ui/public/irs/plate.wav',
            mix: 0.5, // Optional: wet/dry mix (0-1)
            gain: 1.0, // Optional: gain adjustment
          },
        ]}
        inputs={[
          {
            name: 'Mayer - Guitar',
            url: 'https://raw.githubusercontent.com/tone-3000/neural-amp-modeler-wasm/refs/heads/main/ui/public/inputs/Mayer%20-%20Guitar.wav',
            default: true,
          },
          {
            name: 'Downtown - Bass',
            url: 'https://raw.githubusercontent.com/tone-3000/neural-amp-modeler-wasm/refs/heads/main/ui/public/inputs/Downtown%20-%20Bass.wav',
          },
        ]}
        previewMode={PREVIEW_MODE.MODEL}
        isLoading={false}
        onPlay={({ model, ir, input }) => {
          console.log('Playing with:', { model, ir, input });
        }}
        onModelChange={model => {
          console.log('Model changed to:', model);
        }}
        onInputChange={input => {
          console.log('Input changed to:', input);
        }}
        onIrChange={ir => {
          console.log('IR changed to:', ir);
        }}
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
- `url`: URL to the NAM model file
- `default`: Optional boolean to mark as default selection

### irs

Array of IR (Impulse Response) objects, each containing:

- `name`: Display name for the IR
- `url`: URL to the IR file (use empty string for "None")
- `mix`: Optional wet/dry mix ratio (0-1)
- `gain`: Optional gain adjustment
- `default`: Optional boolean to mark as default selection

### inputs

Array of input audio objects, each containing:

- `name`: Display name for the input
- `url`: URL to the audio file
- `default`: Optional boolean to mark as default selection

### previewMode

Optional enum value to control the preview mode:

- `PREVIEW_MODE.MODEL`: Show model selection interface (default)
- `PREVIEW_MODE.IR`: Show IR selection interface

### isLoading

Optional boolean to show loading state

### Event Callbacks

#### onPlay

Callback function triggered when audio playback starts:

```tsx
onPlay?: ({ model, ir, input }: {
  model: Model,
  ir: IR,
  input: Input
}) => void;
```

#### onModelChange

Callback function triggered when model selection changes:

```tsx
onModelChange?: (model: Model) => void;
```

#### onInputChange

Callback function triggered when input selection changes:

```tsx
onInputChange?: (input: Input) => void;
```

#### onIrChange

Callback function triggered when IR selection changes:

```tsx
onIrChange?: (ir: IR) => void;
```

## Requirements

- Your web server must include the following CORS headers to enable WebAssembly and SharedArrayBuffer support:
  ```http
  Cross-Origin-Embedder-Policy: require-corp
  Cross-Origin-Opener-Policy: same-origin
  ```
  These headers are required because WebAssembly audio processing uses SharedArrayBuffer, which requires these security policies to be enabled.

## Development

This package is part of the [neural-amp-modeler-wasm](https://github.com/tone-3000/neural-amp-modeler-wasm) project, which includes both the WebAssembly compilation code and the React UI components. For more information about the project structure and development setup, please refer to the [main repository](https://github.com/tone-3000/neural-amp-modeler-wasm).

## License

MIT
