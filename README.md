# Neural Amp Modeler Wasm

[![npm version](https://img.shields.io/npm/v/neural-amp-modeler-wasm.svg)](https://www.npmjs.com/package/neural-amp-modeler-wasm)
[![Build](https://github.com/sdatkinson/NeuralAmpModelerCore/actions/workflows/build.yml/badge.svg)](https://github.com/sdatkinson/NeuralAmpModelerCore/actions/workflows/build.yml)

This is a [TONE3000](https://tone3000.com) fork of [Steve Atkinson's Neural Amp Modeler Core](https://github.com/sdatkinson/NeuralAmpModelerCore) DSP library, specifically adapted to run Neural Amp Modeler inference in web browsers using WebAssembly. This enables real-time audio modeling directly in the browser without requiring native plugins.

The original Neural Amp Modeler Core is a C++ DSP library for NAM plugins. This fork extends its capabilities to the web platform, allowing you to run Neural Amp Modeler models in any modern web browser.

![screenshot](https://raw.githubusercontent.com/tone-3000/neural-amp-modeler-wasm/refs/heads/main/ui/public/screenshot.png)

## Included Tools

This repo includes a few helpful tools.
For guidance on building them, have a look at the workflow provided in `.github/workflows/build.yml`.

* [`run_tests`](https://github.com/sdatkinson/NeuralAmpModelerCore/blob/761fa968766bcf67d3035320c195969d9ba41fa1/tools/CMakeLists.txt#L15), which runs a suite of unit tests.
* [`loadmodel`](https://github.com/sdatkinson/NeuralAmpModelerCore/blob/761fa968766bcf67d3035320c195969d9ba41fa1/tools/CMakeLists.txt#L13), which allows you to test loading a `.nam` file.
* [`benchmodel`](https://github.com/sdatkinson/NeuralAmpModelerCore/blob/761fa968766bcf67d3035320c195969d9ba41fa1/tools/CMakeLists.txt#L14), which allows you to test how quickly a model runs in real time. _Note: For more granular profiling tools, check out the [`main-profiling`](https://github.com/sdatkinson/NeuralAmpModelerCore/tree/main-profiling) branch.

## Building
Before building the project, you need to initialize the required submodules:

```bash
git submodule update --init --recursive
```

This will fetch and initialize the Eigen library and other dependencies required for building.

## WebAssembly (WASM) Build
To build the WebAssembly version of the library, you'll need to install Emscripten and Node.js first:

1. Install Node.js and npm (which includes npx):
   ```bash
   # On macOS with Homebrew
   brew install node

   # On Ubuntu/Debian
   curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
   sudo apt-get install -y nodejs

   # On Windows
   # Download and install from https://nodejs.org/
   ```

2. Install Emscripten:
   ```bash
   # Clone the emsdk repository
   git clone https://github.com/emscripten-core/emsdk.git
   cd emsdk
   
   # Install and activate Emscripten
   ./emsdk install 3.1.41
   ./emsdk activate 3.1.41
   
   # Set up the environment variables
   source ./emsdk_env.sh

   # Add Emscripten to your PATH permanently
   # For bash/zsh, add this line to your ~/.bashrc or ~/.zshrc:
   echo 'source "$HOME/emsdk/emsdk_env.sh"' >> ~/.bashrc  # or ~/.zshrc
   # Then reload your shell configuration:
   source ~/.bashrc  # or source ~/.zshrc
   ```

3. Build the WASM version:
   ```bash
   cd wasm
   # Run the WASM build script
   ./build.bash
   ```

The build script will create the WebAssembly files in the `build/wasm` directory. The main output files will be:
- `t3k-wasm-module.js` - JavaScript wrapper
- `t3k-wasm-module.wasm` - WebAssembly binary
- `t3k-wasm-module.worker.js` - Web Worker implementation for parallel processing
- `t3k-wasm-module.aw.js` - Audio Worklet implementation for real-time audio processing
- `t3k-wasm-module.ww.js` - Web Worker wrapper for thread management

## UI Development
The project includes a React-based UI for testing and demonstrating the WebAssembly implementation. To build and run the UI:

1. First, ensure you have built the WebAssembly module as described above.

2. Install UI dependencies:
   ```bash
   cd ui
   npm install
   ```

3. Start the development server:
   ```bash
   npm start
   ```
   This will:
   - Copy the WebAssembly files to the public directory
   - Start a development server with hot reloading
   - Open the UI in your default browser at `http://localhost:3000`

4. To build the UI for production:
   ```bash
   npm run build
   ```
   This will create a production build in the `ui/dist` directory.

## Using the T3kPlayer Component
The project exports a React component that can be used in other projects. To use it:

1. Install the package:
   ```bash
   npm install neural-amp-modeler-wasm
   ```

2. Import and use the component in your React application:
   ```tsx
   import { T3kPlayer, T3kPlayerProvider } from 'neural-amp-modeler-wasm';
   import 'neural-amp-modeler-wasm/dist/styles.css';

   function App() {
     return (
       <T3kPlayerProvider>
         <T3kPlayer
           models={[
             {
               name: "Vox AC10",
               url: "https://raw.githubusercontent.com/tone-3000/neural-amp-modeler-wasm/refs/heads/main/ui/public/models/ac10.nam",
               default: true
             },
             {
               name: "Fender Deluxe Reverb",
               url: "https://raw.githubusercontent.com/tone-3000/neural-amp-modeler-wasm/refs/heads/main/ui/public/models/deluxe.nam"
             }
           ]}
           irs={[
             {
               name: "Celestion",
               url: "https://raw.githubusercontent.com/tone-3000/neural-amp-modeler-wasm/refs/heads/main/ui/public/irs/celestion.wav",
               default: true
             },
             {
               name: "EMT 140 Plate Reverb",
               url: "https://raw.githubusercontent.com/tone-3000/neural-amp-modeler-wasm/refs/heads/main/ui/public/irs/plate.wav",
               mix: 0.5,  // Optional: wet/dry mix (0-1)
               gain: 1.0  // Optional: gain adjustment
             }
           ]}
           inputs={[
             {
               name: "Mayer - Guitar",
               url: "https://raw.githubusercontent.com/tone-3000/neural-amp-modeler-wasm/refs/heads/main/ui/public/inputs/Mayer%20-%20Guitar.wav",
               default: true
             },
             {
               name: "Downtown - Bass",
               url: "https://raw.githubusercontent.com/tone-3000/neural-amp-modeler-wasm/refs/heads/main/ui/public/inputs/Downtown%20-%20Bass.wav"
             }
           ]}
         />
       </T3kPlayerProvider>
     );
   }
   ```

The component must be wrapped in a `T3kPlayerProvider` which handles the WebAssembly module initialization, audio context setup, and includes the settings dialog (rendered once for all players on the page). The provider manages the audio processing pipeline and ensures proper cleanup of resources.

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
