# Neural Amp Modeler Wasm

This is a [TONE3000](https://tone3000.com) fork of [Steve Atkinson's Neural Amp Modeler Core](https://github.com/sdatkinson/NeuralAmpModelerCore) DSP library, specifically adapted to run Neural Amp Modeler inference in web browsers using WebAssembly. This enables real-time guitar amp modeling directly in the browser without requiring native plugins.

The original Neural Amp Modeler Core is a C++ DSP library for NAM plugins. This fork extends its capabilities to the web platform, allowing you to run Neural Amp Modeler models in any modern web browser.

## Testing
A workflow for testing the library is provided in `.github/workflows/build.yml`.
You should be able to run it locally to test if you'd like.

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

The component must be wrapped in a `T3kPlayerContextProvider` which handles the WebAssembly module initialization and audio context setup. The provider manages the audio processing pipeline and ensures proper cleanup of resources.

The component accepts the following props:
- `models`: Array of model objects, each containing:
  - `name`: Display name for the model
  - `model_url`: URL to the NAM model file
- `irs`: Array of IR (Impulse Response) objects, each containing:
  - `name`: Display name for the IR
  - `ir_url`: URL to the IR file
  - `mix`: Optional wet/dry mix ratio (0-1)
  - `gain`: Optional gain adjustment
- `inputs`: Array of input audio objects, each containing:
  - `name`: Display name for the input
  - `input_url`: URL to the audio file
- `isLoading`: Optional boolean to show loading state

## Sharp edges
This library uses [Eigen](http://eigen.tuxfamily.org) to do the linear algebra routines that its neural networks require. Since these models hold their parameters as eigen object members, there is a risk with certain compilers and compiler optimizations that their memory is not aligned properly. This can be worked around by providing two preprocessor macros: `EIGEN_MAX_ALIGN_BYTES 0` and `EIGEN_DONT_VECTORIZE`, though this will probably harm performance. See [Structs Having Eigen Members](http://eigen.tuxfamily.org/dox-3.2/group__TopicStructHavingEigenMembers.html) for more information. This is being tracked as [Issue 67](https://github.com/sdatkinson/NeuralAmpModelerCore/issues/67).
