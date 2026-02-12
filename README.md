# NeuralAmpModelerCore

[![Build](https://github.com/sdatkinson/NeuralAmpModelerCore/actions/workflows/build.yml/badge.svg)](https://github.com/sdatkinson/NeuralAmpModelerCore/actions/workflows/build.yml)

Core C++ DSP library for NAM plugins.

For an example how to use, see [NeuralAmpModelerPlugin](https://github.com/sdatkinson/NeuralAmpModelerPlugin).

## Compiling

Configure and build:

- `cmake -S . -B build`
- `cmake --build build`

Release builds:

- Single-config generators (Unix Makefiles, Ninja):
  - `cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release`
  - `cmake --build build-release --target nam_static nam_shared tools`
- Multi-config generators (Visual Studio, Xcode):
  - `cmake -S . -B build`
  - `cmake --build build --config Release --target nam_static nam_shared tools`

Build only the NAM libraries:

- `cmake --build build --target nam_static`
- `cmake --build build --target nam_shared`

The public C API for external projects is exposed by:

- `NAM/nam_c_api.h`

Simple example using the C API (linked against the static library):

- build target: `cmake --build build --target c_api_example`
- run: `./build/tools/c_api_example example_models/wavenet.nam`

You can also install libraries + C API header with:

- `cmake --install build`

## Testing
A workflow for testing the library is provided in `.github/workflows/build.yml`.
You should be able to run it locally to test if you'd like.

## Sharp edges
This library uses [Eigen](http://eigen.tuxfamily.org) to do the linear algebra routines that its neural networks require. Since these models hold their parameters as eigen object members, there is a risk with certain compilers and compiler optimizations that their memory is not aligned properly. This can be worked around by providing two preprocessor macros: `EIGEN_MAX_ALIGN_BYTES 0` and `EIGEN_DONT_VECTORIZE`, though this will probably harm performance. See [Structs Having Eigen Members](http://eigen.tuxfamily.org/dox-3.2/group__TopicStructHavingEigenMembers.html) for more information. This is being tracked as [Issue 67](https://github.com/sdatkinson/NeuralAmpModelerCore/issues/67).

## Sponsors

<div align="center">
  <img src="media/tone3000-logo.svg" alt="Tone3000 logo">
</div>

Development of version 0.4.0 of this library has been generously supported by [TONE3000](https://tone3000.com). 
**Thank you!**
