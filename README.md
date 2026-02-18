# NeuralAmpModelerCore

[![Build](https://github.com/sdatkinson/NeuralAmpModelerCore/actions/workflows/build.yml/badge.svg)](https://github.com/sdatkinson/NeuralAmpModelerCore/actions/workflows/build.yml)

Core C++ DSP library for NAM plugins.

For an example how to use, see [NeuralAmpModelerPlugin](https://github.com/sdatkinson/NeuralAmpModelerPlugin).

## Included Tools

There are a couple tools that exist to help you use this repo. 
For guidance on building them, have a look at the workflow provided in `.github/workflows/build.yml`.

* [`run_tests`](https://github.com/sdatkinson/NeuralAmpModelerCore/blob/761fa968766bcf67d3035320c195969d9ba41fa1/tools/CMakeLists.txt#L15), which runs a suite of unit tests.
* [`loadmodel`](https://github.com/sdatkinson/NeuralAmpModelerCore/blob/761fa968766bcf67d3035320c195969d9ba41fa1/tools/CMakeLists.txt#L13), which allows you to test loading a `.nam` file.
* [`benchmodel`](https://github.com/sdatkinson/NeuralAmpModelerCore/blob/761fa968766bcf67d3035320c195969d9ba41fa1/tools/CMakeLists.txt#L14), which allows you to test how quickly a model runs in real time. _Note: For more granular profiling tools, check out the [`main-profiling`](https://github.com/sdatkinson/NeuralAmpModelerCore/tree/main-profiling) branch.

## Sharp edges
This library uses [Eigen](http://eigen.tuxfamily.org) to do the linear algebra routines that its neural networks require. Since these models hold their parameters as eigen object members, there is a risk with certain compilers and compiler optimizations that their memory is not aligned properly. This can be worked around by providing two preprocessor macros: `EIGEN_MAX_ALIGN_BYTES 0` and `EIGEN_DONT_VECTORIZE`, though this will probably harm performance. See [Structs Having Eigen Members](http://eigen.tuxfamily.org/dox-3.2/group__TopicStructHavingEigenMembers.html) for more information. This is being tracked as [Issue 67](https://github.com/sdatkinson/NeuralAmpModelerCore/issues/67).

## Sponsors

<div align="center">
  <img src="media/tone3000-logo.svg" alt="Tone3000 logo">
</div>

Development of version 0.4.0 of this library has been generously supported by [TONE3000](https://tone3000.com). 
**Thank you!**
