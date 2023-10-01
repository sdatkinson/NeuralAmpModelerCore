# NeuralAmpModelerCore
Core DSP library for NAM plugins.

For an example how to use, see [NeuralAmpModelerPlugin](https://github.com/sdatkinson/NeuralAmpModelerPlugin).

_The general Audio DSP tools from version 0.0 have been moved to [AudioDSPTools](https://github.com/sdatkinson/AudioDSPTools)._

## Sharp edges
This library uses [Eigen](http://eigen.tuxfamily.org) to do the linear algebra routines that its neural networks require. Since these models hold their parameters as eigen object members, there is a risk with certain compilers and compiler optimizations that their memory is not aligned properly. This can be worked around by providing two preprocessor macros: `EIGEN_MAX_ALIGN_BYTES 0` and `EIGEN_DONT_VECTORIZE`, though this will probably harm performance. See [Structs Having Eigen Members](http://eigen.tuxfamily.org/dox-3.2/group__TopicStructHavingEigenMembers.html) for more information. This is being tracked as [Issue 67](https://github.com/sdatkinson/NeuralAmpModelerCore/issues/67).
