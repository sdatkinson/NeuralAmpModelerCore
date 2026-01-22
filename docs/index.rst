NeuralAmpModelerCore Documentation
===================================

Welcome to the NeuralAmpModelerCore documentation. This library provides a core C++ DSP implementation for Neural Amp Modeler plugins.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   wavenet_walkthrough
   api/index

Overview
--------

NeuralAmpModelerCore is a high-performance C++ library for running neural network-based audio processing models. It supports multiple architectures including:

* **WaveNet**: Dilated convolutional neural networks with gating and conditioning
* **ConvNet**: Convolutional neural networks with batch normalization
* **LSTM**: Long Short-Term Memory networks
* **Linear**: Simple linear models (impulse responses)

The library is designed for real-time audio processing with a focus on:

* **Real-time safety**: Pre-allocated buffers and no dynamic allocations during processing
* **Performance**: Optimized implementations using Eigen for linear algebra
* **Flexibility**: Support for various activation functions, gating modes, and conditioning mechanisms

Getting Started
---------------

For an example of how to use this library, see the `NeuralAmpModelerPlugin <https://github.com/sdatkinson/NeuralAmpModelerPlugin>`_ repository.

Architecture
------------

The library is organized into several namespaces:

* :ref:`nam::wavenet <namespace_nam_wavenet>`: WaveNet architecture implementation
* :ref:`nam::convnet <namespace_nam_convnet>`: ConvNet architecture implementation
* :ref:`nam::lstm <namespace_nam_lstm>`: LSTM architecture implementation
* :ref:`nam::activations <namespace_nam_activations>`: Activation function implementations
* :ref:`nam::gating_activations <namespace_nam_gating_activations>`: Gating and blending activation functions

Key Components
--------------

* :ref:`DSP <class_nam_dsp>`: Base class for all DSP models
* :ref:`WaveNet <class_nam_wavenet_wavenet>`: Main WaveNet model class
* :ref:`Conv1D <class_nam_conv1d>`: Dilated 1D convolution implementation
* :ref:`FiLM <class_nam_film>`: Feature-wise Linear Modulation

Documentation
-------------

* :doc:`wavenet_walkthrough`: Step-by-step explanation of WaveNet architecture, LayerArray, and Layer computations
* :doc:`api/index`: Complete API reference

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
