WaveNet Computation Walkthrough
==================================

This document provides a detailed step-by-step explanation of how the WaveNet architecture performs its computations, including the LayerArray and Layer objects that make up a WaveNet model.

.. note::
   This walkthrough is specific to the WaveNet architecture. Walkthroughs for other architectures (ConvNet, LSTM, etc.) will be added in future documentation updates.

WaveNet Overview
----------------

WaveNet is a dilated convolutional neural network architecture designed for audio processing. The model consists of:

* **Multiple LayerArrays**: Each LayerArray contains multiple layers with the same channel configuration
* **Conditioning**: Optional DSP processing of the input to generate conditioning signals
* **Residual and Skip Connections**: Information flows through both residual (layer-to-layer) and skip (to head) paths

.. mermaid::
   :caption: High-level WaveNet Architecture

   graph TD
       Input[Audio Input] --> ConditionDSP{Condition DSP<br/>Optional}
       ConditionDSP --> Condition[Condition Signal]
       Input --> LayerArray1[LayerArray 1]
       Condition --> LayerArray1
       LayerArray1 --> LayerArray2[LayerArray 2]
       Condition --> LayerArray2
       LayerArray2 --> LayerArrayN[LayerArray N]
       Condition --> LayerArrayN
       LayerArray1 --> Head[Head Output]
       LayerArray2 --> Head
       LayerArrayN --> Head
       Head --> Scale[Head Scale]
       Scale --> Output[Audio Output]

Layer Computation
-----------------

A single Layer performs the core computation of a WaveNet block. The computation proceeds through several stages:

Step 1: Input Convolution
~~~~~~~~~~~~~~~~~~~~~~~~~~

The input first goes through a dilated 1D convolution:

1. **Optional Pre-FiLM**: If `conv_pre_film` is active, the input is modulated by the condition signal before convolution
2. **Dilated Convolution**: The input is convolved with a dilated kernel (dilation factor increases with layer depth)
3. **Optional Post-FiLM**: If `conv_post_film` is active, the convolution output is modulated by the condition signal

.. code-block:: cpp
   :caption: Input convolution processing

   if (this->_conv_pre_film) {
       this->_conv_pre_film->Process(input, condition, num_frames);
       this->_conv.Process(this->_conv_pre_film->GetOutput(), num_frames);
   } else {
       this->_conv.Process(input, num_frames);
   }
   if (this->_conv_post_film) {
       Eigen::MatrixXf& conv_output = this->_conv.GetOutput();
       this->_conv_post_film->Process_(conv_output, condition, num_frames);
   }

Step 2: Input Mixin
~~~~~~~~~~~~~~~~~~~

The conditioning input is processed separately and added to the convolution output:

1. **Optional Pre-FiLM**: If `input_mixin_pre_film` is active, the condition is modulated before the mixin convolution
2. **Input Mixin Convolution**: A 1x1 convolution processes the condition signal
3. **Optional Post-FiLM**: If `input_mixin_post_film` is active, the mixin output is modulated

.. code-block:: cpp
   :caption: Input mixin processing

   if (this->_input_mixin_pre_film) {
       this->_input_mixin_pre_film->Process(condition, condition, num_frames);
       this->_input_mixin.process_(this->_input_mixin_pre_film->GetOutput(), num_frames);
   } else {
       this->_input_mixin.process_(condition, num_frames);
   }
   if (this->_input_mixin_post_film) {
       Eigen::MatrixXf& input_mixin_output = this->_input_mixin.GetOutput();
       this->_input_mixin_post_film->Process_(input_mixin_output, condition, num_frames);
   }

Step 3: Sum and Pre-Activation FiLM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The convolution output and input mixin output are summed, and optionally modulated:

.. code-block:: cpp
   :caption: Sum and pre-activation FiLM

   this->_z.leftCols(num_frames).noalias() =
       _conv.GetOutput().leftCols(num_frames) + _input_mixin.GetOutput().leftCols(num_frames);
   if (this->_activation_pre_film) {
       this->_activation_pre_film->Process_(this->_z, condition, num_frames);
   }

Step 4: Activation
~~~~~~~~~~~~~~~~~~

The activation stage depends on the gating mode:

**No Gating (GatingMode::NONE)**
   Simple activation function applied to the summed output.

**Gated (GatingMode::GATED)**
   The output channels are doubled (2 * bottleneck). The top half goes through the primary activation,
   the bottom half through a secondary activation (typically sigmoid). The results are multiplied element-wise.

**Blended (GatingMode::BLENDED)**
   Similar to gated, but instead of multiplication, a weighted blend is performed:
   output = alpha * activated_input + (1 - alpha) * pre_activation_input
   where alpha comes from the secondary activation.

After activation, an optional post-activation FiLM may be applied.

.. code-block:: cpp
   :caption: Activation processing (gated mode example)

   if (this->_gating_mode == GatingMode::GATED) {
       auto input_block = this->_z.leftCols(num_frames);
       auto output_block = this->_z.topRows(bottleneck).leftCols(num_frames);
       this->_gating_activation->apply(input_block, output_block);
       if (this->_activation_post_film) {
           this->_activation_post_film->Process(this->_z.topRows(bottleneck), condition, num_frames);
           this->_z.topRows(bottleneck).leftCols(num_frames).noalias() =
               this->_activation_post_film->GetOutput().leftCols(num_frames);
       }
   }

Step 5: 1x1 Convolution
~~~~~~~~~~~~~~~~~~~~~~~~

A 1x1 convolution reduces the bottleneck channels back to the layer channel count:

.. code-block:: cpp
   :caption: 1x1 convolution

   _1x1.process_(this->_z.topRows(bottleneck), num_frames);
   if (this->_1x1_post_film) {
       Eigen::MatrixXf& _1x1_output = this->_1x1.GetOutput();
       this->_1x1_post_film->Process_(_1x1_output, condition, num_frames);
   }

Step 6: Head 1x1 (Optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

If a head1x1 convolution is configured, it processes the activated output for the skip connection:

.. code-block:: cpp
   :caption: Head 1x1 processing

   if (this->_head1x1) {
       this->_head1x1->process_(this->_z.topRows(bottleneck).leftCols(num_frames), num_frames);
       if (this->_head1x1_post_film) {
           Eigen::MatrixXf& head1x1_output = this->_head1x1->GetOutput();
           this->_head1x1_post_film->Process_(head1x1_output, condition, num_frames);
       }
       this->_output_head.leftCols(num_frames).noalias() = 
           this->_head1x1->GetOutput().leftCols(num_frames);
   }

Step 7: Residual and Skip Connections
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Finally, the outputs are computed:

* **Residual Connection**: `output_next_layer = input + 1x1_output`
* **Skip Connection**: `output_head = activated_output` (or head1x1 output if present)

.. code-block:: cpp
   :caption: Residual and skip connections

   // Store output to next layer (residual connection)
   this->_output_next_layer.leftCols(num_frames).noalias() =
       input.leftCols(num_frames) + _1x1.GetOutput().leftCols(num_frames);
   
   // Store output to head (skip connection)
   if (this->_head1x1) {
       this->_output_head.leftCols(num_frames).noalias() = 
           this->_head1x1->GetOutput().leftCols(num_frames);
   } else {
       this->_output_head.leftCols(num_frames).noalias() = 
           this->_z.topRows(bottleneck).leftCols(num_frames);
   }

Data Flow Diagram
~~~~~~~~~~~~~~~~~

.. mermaid::
   :caption: Layer Computation Flow

   graph LR
       Input[Input] --> PreFiLM1{Pre-FiLM?}
       PreFiLM1 -->|Yes| ConvPre[Conv Pre-FiLM]
       PreFiLM1 -->|No| Conv[Dilated Conv]
       ConvPre --> Conv
       Conv --> PostFiLM1{Post-FiLM?}
       PostFiLM1 -->|Yes| ConvPost[Conv Post-FiLM]
       PostFiLM1 -->|No| Sum[Sum]
       ConvPost --> Sum
       
       Condition[Condition] --> PreFiLM2{Pre-FiLM?}
       PreFiLM2 -->|Yes| MixinPre[Input Mixin Pre-FiLM]
       PreFiLM2 -->|No| Mixin[Input Mixin]
       MixinPre --> Mixin
       Mixin --> PostFiLM2{Post-FiLM?}
       PostFiLM2 -->|Yes| MixinPost[Input Mixin Post-FiLM]
       PostFiLM2 -->|No| Sum
       MixinPost --> Sum
       
       Sum --> PreActFiLM{Pre-Act FiLM?}
       PreActFiLM -->|Yes| PreAct[Pre-Activation FiLM]
       PreActFiLM -->|No| Act[Activation]
       PreAct --> Act
       
       Act --> PostActFiLM{Post-Act FiLM?}
       PostActFiLM -->|Yes| PostAct[Post-Activation FiLM]
       PostActFiLM -->|No| Conv1x1[1x1 Conv]
       PostAct --> Conv1x1
       
       Conv1x1 --> Post1x1FiLM{Post-1x1 FiLM?}
       Post1x1FiLM -->|Yes| Post1x1[Post-1x1 FiLM]
       Post1x1FiLM -->|No| Residual[Residual: Input + 1x1]
       Post1x1 --> Residual
       
       Act --> Head1x1{Head 1x1?}
       Head1x1 -->|Yes| HeadConv[Head 1x1 Conv]
       Head1x1 -->|No| Skip[Skip Output]
       HeadConv --> HeadFiLM{Head FiLM?}
       HeadFiLM -->|Yes| HeadPost[Head Post-FiLM]
       HeadFiLM -->|No| Skip
       HeadPost --> Skip

LayerArray Computation
----------------------

A LayerArray chains multiple Layer objects together, processing them sequentially with residual connections.

Step 1: Rechanneling
~~~~~~~~~~~~~~~~~~~~~

The input is first rechanneled to match the layer channel count:

.. code-block:: cpp
   :caption: Input rechanneling

   this->_rechannel.process_(layer_inputs, num_frames);
   Eigen::MatrixXf& rechannel_output = _rechannel.GetOutput();

Step 2: Layer Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~

Each layer processes the output of the previous layer:

1. **First Layer**: Processes the rechanneled input
2. **Subsequent Layers**: Process the residual output from the previous layer
3. **Head Accumulation**: Each layer's skip output is accumulated into the head buffer

.. code-block:: cpp
   :caption: Layer processing loop

   for (size_t i = 0; i < this->_layers.size(); i++) {
       if (i == 0) {
           // First layer consumes the rechannel output buffer
           this->_layers[i].Process(rechannel_output, condition, num_frames);
       } else {
           // Subsequent layers consume the previous layer's output
           Eigen::MatrixXf& prev_output = this->_layers[i - 1].GetOutputNextLayer();
           this->_layers[i].Process(prev_output, condition, num_frames);
       }
       
       // Accumulate head output from this layer
       this->_head_inputs.leftCols(num_frames).noalias() += 
           this->_layers[i].GetOutputHead().leftCols(num_frames);
   }

Step 3: Head Rechanneling
~~~~~~~~~~~~~~~~~~~~~~~~~~

The accumulated head outputs are rechanneled to the final head size:

.. code-block:: cpp
   :caption: Head rechanneling

   _head_rechannel.process_(this->_head_inputs, num_frames);

LayerArray Structure
~~~~~~~~~~~~~~~~~~~~

.. mermaid::
   :caption: LayerArray Structure

   graph TD
       Input[Layer Input] --> Rechannel[Rechannel]
       Rechannel --> Layer1[Layer 1]
       Layer1 -->|Residual| Layer2[Layer 2]
       Layer2 -->|Residual| Layer3[Layer 3]
       Layer3 -->|Residual| LayerN[Layer N]
       Layer1 -->|Skip| HeadAccum[Head Accumulator]
       Layer2 -->|Skip| HeadAccum
       Layer3 -->|Skip| HeadAccum
       LayerN -->|Skip| HeadAccum
       HeadAccum --> HeadRechannel[Head Rechannel]
       HeadRechannel --> HeadOut[Head Output]
       LayerN --> LayerOut[Layer Output]

WaveNet Processing
------------------

The complete WaveNet processing pipeline:

Step 1: Condition Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If a condition DSP is provided, the input is processed through it to generate the conditioning signal:

.. code-block:: cpp
   :caption: Condition processing

   void WaveNet::_process_condition(const int num_frames) {
       if (this->_condition_dsp != nullptr) {
           // Process input through condition DSP
           this->_condition_dsp->process(/* input */, /* output */, num_frames);
           // Copy output to condition buffer
       } else {
           // Use input directly as condition
           this->_condition_output = this->_condition_input;
       }
   }

Step 2: LayerArray Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each LayerArray processes the output of the previous array:

1. **First LayerArray**: Processes the input with zeroed head inputs
2. **Subsequent LayerArrays**: Process the previous array's output and accumulate head inputs

.. code-block:: cpp
   :caption: LayerArray processing

   // First layer array
   this->_layer_arrays[0].Process(input, condition, num_frames);
   
   // Subsequent layer arrays
   for (size_t i = 1; i < this->_layer_arrays.size(); i++) {
       Eigen::MatrixXf& prev_output = this->_layer_arrays[i-1].GetLayerOutputs();
       Eigen::MatrixXf& prev_head = this->_layer_arrays[i-1].GetHeadOutputs();
       this->_layer_arrays[i].Process(prev_output, condition, prev_head, num_frames);
   }

Step 3: Head Scaling and Output
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The final head output from the last LayerArray is scaled and written to output:

.. code-block:: cpp
   :caption: Head scaling and output

   Eigen::MatrixXf& final_head = this->_layer_arrays.back().GetHeadOutputs();
   // Apply head scale and write to output buffers
   // (implementation details in wavenet.cpp)

Complete WaveNet Flow
~~~~~~~~~~~~~~~~~~~~~~

.. mermaid::
   :caption: Complete WaveNet Processing Flow

   graph TD
       AudioIn[Audio Input] --> ConditionProc{Condition DSP?}
       ConditionProc -->|Yes| CondDSP[Condition DSP]
       ConditionProc -->|No| Condition[Condition Signal]
       CondDSP --> Condition
       AudioIn --> LayerArray1[LayerArray 1]
       Condition --> LayerArray1
       LayerArray1 --> LayerArray2[LayerArray 2]
       Condition --> LayerArray2
       LayerArray2 --> LayerArrayN[LayerArray N]
       Condition --> LayerArrayN
       LayerArray1 -->|Head| HeadAccum[Head Accumulator]
       LayerArray2 -->|Head| HeadAccum
       LayerArrayN -->|Head| HeadAccum
       HeadAccum --> HeadScale[Head Scale]
       HeadScale --> AudioOut[Audio Output]

Buffer Management and Real-Time Safety
---------------------------------------

All processing uses pre-allocated buffers to ensure real-time safety:

* **No Dynamic Allocations**: All buffers are allocated during `SetMaxBufferSize()`
* **Buffer Reuse**: The same buffers are reused across processing calls
* **Column Slicing**: Only the first `num_frames` columns are valid; use `.leftCols(num_frames)` to access
* **Eigen::Ref**: Used to avoid temporary allocations when passing block expressions

Key Design Principles
---------------------

1. **Residual Connections**: Allow gradients to flow directly through layers, enabling deeper networks
2. **Skip Connections**: Preserve information at multiple scales for the final output
3. **Dilated Convolutions**: Increase receptive field without increasing parameters
4. **Conditioning**: Allow the model to adapt based on input characteristics
5. **FiLM**: Fine-grained control over feature modulation at multiple points
6. **Gating/Blending**: Enable more expressive activations and better gradient flow

See Also
--------

* :doc:`api/wavenet` - Complete API reference for WaveNet classes
* :doc:`api/dsp` - Base DSP interface documentation
* :doc:`api/conv1d` - Convolution implementation details
