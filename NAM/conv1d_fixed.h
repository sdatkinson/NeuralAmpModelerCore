#pragma once

#include <Eigen/Dense>
#include <array>
#include <cassert>
#include <memory>
#include <vector>

namespace nam
{

/// \brief Type-erased interface for Conv1D implementations
///
/// This interface allows runtime polymorphism while enabling compile-time
/// optimized implementations via templates. All Conv1D variants (fixed-size
/// and dynamic) implement this interface.
class IConv1D
{
public:
  virtual ~IConv1D() = default;

  /// \brief Get the entire internal output buffer
  /// \return Reference to the output buffer
  virtual Eigen::MatrixXf& GetOutput() = 0;

  /// \brief Get the entire internal output buffer (const version)
  /// \return Const reference to the output buffer
  virtual const Eigen::MatrixXf& GetOutput() const = 0;

  /// \brief Resize the output buffer and reset ring buffer
  /// \param maxBufferSize Maximum number of frames to process in a single call
  virtual void SetMaxBufferSize(int maxBufferSize) = 0;

  /// \brief Set the parameters (weights) of this module
  /// \param weights Iterator to the weights vector. Will be advanced as weights are consumed.
  virtual void set_weights_(std::vector<float>::iterator& weights) = 0;

  /// \brief Process input and store output to pre-allocated buffer
  /// \param input Input matrix (channels x num_frames)
  /// \param num_frames Number of frames to process
  virtual void Process(const Eigen::MatrixXf& input, int num_frames) = 0;

  /// \brief Get the number of output channels
  /// \return Number of output channels
  virtual long get_out_channels() const = 0;

  /// \brief Get the number of input channels
  /// \return Number of input channels
  virtual long get_in_channels() const = 0;

  /// \brief Get the kernel size
  /// \return Kernel size
  virtual long get_kernel_size() const = 0;

  /// \brief Get the dilation factor
  /// \return Dilation factor
  virtual int get_dilation() const = 0;

  /// \brief Check if bias is used
  /// \return true if bias is present, false otherwise
  virtual bool has_bias() const = 0;
};

/// \brief Fully compile-time optimized Conv1D with fixed dimensions AND buffer size
///
/// This implementation uses fixed-size Eigen matrices for weights, input, and output,
/// enabling the compiler to fully unroll and vectorize all operations.
///
/// Template parameters:
/// \tparam OutChannels Number of output channels
/// \tparam InChannels Number of input channels
/// \tparam KernelSize Size of the convolution kernel
/// \tparam MaxFrames Maximum buffer size (e.g., 32, 64, 128, 256, 512)
/// \tparam Groups Number of groups for grouped convolution
/// \tparam HasBias Whether to use bias
template <int OutChannels, int InChannels, int KernelSize, int MaxFrames, int Groups = 1, bool HasBias = true>
class Conv1DFullyFixed : public IConv1D
{
public:
  static_assert(OutChannels > 0, "OutChannels must be positive");
  static_assert(InChannels > 0, "InChannels must be positive");
  static_assert(KernelSize > 0, "KernelSize must be positive");
  static_assert(MaxFrames > 0, "MaxFrames must be positive");
  static_assert(Groups > 0, "Groups must be positive");
  static_assert(OutChannels % Groups == 0, "OutChannels must be divisible by Groups");
  static_assert(InChannels % Groups == 0, "InChannels must be divisible by Groups");

  // Derived constants
  static constexpr int OutPerGroup = OutChannels / Groups;
  static constexpr int InPerGroup = InChannels / Groups;

  // Fully fixed-size types for maximum optimization
  using WeightMatrix = Eigen::Matrix<float, OutChannels, InChannels>;
  using BiasVector = Eigen::Matrix<float, OutChannels, 1>;
  using InputBuffer = Eigen::Matrix<float, InChannels, MaxFrames>;
  using OutputBuffer = Eigen::Matrix<float, OutChannels, MaxFrames>;


  Conv1DFullyFixed(int dilation = 1)
  : _dilation(dilation)
  {
    // Initialize weights to zero (critical for block-diagonal structure)
    for (int k = 0; k < KernelSize; k++)
    {
      _weight[k].setZero();
    }

    if constexpr (HasBias)
    {
      _bias.setZero();
    }

    _output_fixed.setZero();
    _output_dynamic.resize(OutChannels, MaxFrames);
    _output_dynamic.setZero();

    // Initialize contiguous buffer
    _input_contiguous.setZero();
  }

  Eigen::MatrixXf& GetOutput() override { return _output_dynamic; }

  const Eigen::MatrixXf& GetOutput() const override { return _output_dynamic; }

  void SetMaxBufferSize(int maxBufferSize) override
  {
    assert(maxBufferSize <= MaxFrames && "Buffer size exceeds MaxFrames template parameter");
    // Reset contiguous buffer (zeros out history)
    _input_contiguous.setZero();
  }

  void set_weights_(std::vector<float>::iterator& weights) override
  {
    // Weight layout: for each kernel position k, weights are [group0, group1, ..., groupN-1]
    // Crazy ordering because that's how it gets flattened in PyTorch
    for (int g = 0; g < Groups; g++)
    {
      for (int i = 0; i < OutPerGroup; i++)
      {
        for (int j = 0; j < InPerGroup; j++)
        {
          for (int k = 0; k < KernelSize; k++)
          {
            _weight[k](g * OutPerGroup + i, g * InPerGroup + j) = *(weights++);
          }
        }
      }
    }

    if constexpr (HasBias)
    {
      for (int i = 0; i < OutChannels; i++)
      {
        _bias(i) = *(weights++);
      }
    }
  }

  void Process(const Eigen::MatrixXf& input, int num_frames) override
  {
    assert(num_frames <= MaxFrames);

    // Calculate receptive field for this dilation
    const int receptive_field = (KernelSize - 1) * _dilation;

    // Buffer layout: [history (receptive_field cols) | new_input (num_frames cols)]
    // History is always stored at leftCols(receptive_field) between calls

    // Copy new input after history region
    _input_contiguous.middleCols(receptive_field, num_frames) = input.leftCols(num_frames);

    // Zero output before accumulation
    _output_fixed.leftCols(num_frames).setZero();

    // Process kernel positions using block operations
    if constexpr (Groups == 1)
    {
      // Non-grouped: use efficient block operations
      process_kernel_block_impl(std::make_integer_sequence<int, KernelSize>{}, num_frames, receptive_field);
    }
    else
    {
      // Grouped: process per-group (still uses block operations per group)
      process_kernel_grouped_impl(std::make_integer_sequence<int, KernelSize>{}, num_frames, receptive_field);
    }

    // Add bias if present
    if constexpr (HasBias)
    {
      _output_fixed.leftCols(num_frames).colwise() += _bias;
    }

    // Copy to dynamic output for interface compatibility
    _output_dynamic.leftCols(num_frames) = _output_fixed.leftCols(num_frames);

    // Save history for next call: copy the last receptive_field frames to the beginning
    // This prepares the buffer for the next Process() call
    if (receptive_field > 0)
    {
      if (num_frames >= receptive_field)
      {
        // Take history from end of current input
        _input_contiguous.leftCols(receptive_field) = input.middleCols(num_frames - receptive_field, receptive_field);
      }
      else
      {
        // Not enough new frames - combine old history with new input
        const int old_history_needed = receptive_field - num_frames;
        // Shift old history left
        _input_contiguous.leftCols(old_history_needed) =
          _input_contiguous.middleCols(receptive_field - old_history_needed, old_history_needed);
        // Append new input as recent history
        _input_contiguous.middleCols(old_history_needed, num_frames) = input.leftCols(num_frames);
      }
    }
  }

  long get_out_channels() const override { return OutChannels; }

  long get_in_channels() const override { return InChannels; }

  long get_kernel_size() const override { return KernelSize; }

  int get_dilation() const override { return _dilation; }

  bool has_bias() const override { return HasBias; }

  /// \brief Get the maximum buffer size this implementation supports
  static constexpr int GetMaxFrames() { return MaxFrames; }

private:
  std::array<WeightMatrix, KernelSize> _weight;
  BiasVector _bias;
  OutputBuffer _output_fixed;
  Eigen::MatrixXf _output_dynamic; // For interface compatibility

  // Contiguous buffer for efficient block operations: [history | current_input]
  // Size: InChannels x (receptive_field + MaxFrames)
  static constexpr int MaxReceptiveField = (KernelSize - 1) * 16; // Support up to dilation=16
  static constexpr int ContiguousBufferSize = MaxReceptiveField + MaxFrames;
  Eigen::Matrix<float, InChannels, ContiguousBufferSize> _input_contiguous;
  int _dilation;

  // Helper to unroll kernel processing using block operations (non-grouped)
  template <int... Ks>
  void process_kernel_block_impl(std::integer_sequence<int, Ks...>, int num_frames, int receptive_field)
  {
    (process_single_kernel_block<Ks>(num_frames, receptive_field), ...);
  }

  template <int K>
  void process_single_kernel_block(int num_frames, int receptive_field)
  {
    // Calculate offset for this kernel position
    // For causal conv: output[t] = sum_k(weight[k] * input[t - dilation*(K-1-k)])
    const int offset = _dilation * (KernelSize - 1 - K);

    // Source position in contiguous buffer
    const int src_start = receptive_field - offset;

    // Use block operation for efficient matmul
    _output_fixed.leftCols(num_frames).noalias() +=
      _weight[K] * _input_contiguous.middleCols(src_start, num_frames);
  }

  // Helper to unroll kernel processing for grouped convolution
  template <int... Ks>
  void process_kernel_grouped_impl(std::integer_sequence<int, Ks...>, int num_frames, int receptive_field)
  {
    (process_single_kernel_grouped<Ks>(num_frames, receptive_field), ...);
  }

  template <int K>
  void process_single_kernel_grouped(int num_frames, int receptive_field)
  {
    const int offset = _dilation * (KernelSize - 1 - K);
    const int src_start = receptive_field - offset;

    // Process each group
    for (int g = 0; g < Groups; g++)
    {
      auto input_group = _input_contiguous.template middleRows<InPerGroup>(g * InPerGroup).middleCols(src_start, num_frames);
      auto weight_group = _weight[K].template block<OutPerGroup, InPerGroup>(g * OutPerGroup, g * InPerGroup);
      _output_fixed.template middleRows<OutPerGroup>(g * OutPerGroup).leftCols(num_frames).noalias() +=
        weight_group * input_group;
    }
  }
};

} // namespace nam
