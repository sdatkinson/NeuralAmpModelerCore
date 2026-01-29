#pragma once

#include <Eigen/Dense>
#include <cassert>
#include <memory>
#include <vector>

namespace nam
{

/// \brief Type-erased interface for Conv1x1 implementations
///
/// This interface allows runtime polymorphism while enabling compile-time
/// optimized implementations via templates. All Conv1x1 variants (fixed-size
/// and dynamic) implement this interface.
class IConv1x1
{
public:
  virtual ~IConv1x1() = default;

  /// \brief Get the entire internal output buffer
  ///
  /// This is intended for internal wiring between layers/arrays; callers should treat
  /// the buffer as pre-allocated storage and only consider the first num_frames columns
  /// valid for a given processing call. Slice with .leftCols(num_frames) as needed.
  /// \return Reference to the output buffer
  virtual Eigen::MatrixXf& GetOutput() = 0;

  /// \brief Get the entire internal output buffer (const version)
  /// \return Const reference to the output buffer
  virtual const Eigen::MatrixXf& GetOutput() const = 0;

  /// \brief Resize the output buffer to handle maxBufferSize frames
  /// \param maxBufferSize Maximum number of frames to process in a single call
  virtual void SetMaxBufferSize(int maxBufferSize) = 0;

  /// \brief Set the parameters (weights) of this module
  /// \param weights Iterator to the weights vector. Will be advanced as weights are consumed.
  virtual void set_weights_(std::vector<float>::iterator& weights) = 0;

  /// \brief Process input and store output to pre-allocated buffer
  ///
  /// Uses Eigen::Ref to accept matrices and block expressions without creating
  /// temporaries (real-time safe). Access output via GetOutput().
  /// \param input Input matrix (channels x num_frames)
  /// \param num_frames Number of frames to process
  virtual void process_(const Eigen::Ref<const Eigen::MatrixXf>& input, int num_frames) = 0;

  /// \brief Process input and return output matrix
  /// \param input Input matrix (channels x num_frames)
  /// \param num_frames Number of frames to process
  /// \return Output matrix (channels x num_frames)
  virtual Eigen::MatrixXf process(const Eigen::MatrixXf& input, int num_frames) const = 0;

  /// \brief Get the number of output channels
  /// \return Number of output channels
  virtual long get_out_channels() const = 0;

  /// \brief Get the number of input channels
  /// \return Number of input channels
  virtual long get_in_channels() const = 0;
};

/// \brief Fully compile-time optimized Conv1x1 with fixed dimensions AND buffer size
///
/// This implementation uses fixed-size Eigen matrices for weights, input, and output,
/// enabling the compiler to fully unroll and vectorize all operations.
///
/// Template parameters:
/// \tparam OutChannels Number of output channels
/// \tparam InChannels Number of input channels
/// \tparam MaxFrames Maximum buffer size (e.g., 32, 64, 128, 256, 512)
/// \tparam Groups Number of groups for grouped convolution
/// \tparam HasBias Whether to use bias
template <int OutChannels, int InChannels, int MaxFrames, int Groups = 1, bool HasBias = true>
class Conv1x1FullyFixed : public IConv1x1
{
public:
  static_assert(OutChannels > 0, "OutChannels must be positive");
  static_assert(InChannels > 0, "InChannels must be positive");
  static_assert(MaxFrames > 0, "MaxFrames must be positive");
  static_assert(Groups > 0, "Groups must be positive");
  static_assert(OutChannels % Groups == 0, "OutChannels must be divisible by Groups");
  static_assert(InChannels % Groups == 0, "InChannels must be divisible by Groups");

  // Fully fixed-size types for maximum optimization
  using WeightMatrix = Eigen::Matrix<float, OutChannels, InChannels>;
  using BiasVector = Eigen::Matrix<float, OutChannels, 1>;
  using InputBuffer = Eigen::Matrix<float, InChannels, MaxFrames>;
  using OutputBuffer = Eigen::Matrix<float, OutChannels, MaxFrames>;

  Conv1x1FullyFixed()
  {
    _weight.setZero();
    if constexpr (HasBias)
    {
      _bias.setZero();
    }
    _output_dynamic.resize(OutChannels, MaxFrames);
  }

  Eigen::MatrixXf& GetOutput() override { return _output_dynamic; }

  const Eigen::MatrixXf& GetOutput() const override { return _output_dynamic; }

  void SetMaxBufferSize(int maxBufferSize) override
  {
    // For fully fixed implementation, we require the buffer size to match
    assert(maxBufferSize <= MaxFrames && "Buffer size exceeds MaxFrames template parameter");
    // Output is already sized correctly
  }

  void set_weights_(std::vector<float>::iterator& weights) override
  {
    if constexpr (Groups == 1)
    {
      // Non-grouped: simple row-major weight loading
      for (int i = 0; i < OutChannels; i++)
      {
        for (int j = 0; j < InChannels; j++)
        {
          _weight(i, j) = *(weights++);
        }
      }
    }
    else
    {
      // Grouped convolution: block-diagonal weight matrix
      constexpr int out_per_group = OutChannels / Groups;
      constexpr int in_per_group = InChannels / Groups;

      for (int g = 0; g < Groups; g++)
      {
        for (int i = 0; i < out_per_group; i++)
        {
          for (int j = 0; j < in_per_group; j++)
          {
            _weight(g * out_per_group + i, g * in_per_group + j) = *(weights++);
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

  void process_(const Eigen::Ref<const Eigen::MatrixXf>& input, int num_frames) override
  {
    assert(num_frames <= MaxFrames);

    // Copy input to fixed-size buffer for fully optimized matmul
    _input_fixed.template leftCols<MaxFrames>().leftCols(num_frames) = input.leftCols(num_frames);

    if constexpr (Groups == 1)
    {
      // Single group: fully fixed matrix multiply
      _output_fixed.noalias() = _weight * _input_fixed;
    }
    else
    {
      // Grouped convolution with compile-time unrolled loop
      constexpr int out_per_group = OutChannels / Groups;
      constexpr int in_per_group = InChannels / Groups;
      process_groups_impl<out_per_group, in_per_group>(std::make_integer_sequence<int, Groups>{});
    }

    // Add bias if present
    if constexpr (HasBias)
    {
      _output_fixed.colwise() += _bias;
    }

    // Copy back to dynamic output for interface compatibility
    _output_dynamic.leftCols(num_frames) = _output_fixed.leftCols(num_frames);
  }

  /// \brief Optimized process for when caller knows the exact frame count at compile time
  template <int NumFrames>
  void process_fixed(const Eigen::Matrix<float, InChannels, NumFrames>& input)
  {
    static_assert(NumFrames <= MaxFrames, "NumFrames exceeds MaxFrames");

    if constexpr (Groups == 1)
    {
      _output_fixed.template leftCols<NumFrames>().noalias() = _weight * input;
    }
    else
    {
      // Copy to internal buffer first
      _input_fixed.template leftCols<NumFrames>() = input;
      constexpr int out_per_group = OutChannels / Groups;
      constexpr int in_per_group = InChannels / Groups;
      process_groups_impl<out_per_group, in_per_group>(std::make_integer_sequence<int, Groups>{});
    }

    if constexpr (HasBias)
    {
      _output_fixed.template leftCols<NumFrames>().colwise() += _bias;
    }
  }

  /// \brief Get output as fixed-size matrix reference
  template <int NumFrames>
  auto GetOutputFixed() -> Eigen::Block<OutputBuffer, OutChannels, NumFrames>
  {
    return _output_fixed.template leftCols<NumFrames>();
  }

  Eigen::MatrixXf process(const Eigen::MatrixXf& input, int num_frames) const override
  {
    Eigen::MatrixXf result(OutChannels, num_frames);

    if constexpr (Groups == 1)
    {
      result.noalias() = _weight * input.leftCols(num_frames);
    }
    else
    {
      constexpr int out_per_group = OutChannels / Groups;
      constexpr int in_per_group = InChannels / Groups;
      for (int g = 0; g < Groups; g++)
      {
        auto input_group = input.middleRows(g * in_per_group, in_per_group).leftCols(num_frames);
        auto weight_group = _weight.template block<out_per_group, in_per_group>(g * out_per_group, g * in_per_group);
        result.middleRows(g * out_per_group, out_per_group).noalias() = weight_group * input_group;
      }
    }

    if constexpr (HasBias)
    {
      result.colwise() += _bias;
    }

    return result;
  }

  long get_out_channels() const override { return OutChannels; }
  long get_in_channels() const override { return InChannels; }

  /// \brief Get the maximum buffer size this implementation supports
  static constexpr int GetMaxFrames() { return MaxFrames; }

private:
  WeightMatrix _weight;
  BiasVector _bias;
  InputBuffer _input_fixed;
  OutputBuffer _output_fixed;
  Eigen::MatrixXf _output_dynamic; // For interface compatibility

  // Helper to unroll group processing at compile time
  template <int OutPerGroup, int InPerGroup, int... Gs>
  void process_groups_impl(std::integer_sequence<int, Gs...>)
  {
    (process_single_group<Gs, OutPerGroup, InPerGroup>(), ...);
  }

  template <int G, int OutPerGroup, int InPerGroup>
  void process_single_group()
  {
    auto input_group = _input_fixed.template middleRows<InPerGroup>(G * InPerGroup);
    auto weight_group = _weight.template block<OutPerGroup, InPerGroup>(G * OutPerGroup, G * InPerGroup);
    _output_fixed.template middleRows<OutPerGroup>(G * OutPerGroup).noalias() = weight_group * input_group;
  }
};

} // namespace nam
