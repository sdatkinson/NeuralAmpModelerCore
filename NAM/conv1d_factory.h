#pragma once

#include <memory>
#include "conv1d_fixed.h"

namespace nam
{

/// \brief Factory for creating Conv1D implementations
///
/// Returns a dynamic Conv1D implementation wrapped in the IConv1D interface.
/// For fully optimized implementations with compile-time known buffer sizes,
/// use Conv1DFullyFixed directly.
class Conv1DFactory
{
public:
  /// \brief Create a Conv1D implementation
  ///
  /// Returns a dynamic implementation. For maximum performance with known
  /// buffer sizes, use Conv1DFullyFixed template directly.
  ///
  /// \param in_channels Number of input channels
  /// \param out_channels Number of output channels
  /// \param kernel_size Size of the convolution kernel
  /// \param dilation Dilation factor for the convolution
  /// \param bias Whether to use bias
  /// \param groups Number of groups for grouped convolution (default: 1)
  /// \return Unique pointer to an IConv1D implementation
  static std::unique_ptr<IConv1D> create(int in_channels, int out_channels, int kernel_size, int dilation, bool bias,
                                         int groups = 1);
};

} // namespace nam
