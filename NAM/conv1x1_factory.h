#pragma once

#include <memory>
#include "conv1x1_fixed.h"

namespace nam
{

/// \brief Factory for creating Conv1x1 implementations
///
/// Returns a dynamic Conv1x1 implementation wrapped in the IConv1x1 interface.
/// For fully optimized implementations with compile-time known buffer sizes,
/// use Conv1x1FullyFixed directly.
class Conv1x1Factory
{
public:
  /// \brief Create a Conv1x1 implementation
  ///
  /// Returns a dynamic implementation. For maximum performance with known
  /// buffer sizes, use Conv1x1FullyFixed template directly.
  ///
  /// \param in_channels Number of input channels
  /// \param out_channels Number of output channels
  /// \param bias Whether to use bias
  /// \param groups Number of groups for grouped convolution (default: 1)
  /// \return Unique pointer to an IConv1x1 implementation
  static std::unique_ptr<IConv1x1> create(int in_channels, int out_channels, bool bias, int groups = 1);
};

} // namespace nam
