#pragma once

#include <vector>

namespace nam
{

/// \brief Interface for models that support dynamic size reduction
///
/// Models implementing this interface can reduce their computational cost
/// at the expense of quality. The interpretation of the size parameter is
/// model-specific (e.g., selecting a sub-model, pruning channels, etc.).
class SlimmableModel
{
public:
  virtual ~SlimmableModel() = default;

  /// \brief Set the slimmable size of the model
  /// \param val Value between 0.0 (minimum size) and 1.0 (maximum size)
  ///
  /// Thread-safe
  /// Not real-time safe
  virtual void SetSlimmableSize(const double val) = 0;

  /// \brief Get normalized size-control values that divide the selectable slimmed models
  /// \return Sorted internal breakpoints in (0.0, 1.0); 0.0 and 1.0 are implied bounds
  // TODO: Make this abstract in the next breaking release.
  virtual std::vector<double> GetSlimmableSizeBreakpoints() const { return {}; }
};

} // namespace nam
