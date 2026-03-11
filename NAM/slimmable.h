#pragma once

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
  virtual void SetSlimmableSize(const double val) = 0;
};

} // namespace nam
