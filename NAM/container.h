#pragma once

#include <memory>
#include <stdexcept>
#include <vector>

#include "dsp.h"
#include "model_config.h"
#include "slimmable.h"

namespace nam
{
namespace container
{

struct Submodel
{
  double max_value;
  std::unique_ptr<DSP> model;
};

/// \brief A container model that holds multiple submodels at different sizes
///
/// SetSlimmableSize selects the active submodel based on the max_value thresholds.
/// Each submodel covers values up to (but not including) its max_value.
/// The last submodel is the fallback for values at or above the last threshold.
class ContainerModel : public DSP, public SlimmableModel
{
public:
  /// \brief Constructor
  /// \param submodels Vector of submodels sorted by max_value ascending
  /// \param expected_sample_rate Expected sample rate in Hz
  ContainerModel(std::vector<Submodel> submodels, const double expected_sample_rate);

  void process(NAM_SAMPLE** input, NAM_SAMPLE** output, const int num_frames) override;
  void prewarm() override;
  void Reset(const double sampleRate, const int maxBufferSize) override;
  void SetSlimmableSize(const double val) override;

protected:
  int PrewarmSamples() override { return 0; }

private:
  std::vector<Submodel> _submodels;
  size_t _active_index = 0;

  DSP& _active_model() { return *_submodels[_active_index].model; }
};

// Config / registration

struct ContainerConfig : public ModelConfig
{
  nlohmann::json raw_config;
  double sample_rate;

  std::unique_ptr<DSP> create(std::vector<float> weights, double sampleRate) override;
};

std::unique_ptr<ModelConfig> create_config(const nlohmann::json& config, double sampleRate);

} // namespace container
} // namespace nam
