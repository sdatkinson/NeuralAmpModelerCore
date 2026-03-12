#pragma once

#include <memory>
#include <vector>

#include "dsp.h"
#include "json.hpp"
#include "model_config.h"
#include "slimmable.h"
#include "wavenet.h"

namespace nam
{
namespace slimmable_wavenet
{

/// \brief A WaveNet model that supports per-layer-array dynamic channel reduction
///
/// Stores the full WaveNet LayerArrayParams and weights. Each layer array has its
/// own allowed_channels list (from the "slimmable" config field). On SetSlimmableSize(),
/// maps the ratio to a channel count per array, extracts a weight subset, builds
/// modified LayerArrayParams, and reconstructs the WaveNet.
class SlimmableWavenet : public DSP, public SlimmableModel
{
public:
  /// \param original_params Full-size LayerArrayParams from parse_config_json
  /// \param per_array_allowed_channels Per-array sorted allowed channel counts (empty = non-slimmable)
  /// \param in_channels WaveNet input channels
  /// \param head_scale WaveNet head scale
  /// \param with_head WaveNet head flag
  /// \param condition_dsp_json JSON for rebuilding condition_dsp (nullptr if none)
  /// \param full_weights Full weight vector for the max-channel model
  /// \param expected_sample_rate Expected sample rate
  SlimmableWavenet(std::vector<wavenet::LayerArrayParams> original_params,
                   std::vector<std::vector<int>> per_array_allowed_channels, int in_channels, float head_scale,
                   bool with_head, nlohmann::json condition_dsp_json, std::vector<float> full_weights,
                   double expected_sample_rate);

  void process(NAM_SAMPLE** input, NAM_SAMPLE** output, const int num_frames) override;
  void prewarm() override;
  void Reset(const double sampleRate, const int maxBufferSize) override;
  void SetSlimmableSize(const double val) override;

protected:
  int PrewarmSamples() override { return 0; }

private:
  std::vector<wavenet::LayerArrayParams> _original_params;
  std::vector<std::vector<int>> _per_array_allowed_channels;
  int _in_channels;
  float _head_scale;
  bool _with_head;
  nlohmann::json _condition_dsp_json;
  std::vector<float> _full_weights;
  std::unique_ptr<DSP> _active_model;
  std::vector<int> _current_channels;
  int _current_buffer_size = 0;
  double _current_sample_rate = 0.0;

  void _rebuild_model(const std::vector<int>& target_channels);
};

// Config / registration

struct SlimmableWavenetConfig : public ModelConfig
{
  nlohmann::json raw_config;
  double sample_rate;

  std::unique_ptr<DSP> create(std::vector<float> weights, double sampleRate) override;
};

std::unique_ptr<ModelConfig> create_config(const nlohmann::json& config, double sampleRate);

} // namespace slimmable_wavenet
} // namespace nam
