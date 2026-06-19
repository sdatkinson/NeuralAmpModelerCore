#pragma once

#include <memory>
#include <vector>

#include "dsp.h"
#include "model_config.h"

namespace nam
{
namespace sequential
{

/// \brief A serial composition of DSP models.
///
/// Each child model processes the output of the previous child. Intermediate
/// buffers are allocated during reset / max-buffer-size updates so process()
/// can reuse them.
class SequentialModel : public DSP
{
public:
  /// \brief Constructor
  /// \param models Child DSP models in processing order
  /// \param expected_sample_rate Expected sample rate in Hz, or -1.0 to derive from children
  SequentialModel(std::vector<std::unique_ptr<DSP>> models, const double expected_sample_rate);

  void process(NAM_SAMPLE** input, NAM_SAMPLE** output, const int num_frames) override;
  void prewarm() override;
  void Reset(const double sampleRate, const int maxBufferSize) override;
  void SetPrewarmOnReset(const bool prewarmOnReset) override;
  int GetPrewarmSamples() override;

protected:
  void SetMaxBufferSize(const int maxBufferSize) override;

private:
  std::vector<std::unique_ptr<DSP>> _models;
  std::vector<std::vector<std::vector<NAM_SAMPLE>>> _stage_buffers;
  std::vector<std::vector<NAM_SAMPLE*>> _stage_buffer_ptrs;
};

struct SequentialConfig : public ModelConfig
{
  nlohmann::json raw_config;

  std::unique_ptr<DSP> create(std::vector<float> weights, double sampleRate) override;
};

std::unique_ptr<ModelConfig> create_config(const nlohmann::json& config, double sampleRate);

} // namespace sequential
} // namespace nam
