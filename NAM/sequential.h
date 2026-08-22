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
/// buffers are allocated when the maximum buffer size is set and reused by
/// process().
class SequentialModel : public DSP
{
public:
  /// \param models Child DSP models in processing order
  /// \param expected_sample_rate Expected sample rate in Hz, or -1.0 to derive from children
  SequentialModel(std::vector<std::unique_ptr<DSP>> models, double expected_sample_rate);

  void process(NAM_SAMPLE** input, NAM_SAMPLE** output, int num_frames) override;
  void prewarm() override;
  void Reset(double sampleRate, int maxBufferSize) override;
  void SetPrewarmOnReset(bool prewarmOnReset) override;
  int GetPrewarmSamples() override;

protected:
  void SetMaxBufferSize(int maxBufferSize) override;

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
