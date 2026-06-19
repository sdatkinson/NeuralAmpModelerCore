#include <algorithm>
#include <limits>
#include <sstream>
#include <stdexcept>

#include "get_dsp.h"
#include "sequential.h"

namespace
{

void validate_models_present(const std::vector<std::unique_ptr<nam::DSP>>& models)
{
  if (models.empty())
    throw std::runtime_error("SequentialModel: no models provided");
  for (const auto& model : models)
  {
    if (model == nullptr)
      throw std::runtime_error("SequentialModel: null model provided");
  }
}

int get_input_channels(const std::vector<std::unique_ptr<nam::DSP>>& models)
{
  validate_models_present(models);
  return models.front()->NumInputChannels();
}

int get_output_channels(const std::vector<std::unique_ptr<nam::DSP>>& models)
{
  validate_models_present(models);
  return models.back()->NumOutputChannels();
}

double resolve_expected_sample_rate(const std::vector<std::unique_ptr<nam::DSP>>& models, double expected_sample_rate)
{
  validate_models_present(models);
  double resolved = expected_sample_rate;

  for (const auto& model : models)
  {
    const double child_sample_rate = model->GetExpectedSampleRate();
    if (child_sample_rate == NAM_UNKNOWN_EXPECTED_SAMPLE_RATE)
      continue;
    if (resolved == NAM_UNKNOWN_EXPECTED_SAMPLE_RATE)
    {
      resolved = child_sample_rate;
      continue;
    }
    if (child_sample_rate != resolved)
    {
      std::stringstream ss;
      ss << "SequentialModel: submodel sample rate mismatch (expected " << resolved << ", got " << child_sample_rate
         << ")";
      throw std::runtime_error(ss.str());
    }
  }

  return resolved;
}

void validate_channel_links(const std::vector<std::unique_ptr<nam::DSP>>& models)
{
  validate_models_present(models);
  for (size_t i = 1; i < models.size(); ++i)
  {
    const int previous_output_channels = models[i - 1]->NumOutputChannels();
    const int next_input_channels = models[i]->NumInputChannels();
    if (previous_output_channels != next_input_channels)
    {
      std::stringstream ss;
      ss << "SequentialModel: channel mismatch between submodels " << i - 1 << " and " << i << " ("
         << previous_output_channels << " output channels versus " << next_input_channels << " input channels)";
      throw std::runtime_error(ss.str());
    }
  }
}

const nlohmann::json& get_model_list(const nlohmann::json& config)
{
  if (config.contains("models"))
    return config.at("models");
  if (config.contains("layers"))
    return config.at("layers");
  throw std::runtime_error("Sequential: config must contain a 'models' or 'layers' array");
}

nlohmann::json normalize_model_entry(const nlohmann::json& entry)
{
  if (entry.contains("model"))
    return entry.at("model");
  return entry;
}

std::vector<std::unique_ptr<nam::DSP>> build_models(const nlohmann::json& config)
{
  const auto& models_json = get_model_list(config);
  if (!models_json.is_array() || models_json.empty())
    throw std::runtime_error("Sequential: 'models'/'layers' must be a non-empty array");

  std::vector<std::unique_ptr<nam::DSP>> models;
  models.reserve(models_json.size());

  for (const auto& entry : models_json)
  {
    auto model_json = normalize_model_entry(entry);
    if (!model_json.contains("architecture") || !model_json.contains("config"))
    {
      throw std::runtime_error(
        "Sequential: each child must be a complete NAM model object with architecture and config");
    }
    models.push_back(nam::get_dsp(model_json));
  }

  return models;
}

void restore_child_prewarm_states(const std::vector<std::unique_ptr<nam::DSP>>& models,
                                  const std::vector<bool>& prewarm_states)
{
  for (size_t i = 0; i < models.size(); ++i)
    models[i]->SetPrewarmOnReset(prewarm_states[i]);
}

int get_weights_version(const nlohmann::json& config)
{
  const int weights_version = config.value("weights_version", 1);
  if (weights_version != 1 && weights_version != 2)
  {
    throw std::runtime_error("Sequential: unsupported weights_version " + std::to_string(weights_version));
  }
  return weights_version;
}

} // namespace

namespace nam
{
namespace sequential
{

SequentialModel::SequentialModel(std::vector<std::unique_ptr<DSP>> models, const double expected_sample_rate)
: DSP(
    get_input_channels(models), get_output_channels(models), resolve_expected_sample_rate(models, expected_sample_rate))
, _models(std::move(models))
{
  validate_channel_links(_models);
}

void SequentialModel::process(NAM_SAMPLE** input, NAM_SAMPLE** output, const int num_frames)
{
  if (num_frames < 0)
    throw std::runtime_error("SequentialModel: num_frames cannot be negative");
  if (num_frames > GetMaxBufferSize())
    SetMaxBufferSize(num_frames);

  NAM_SAMPLE** stage_input = input;
  for (size_t i = 0; i < _models.size(); ++i)
  {
    NAM_SAMPLE** stage_output = nullptr;
    if (i + 1 == _models.size())
      stage_output = output;
    else
      stage_output = _stage_buffer_ptrs[i].data();

    _models[i]->process(stage_input, stage_output, num_frames);
    stage_input = stage_output;
  }
}

void SequentialModel::prewarm()
{
  DSP::prewarm();
}

void SequentialModel::Reset(const double sampleRate, const int maxBufferSize)
{
  mExternalSampleRate = sampleRate;
  mHaveExternalSampleRate = true;
  SetMaxBufferSize(maxBufferSize);

  std::vector<bool> child_prewarm_states;
  child_prewarm_states.reserve(_models.size());
  for (auto& model : _models)
  {
    child_prewarm_states.push_back(model->GetPrewarmOnReset());
    model->SetPrewarmOnReset(false);
  }

  try
  {
    for (auto& model : _models)
      model->Reset(sampleRate, maxBufferSize);
  }
  catch (...)
  {
    restore_child_prewarm_states(_models, child_prewarm_states);
    throw;
  }
  restore_child_prewarm_states(_models, child_prewarm_states);

  if (GetPrewarmOnReset())
    prewarm();
}

void SequentialModel::SetPrewarmOnReset(const bool prewarmOnReset)
{
  DSP::SetPrewarmOnReset(prewarmOnReset);
  for (auto& model : _models)
    model->SetPrewarmOnReset(prewarmOnReset);
}

int SequentialModel::GetPrewarmSamples()
{
  int samples = 0;
  for (auto& model : _models)
  {
    const int child_samples = model->GetPrewarmSamples();
    if (child_samples > std::numeric_limits<int>::max() - samples)
      return std::numeric_limits<int>::max();
    samples += child_samples;
  }
  return samples;
}

void SequentialModel::SetMaxBufferSize(const int maxBufferSize)
{
  DSP::SetMaxBufferSize(maxBufferSize);

  const size_t intermediate_stages = _models.size() > 0 ? _models.size() - 1 : 0;
  _stage_buffers.resize(intermediate_stages);
  _stage_buffer_ptrs.resize(intermediate_stages);

  const int buffer_size = std::max(maxBufferSize, 0);
  for (size_t stage = 0; stage < intermediate_stages; ++stage)
  {
    const int channels = _models[stage]->NumOutputChannels();
    _stage_buffers[stage].resize(channels);
    _stage_buffer_ptrs[stage].resize(channels);
    for (int ch = 0; ch < channels; ++ch)
    {
      _stage_buffers[stage][ch].resize(buffer_size);
      _stage_buffer_ptrs[stage][ch] = _stage_buffers[stage][ch].data();
    }
  }
}

std::unique_ptr<DSP> SequentialConfig::create(std::vector<float> weights, double sampleRate)
{
  const int weights_version = get_weights_version(raw_config);
  if (weights_version == 1)
  {
    throw std::runtime_error(
      "Sequential: weights_version=1 uses deprecated top-level concatenated weights and is not supported yet");
  }

  if (!weights.empty())
  {
    throw std::runtime_error(
      "Sequential: top-level weights are not supported for weights_version=2; embed weights in each child model "
      "instead");
  }

  auto models = build_models(raw_config);
  return std::make_unique<SequentialModel>(std::move(models), sampleRate);
}

std::unique_ptr<ModelConfig> create_config(const nlohmann::json& config, double sampleRate)
{
  (void)sampleRate;
  auto c = std::make_unique<SequentialConfig>();
  c->raw_config = config;
  return c;
}

static ConfigParserHelper _register_sequential("sequential", create_config);
static ConfigParserHelper _register_Sequential("Sequential", create_config);

} // namespace sequential
} // namespace nam
