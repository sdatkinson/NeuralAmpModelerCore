#include "linear.h"

#include <algorithm>
#include <stdexcept>

#include "registry.h"

nam::Linear::Linear(const int in_channels, const int out_channels, const int receptive_field, const bool _bias,
                    const std::vector<float>& weights, const double expected_sample_rate)
: nam::Buffer(in_channels, out_channels, receptive_field, expected_sample_rate)
{
  if ((int)weights.size() != (receptive_field + (_bias ? 1 : 0)))
    throw std::runtime_error(
      "Params vector does not match expected size based "
      "on architecture parameters");

  this->_weight.resize(this->_receptive_field);
  // Pass in in reverse order so that dot products work out of the box.
  for (int i = 0; i < this->_receptive_field; i++)
    this->_weight(i) = weights[receptive_field - 1 - i];
  this->_bias = _bias ? weights[receptive_field] : (float)0.0;
}

void nam::Linear::process(NAM_SAMPLE** input, NAM_SAMPLE** output, const int num_frames)
{
  this->nam::Buffer::_update_buffers_(input, num_frames);

  const int in_channels = NumInputChannels();
  const int out_channels = NumOutputChannels();

  // For now, Linear processes each input channel independently to corresponding output channel
  // This is a simple implementation - can be extended later for cross-channel mixing
  const int channelsToProcess = std::min(in_channels, out_channels);

  // Main computation!
  for (int ch = 0; ch < channelsToProcess; ch++)
  {
    for (int i = 0; i < num_frames; i++)
    {
      const long offset = this->_input_buffer_offset - this->_weight.size() + i + 1;
      auto input_vec = Eigen::Map<const Eigen::VectorXf>(&this->_input_buffers[ch][offset], this->_receptive_field);
      output[ch][i] = this->_bias + this->_weight.dot(input_vec);
    }
  }

  // Zero out any extra output channels
  for (int ch = channelsToProcess; ch < out_channels; ch++)
  {
    for (int i = 0; i < num_frames; i++)
      output[ch][i] = (NAM_SAMPLE)0.0;
  }

  // Prepare for next call:
  nam::Buffer::_advance_input_buffer_(num_frames);
}

nam::linear::LinearConfig nam::linear::parse_config_json(const nlohmann::json& config)
{
  LinearConfig c;
  c.receptive_field = config["receptive_field"];
  c.bias = config["bias"];
  // Default to 1 channel in/out for backward compatibility
  c.in_channels = config.value("in_channels", 1);
  c.out_channels = config.value("out_channels", 1);
  return c;
}

std::unique_ptr<nam::DSP> nam::linear::LinearConfig::create(std::vector<float> weights, double sampleRate)
{
  return std::make_unique<nam::Linear>(in_channels, out_channels, receptive_field, bias, weights, sampleRate);
}

std::unique_ptr<nam::ModelConfig> nam::linear::create_config(const nlohmann::json& config, double sampleRate)
{
  (void)sampleRate;
  auto c = std::make_unique<LinearConfig>();
  auto parsed = parse_config_json(config);
  *c = parsed;
  return c;
}

namespace
{
static nam::ConfigParserHelper _register_Linear("Linear", nam::linear::create_config);
}
