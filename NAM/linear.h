#pragma once

#include "dsp.h"

namespace nam
{

/// \brief Basic linear model
///
/// Implements a simple linear convolution, (i.e. an impulse response).
class Linear : public Buffer
{
public:
  /// \brief Constructor
  /// \param in_channels Number of input channels
  /// \param out_channels Number of output channels
  /// \param receptive_field Size of the impulse response
  /// \param _bias Whether to use bias
  /// \param weights Model weights (impulse response coefficients)
  /// \param expected_sample_rate Expected sample rate in Hz (-1.0 if unknown)
  Linear(const int in_channels, const int out_channels, const int receptive_field, const bool _bias,
         const std::vector<float>& weights, const double expected_sample_rate = -1.0);

  /// \brief Process audio frames
  /// \param input Input audio buffers
  /// \param output Output audio buffers
  /// \param num_frames Number of frames to process
  void process(NAM_SAMPLE** input, NAM_SAMPLE** output, const int num_frames) override;

protected:
  Eigen::VectorXf _weight;
  float _bias;
};

namespace linear
{

/// \brief Configuration for a Linear model
struct LinearConfig : public ModelConfig
{
  int receptive_field;
  bool bias;
  int in_channels;
  int out_channels;

  std::unique_ptr<DSP> create(std::vector<float> weights, double sampleRate) override;
};

/// \brief Parse Linear configuration from JSON
/// \param config JSON configuration object
/// \return LinearConfig
LinearConfig parse_config_json(const nlohmann::json& config);

/// \brief Config parser for ConfigParserRegistry
/// \param config JSON configuration object
/// \param sampleRate Expected sample rate in Hz
/// \return unique_ptr<ModelConfig> wrapping a LinearConfig
std::unique_ptr<ModelConfig> create_config(const nlohmann::json& config, double sampleRate);
} // namespace linear

} // namespace nam
