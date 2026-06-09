#pragma once

#include "dsp.h"

namespace nam
{

struct LinearFFTState;

/// \brief Selects the convolution engine used by Linear models.
enum class LinearImplementation
{
  Auto, ///< Choose direct or FFT convolution from the impulse-response length.
  Direct, ///< Legacy per-sample direct convolution.
  FFT ///< Zero-latency partitioned FFT convolution.
};

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
  /// \param implementation Convolution implementation to use
  Linear(const int in_channels, const int out_channels, const int receptive_field, const bool _bias,
         const std::vector<float>& weights, const double expected_sample_rate = -1.0,
         const LinearImplementation implementation = LinearImplementation::Auto);

  ~Linear() override;

  /// \brief Process audio frames
  /// \param input Input audio buffers
  /// \param output Output audio buffers
  /// \param num_frames Number of frames to process
  void process(NAM_SAMPLE** input, NAM_SAMPLE** output, const int num_frames) override;

  LinearImplementation GetRequestedImplementation() const { return _requested_implementation; }
  LinearImplementation GetActiveImplementation() const { return _active_implementation; }

protected:
  void SetMaxBufferSize(const int maxBufferSize) override;

protected:
  Eigen::VectorXf _weight;
  Eigen::VectorXf _fft_direct_weight;
  float _bias;

private:
  std::vector<float> _impulse_response;
  LinearImplementation _requested_implementation;
  LinearImplementation _active_implementation;
  std::unique_ptr<LinearFFTState> _fft_state;

  void _configure_implementation();
  void _configure_fft_state();
  void _process_direct(NAM_SAMPLE** input, NAM_SAMPLE** output, const int num_frames);
  void _process_fft(NAM_SAMPLE** input, NAM_SAMPLE** output, const int num_frames);
  void _run_fft_block(const int channel);
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
  LinearImplementation implementation = LinearImplementation::Auto;

  std::unique_ptr<DSP> create(std::vector<float> weights, double sampleRate) override;
};

/// \brief Parse a Linear implementation string.
LinearImplementation parse_implementation(const std::string& implementation);

/// \brief String name for a Linear implementation.
std::string implementation_to_string(const LinearImplementation implementation);

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
