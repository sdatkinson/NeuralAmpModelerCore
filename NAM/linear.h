#pragma once

#include "dsp.h"

namespace nam
{

struct LinearFFTState;

struct LinearFFTPlan
{
  int direct_taps;
  int max_partition_size;
};

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
  /// \param expected_sample_rate Training sample rate in Hz (-1.0 if unknown)
  /// \param implementation Convolution implementation to use
  Linear(const int in_channels, const int out_channels, const int receptive_field, const bool _bias,
         const std::vector<float>& weights, const double expected_sample_rate = -1.0,
         const LinearImplementation implementation = LinearImplementation::Auto);

  ~Linear() override;

  /// \brief Whether the training sample rate is known, finite, and positive
  bool SupportsArbitrarySampleRate() override;

  /// \brief Adapt the original impulse response to the processing sample rate and clear history
  ///
  /// Uses cubic interpolation with sample-rate-dependent gain compensation.
  /// The bias and training sample rate are unchanged. If the training rate is
  /// unknown (-1.0), the original coefficients are used without conversion.
  /// This may allocate and must be called outside real-time audio processing.
  /// \throws std::invalid_argument If the processing rate is not finite and positive
  /// \throws std::length_error If the resampled response or buffer size is too large
  void Reset(const double sampleRate, const int maxBufferSize) override;

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
  // Keep the trained coefficients so repeated rate changes never compound interpolation error.
  std::vector<float> _original_impulse_response;
  std::vector<float> _impulse_response;
  LinearImplementation _requested_implementation;
  LinearImplementation _active_implementation;
  std::unique_ptr<LinearFFTState> _fft_state;

  void _configure_implementation();
  void _configure_fft_state();
  void _process_direct(NAM_SAMPLE** input, NAM_SAMPLE** output, const int num_frames);
  void _process_fft(NAM_SAMPLE** input, NAM_SAMPLE** output, const int num_frames);
  void _advance_fft_job(const int tier, const int channel);
  void _advance_fft_jobs(const int channel);
  void _start_fft_block(const int tier, const int channel, const long long block_start);
  void _finish_fft_block(const int tier, const int channel);
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

/// \brief Select the tuned convolution plan for an impulse-response length.
LinearFFTPlan select_fft_plan(int receptive_field);

/// \brief Select the default implementation for an impulse-response length.
LinearImplementation select_implementation(int receptive_field);

/// \brief Parse Linear configuration from JSON
/// \param config JSON configuration object
/// \return LinearConfig
LinearConfig parse_config_json(const nlohmann::json& config);

/// \brief Config parser for ConfigParserRegistry
/// \param config JSON configuration object
/// \param sampleRate Training sample rate in Hz
/// \return unique_ptr<ModelConfig> wrapping a LinearConfig
std::unique_ptr<ModelConfig> create_config(const nlohmann::json& config, double sampleRate);
} // namespace linear

} // namespace nam
