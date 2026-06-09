#include "linear.h"

#include <algorithm>
#include <cctype>
#include <complex>
#include <stdexcept>

#include "registry.h"

#include <unsupported/Eigen/FFT>

namespace
{
constexpr int _LINEAR_AUTO_DIRECT_MAX_TAPS = 256;
constexpr int _LINEAR_FFT_SMALL_BLOCK_SIZE = 256;
constexpr int _LINEAR_FFT_MEDIUM_BLOCK_SIZE = 512;
constexpr int _LINEAR_FFT_LARGE_BLOCK_SIZE = 1024;

int _ceil_div(const int numerator, const int denominator)
{
  return (numerator + denominator - 1) / denominator;
}

int _choose_linear_fft_block_size(const int receptive_field)
{
  if (receptive_field <= 2048)
    return _LINEAR_FFT_SMALL_BLOCK_SIZE;
  if (receptive_field <= 8192)
    return _LINEAR_FFT_MEDIUM_BLOCK_SIZE;
  return _LINEAR_FFT_LARGE_BLOCK_SIZE;
}

} // namespace

struct nam::LinearFFTState
{
  using Complex = std::complex<float>;

  struct ChannelState
  {
    std::vector<float> input_time;
    std::vector<std::vector<Complex>> input_spectra;
    std::vector<float> output_ring;
    int input_pos = 0;
    int spectrum_write_index = 0;
  };

  Eigen::FFT<float> fft;
  int block_size = 0;
  int fft_size = 0;
  int direct_taps = 0;
  int num_partitions = 0;
  int output_ring_size = 0;
  long long sample_index = 0;
  std::vector<std::vector<Complex>> kernel_spectra;
  std::vector<ChannelState> channels;
  std::vector<Complex> accumulator;
  std::vector<float> ifft_time;
};

nam::Linear::Linear(const int in_channels, const int out_channels, const int receptive_field, const bool _bias,
                    const std::vector<float>& weights, const double expected_sample_rate,
                    const LinearImplementation implementation)
: nam::Buffer(in_channels, out_channels, receptive_field, expected_sample_rate)
, _requested_implementation(implementation)
, _active_implementation(LinearImplementation::Direct)
{
  if ((int)weights.size() != (receptive_field + (_bias ? 1 : 0)))
    throw std::runtime_error(
      "Params vector does not match expected size based "
      "on architecture parameters");

  this->_impulse_response.assign(weights.begin(), weights.begin() + receptive_field);
  this->_weight.resize(this->_receptive_field);
  // Pass in in reverse order so that dot products work out of the box.
  for (int i = 0; i < this->_receptive_field; i++)
    this->_weight(i) = weights[receptive_field - 1 - i];
  this->_bias = _bias ? weights[receptive_field] : (float)0.0;

  this->_configure_implementation();
}

nam::Linear::~Linear() = default;

void nam::Linear::process(NAM_SAMPLE** input, NAM_SAMPLE** output, const int num_frames)
{
  if (this->_active_implementation == LinearImplementation::FFT)
    this->_process_fft(input, output, num_frames);
  else
    this->_process_direct(input, output, num_frames);
}

void nam::Linear::SetMaxBufferSize(const int maxBufferSize)
{
  nam::Buffer::SetMaxBufferSize(maxBufferSize);
  this->_configure_implementation();
}

void nam::Linear::_configure_implementation()
{
  if (this->_requested_implementation == LinearImplementation::Direct)
    this->_active_implementation = LinearImplementation::Direct;
  else if (this->_requested_implementation == LinearImplementation::FFT)
    this->_active_implementation = LinearImplementation::FFT;
  else
    this->_active_implementation =
      this->_receptive_field <= _LINEAR_AUTO_DIRECT_MAX_TAPS ? LinearImplementation::Direct : LinearImplementation::FFT;

  if (this->_active_implementation == LinearImplementation::FFT)
    this->_configure_fft_state();
  else
    this->_fft_state.reset();
}

void nam::Linear::_configure_fft_state()
{
  this->_fft_state = std::make_unique<LinearFFTState>();
  auto& state = *this->_fft_state;

  state.block_size = _choose_linear_fft_block_size(this->_receptive_field);
  state.fft_size = 2 * state.block_size;
  state.direct_taps = std::min(this->_receptive_field, state.block_size);
  state.num_partitions = this->_receptive_field > state.direct_taps
                           ? _ceil_div(this->_receptive_field - state.direct_taps, state.block_size)
                           : 0;
  state.output_ring_size = 4 * state.block_size;
  state.sample_index = 0;

  this->_fft_direct_weight.resize(state.direct_taps);
  for (int i = 0; i < state.direct_taps; i++)
    this->_fft_direct_weight(i) = this->_impulse_response[state.direct_taps - 1 - i];

  state.kernel_spectra.assign(state.num_partitions, std::vector<LinearFFTState::Complex>(state.fft_size));
  std::vector<float> kernel_time(state.fft_size, 0.0f);
  for (int partition = 0; partition < state.num_partitions; partition++)
  {
    std::fill(kernel_time.begin(), kernel_time.end(), 0.0f);
    const int start = state.direct_taps + partition * state.block_size;
    const int partition_size = std::min(state.block_size, this->_receptive_field - start);
    for (int i = 0; i < partition_size; i++)
      kernel_time[i] = this->_impulse_response[start + i];
    state.fft.fwd(state.kernel_spectra[partition].data(), kernel_time.data(), state.fft_size);
  }

  const int channels_to_process = std::min(NumInputChannels(), NumOutputChannels());
  state.channels.resize(channels_to_process);
  for (auto& channel : state.channels)
  {
    channel.input_time.assign(state.fft_size, 0.0f);
    channel.input_spectra.assign(
      state.num_partitions, std::vector<LinearFFTState::Complex>(state.fft_size, LinearFFTState::Complex{}));
    channel.output_ring.assign(state.output_ring_size, 0.0f);
    channel.input_pos = 0;
    channel.spectrum_write_index = 0;
  }
  state.accumulator.assign(state.fft_size, LinearFFTState::Complex{});
  state.ifft_time.assign(state.fft_size, 0.0f);

  if (state.num_partitions > 0)
  {
    std::vector<LinearFFTState::Complex> warm_spectrum(state.fft_size);
    std::vector<float> warm_time(state.fft_size, 0.0f);
    state.fft.fwd(warm_spectrum.data(), warm_time.data(), state.fft_size);
    state.fft.inv(warm_time.data(), warm_spectrum.data(), state.fft_size);
  }
}

void nam::Linear::_process_direct(NAM_SAMPLE** input, NAM_SAMPLE** output, const int num_frames)
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

void nam::Linear::_process_fft(NAM_SAMPLE** input, NAM_SAMPLE** output, const int num_frames)
{
  this->nam::Buffer::_update_buffers_(input, num_frames);

  const int in_channels = NumInputChannels();
  const int out_channels = NumOutputChannels();
  const int channels_to_process = std::min(in_channels, out_channels);
  auto& state = *this->_fft_state;
  const int direct_taps = state.direct_taps;

  for (int i = 0; i < num_frames; i++)
  {
    const long direct_offset = this->_input_buffer_offset - direct_taps + i + 1;
    for (int ch = 0; ch < channels_to_process; ch++)
    {
      const int ring_index = (int)(state.sample_index % state.output_ring_size);
      const float tail = state.channels[ch].output_ring[ring_index];
      state.channels[ch].output_ring[ring_index] = 0.0f;

      auto input_vec = Eigen::Map<const Eigen::VectorXf>(&this->_input_buffers[ch][direct_offset], direct_taps);
      output[ch][i] = this->_bias + this->_fft_direct_weight.dot(input_vec) + tail;

      if (state.num_partitions > 0)
      {
        auto& channel = state.channels[ch];
        channel.input_time[channel.input_pos] = (float)input[ch][i];
        channel.input_pos++;
        if (channel.input_pos == state.block_size)
          this->_run_fft_block(ch);
      }
    }

    for (int ch = channels_to_process; ch < out_channels; ch++)
      output[ch][i] = (NAM_SAMPLE)0.0;

    state.sample_index++;
  }

  nam::Buffer::_advance_input_buffer_(num_frames);
}

void nam::Linear::_run_fft_block(const int channel_index)
{
  auto& state = *this->_fft_state;
  auto& channel = state.channels[channel_index];

  auto& current_spectrum = channel.input_spectra[channel.spectrum_write_index];
  state.fft.fwd(current_spectrum.data(), channel.input_time.data(), state.fft_size);

  std::fill(state.accumulator.begin(), state.accumulator.end(), LinearFFTState::Complex{});
  for (int partition = 0; partition < state.num_partitions; partition++)
  {
    int input_spectrum_index = channel.spectrum_write_index - partition;
    if (input_spectrum_index < 0)
      input_spectrum_index += state.num_partitions;
    const auto& input_spectrum = channel.input_spectra[input_spectrum_index];
    const auto& kernel_spectrum = state.kernel_spectra[partition];
    for (int bin = 0; bin < state.fft_size; bin++)
      state.accumulator[bin] += input_spectrum[bin] * kernel_spectrum[bin];
  }

  state.fft.inv(state.ifft_time.data(), state.accumulator.data(), state.fft_size);

  const long long block_start = state.sample_index - state.block_size + 1;
  const long long output_start = block_start + state.direct_taps;
  auto& output_ring = channel.output_ring;
  for (int i = 0; i < state.fft_size - 1; i++)
  {
    const int ring_index = (int)((output_start + i) % state.output_ring_size);
    output_ring[ring_index] += state.ifft_time[i];
  }

  std::fill(channel.input_time.begin(), channel.input_time.begin() + state.block_size, 0.0f);
  channel.input_pos = 0;
  channel.spectrum_write_index++;
  if (channel.spectrum_write_index == state.num_partitions)
    channel.spectrum_write_index = 0;
}

nam::LinearImplementation nam::linear::parse_implementation(const std::string& implementation)
{
  std::string normalized = implementation;
  std::transform(
    normalized.begin(), normalized.end(), normalized.begin(), [](unsigned char c) { return (char)std::tolower(c); });

  if (normalized == "auto")
    return LinearImplementation::Auto;
  if (normalized == "direct" || normalized == "legacy" || normalized == "old")
    return LinearImplementation::Direct;
  if (normalized == "fft" || normalized == "partitioned_fft" || normalized == "partitioned-fft")
    return LinearImplementation::FFT;
  throw std::runtime_error("Unsupported Linear implementation: " + implementation);
}

std::string nam::linear::implementation_to_string(const LinearImplementation implementation)
{
  switch (implementation)
  {
    case LinearImplementation::Auto: return "auto";
    case LinearImplementation::Direct: return "direct";
    case LinearImplementation::FFT: return "fft";
  }
  throw std::runtime_error("Unsupported Linear implementation enum");
}

nam::linear::LinearConfig nam::linear::parse_config_json(const nlohmann::json& config)
{
  LinearConfig c;
  c.receptive_field = config["receptive_field"];
  c.bias = config["bias"];
  // Default to 1 channel in/out for backward compatibility
  c.in_channels = config.value("in_channels", 1);
  c.out_channels = config.value("out_channels", 1);
  c.implementation = parse_implementation(config.value("implementation", "auto"));
  return c;
}

std::unique_ptr<nam::DSP> nam::linear::LinearConfig::create(std::vector<float> weights, double sampleRate)
{
  return std::make_unique<nam::Linear>(
    in_channels, out_channels, receptive_field, bias, weights, sampleRate, implementation);
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
