#include "linear.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cctype>
#include <complex>
#include <limits>
#include <stdexcept>

#include "registry.h"

#include <unsupported/Eigen/FFT>

namespace
{
struct LinearFFTDispatchEntry
{
  int max_taps;
  nam::LinearImplementation implementation;
  nam::LinearFFTPlan plan;
};

// Tuned using tools/bench_linear. The table keeps model-size policy separate
// from the DSP implementation.
constexpr std::array<LinearFFTDispatchEntry, 7> _LINEAR_FFT_DISPATCH{{
  {1024, nam::LinearImplementation::Direct, {128, 256}},
  {2048, nam::LinearImplementation::FFT, {128, 512}},
  {4096, nam::LinearImplementation::FFT, {128, 1024}},
  {8192, nam::LinearImplementation::FFT, {128, 2048}},
  {48000, nam::LinearImplementation::FFT, {64, 4096}},
  {240000, nam::LinearImplementation::FFT, {64, 8192}},
  {std::numeric_limits<int>::max(), nam::LinearImplementation::FFT, {64, 8192}},
}};

int _ceil_div(const int numerator, const int denominator)
{
  return (numerator + denominator - 1) / denominator;
}

} // namespace

struct nam::LinearFFTState
{
  using Complex = std::complex<float>;

  struct TierChannelState
  {
    std::vector<float> input_time;
    std::vector<std::vector<Complex>> input_spectra;
    std::vector<Complex> accumulator;
    std::vector<float> ifft_time;
    int input_pos = 0;
    int spectrum_write_index = 0;
    int job_spectrum_write_index = 0;
    size_t job_work_index = 0;
    int job_ticks_remaining = 0;
    long long job_block_start = 0;
    bool job_active = false;
  };

  struct Tier
  {
    Eigen::FFT<float> fft;
    int offset = 0;
    int block_size = 0;
    int fft_size = 0;
    int spectrum_size = 0;
    int num_partitions = 0;
    bool runs_inline = false;
    std::vector<std::vector<Complex>> kernel_spectra;
    std::vector<TierChannelState> channels;
  };

  struct OutputChannelState
  {
    std::vector<float> output_ring;
  };

  int direct_taps = 0;
  int output_ring_size = 0;
  long long sample_index = 0;
  std::vector<Tier> tiers;
  std::vector<OutputChannelState> output_channels;
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
    this->_active_implementation = linear::select_implementation(this->_receptive_field);

  if (this->_active_implementation == LinearImplementation::FFT)
    this->_configure_fft_state();
  else
    this->_fft_state.reset();
}

void nam::Linear::_configure_fft_state()
{
  this->_fft_state = std::make_unique<LinearFFTState>();
  auto& state = *this->_fft_state;
  const auto plan = linear::select_fft_plan(this->_receptive_field);
  state.direct_taps = std::min(this->_receptive_field, plan.direct_taps);
  state.sample_index = 0;

  this->_fft_direct_weight.resize(state.direct_taps);
  for (int i = 0; i < state.direct_taps; i++)
    this->_fft_direct_weight(i) = this->_impulse_response[state.direct_taps - 1 - i];

  // The inline tier covers [head, 4 * head). Every subsequent power-of-two
  // tier starts at twice its block size, which gives it one full block of
  // scheduling slack. Once the tuned maximum block size is reached, the last
  // tier simply contains as many uniform partitions as are needed.
  int offset = state.direct_taps;
  int block_size = state.direct_taps;
  while (offset < this->_receptive_field)
  {
    const bool first_tier = state.tiers.empty();
    const int partitions = first_tier ? std::min(3, _ceil_div(this->_receptive_field - offset, block_size))
                           : block_size == plan.max_partition_size
                             ? _ceil_div(this->_receptive_field - offset, block_size)
                             : std::min(2, _ceil_div(this->_receptive_field - offset, block_size));

    auto& tier = state.tiers.emplace_back();
    tier.fft.SetFlag(Eigen::FFT<float>::HalfSpectrum);
    tier.offset = offset;
    tier.block_size = block_size;
    tier.fft_size = 2 * block_size;
    tier.spectrum_size = block_size + 1;
    tier.num_partitions = partitions;
    tier.runs_inline = first_tier;
    tier.kernel_spectra.assign(partitions, std::vector<LinearFFTState::Complex>(tier.spectrum_size));

    std::vector<float> kernel_time(tier.fft_size, 0.0f);
    for (int partition = 0; partition < partitions; partition++)
    {
      std::fill(kernel_time.begin(), kernel_time.end(), 0.0f);
      const int start = offset + partition * block_size;
      const int partition_size = std::min(block_size, this->_receptive_field - start);
      std::copy_n(this->_impulse_response.begin() + start, partition_size, kernel_time.begin());
      tier.fft.fwd(tier.kernel_spectra[partition].data(), kernel_time.data(), tier.fft_size);
    }

    offset += partitions * block_size;
    if (block_size < plan.max_partition_size)
      block_size = std::min(2 * block_size, plan.max_partition_size);
  }

  const int channels_to_process = std::min(NumInputChannels(), NumOutputChannels());
  int largest_block_size = state.direct_taps;
  for (auto& tier : state.tiers)
  {
    largest_block_size = std::max(largest_block_size, tier.block_size);
    tier.channels.resize(channels_to_process);
    for (auto& channel : tier.channels)
    {
      channel.input_time.assign(tier.fft_size, 0.0f);
      channel.input_spectra.assign(
        tier.num_partitions, std::vector<LinearFFTState::Complex>(tier.spectrum_size, LinearFFTState::Complex{}));
      channel.accumulator.assign(tier.spectrum_size, LinearFFTState::Complex{});
      channel.ifft_time.assign(tier.fft_size, 0.0f);
      // Half-block phase offsets keep power-of-two tiers from all transforming
      // in the same callback. The leading inline tier must remain unshifted.
      channel.input_pos = tier.runs_inline ? 0 : tier.block_size / 2;
    }
  }

  state.output_ring_size = 4 * largest_block_size;
  state.output_channels.resize(channels_to_process);
  for (auto& channel : state.output_channels)
    channel.output_ring.assign(state.output_ring_size, 0.0f);

  // Create all FFT plans outside the audio callback.
  for (auto& tier : state.tiers)
  {
    std::vector<LinearFFTState::Complex> warm_spectrum(tier.spectrum_size);
    std::vector<float> warm_time(tier.fft_size, 0.0f);
    tier.fft.fwd(warm_spectrum.data(), warm_time.data(), tier.fft_size);
    tier.fft.inv(warm_time.data(), warm_spectrum.data(), tier.fft_size);
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
      this->_advance_fft_jobs(ch);

      const int ring_index = (int)(state.sample_index % state.output_ring_size);
      const float tail = state.output_channels[ch].output_ring[ring_index];
      state.output_channels[ch].output_ring[ring_index] = 0.0f;

      auto input_vec = Eigen::Map<const Eigen::VectorXf>(&this->_input_buffers[ch][direct_offset], direct_taps);
      output[ch][i] = this->_bias + this->_fft_direct_weight.dot(input_vec) + tail;

      for (size_t tier_index = 0; tier_index < state.tiers.size(); ++tier_index)
      {
        auto& tier = state.tiers[tier_index];
        auto& channel = tier.channels[ch];
        channel.input_time[channel.input_pos] = (float)input[ch][i];
        channel.input_pos++;
        if (channel.input_pos == tier.block_size)
        {
          const long long block_start = state.sample_index - tier.block_size + 1;
          this->_start_fft_block((int)tier_index, ch, block_start);
          channel.input_pos = 0;
        }
      }
    }

    for (int ch = channels_to_process; ch < out_channels; ch++)
      output[ch][i] = (NAM_SAMPLE)0.0;

    state.sample_index++;
  }

  nam::Buffer::_advance_input_buffer_(num_frames);
}

void nam::Linear::_advance_fft_jobs(const int channel_index)
{
  for (size_t tier_index = 0; tier_index < this->_fft_state->tiers.size(); ++tier_index)
    this->_advance_fft_job((int)tier_index, channel_index);
}

void nam::Linear::_advance_fft_job(const int tier_index, const int channel_index)
{
  auto& tier = this->_fft_state->tiers[tier_index];
  auto& channel = tier.channels[channel_index];
  if (!channel.job_active)
    return;

  const size_t total_work = (size_t)tier.num_partitions * tier.spectrum_size;
  const size_t remaining_work = total_work - channel.job_work_index;
  const size_t work_this_tick =
    (remaining_work + (size_t)channel.job_ticks_remaining - 1) / (size_t)channel.job_ticks_remaining;
  const size_t work_end = std::min(total_work, channel.job_work_index + work_this_tick);
  while (channel.job_work_index < work_end)
  {
    const int partition = (int)(channel.job_work_index / (size_t)tier.spectrum_size);
    const int bin = (int)(channel.job_work_index % (size_t)tier.spectrum_size);
    int input_spectrum_index = channel.job_spectrum_write_index - partition;
    if (input_spectrum_index < 0)
      input_spectrum_index += tier.num_partitions;
    channel.accumulator[bin] += channel.input_spectra[input_spectrum_index][bin] * tier.kernel_spectra[partition][bin];
    channel.job_work_index++;
  }
  channel.job_ticks_remaining--;
  if (channel.job_work_index == total_work)
    this->_finish_fft_block(tier_index, channel_index);
}

void nam::Linear::_start_fft_block(const int tier_index, const int channel_index, const long long block_start)
{
  auto& tier = this->_fft_state->tiers[tier_index];
  auto& channel = tier.channels[channel_index];
  assert(!channel.job_active);

  channel.job_spectrum_write_index = channel.spectrum_write_index;
  auto& current_spectrum = channel.input_spectra[channel.job_spectrum_write_index];
  tier.fft.fwd(current_spectrum.data(), channel.input_time.data(), tier.fft_size);
  std::fill(channel.accumulator.begin(), channel.accumulator.end(), LinearFFTState::Complex{});
  channel.job_work_index = 0;
  channel.job_ticks_remaining = tier.runs_inline ? 1 : tier.block_size;
  channel.job_block_start = block_start;
  channel.job_active = true;

  channel.spectrum_write_index++;
  if (channel.spectrum_write_index == tier.num_partitions)
    channel.spectrum_write_index = 0;

  std::fill(channel.input_time.begin(), channel.input_time.begin() + tier.block_size, 0.0f);

  if (tier.runs_inline)
    this->_advance_fft_job(tier_index, channel_index);
}

void nam::Linear::_finish_fft_block(const int tier_index, const int channel_index)
{
  auto& state = *this->_fft_state;
  auto& tier = state.tiers[tier_index];
  auto& channel = tier.channels[channel_index];

  tier.fft.inv(channel.ifft_time.data(), channel.accumulator.data(), tier.fft_size);

  const long long output_start = channel.job_block_start + tier.offset;
  auto& output_ring = state.output_channels[channel_index].output_ring;
  for (int i = 0; i < tier.fft_size - 1; i++)
  {
    const int ring_index = (int)((output_start + i) % state.output_ring_size);
    output_ring[ring_index] += channel.ifft_time[i];
  }
  channel.job_active = false;
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

nam::LinearFFTPlan nam::linear::select_fft_plan(const int receptive_field)
{
  for (const auto& entry : _LINEAR_FFT_DISPATCH)
    if (receptive_field <= entry.max_taps)
      return entry.plan;
  throw std::runtime_error("No Linear FFT dispatch entry for receptive field");
}

nam::LinearImplementation nam::linear::select_implementation(const int receptive_field)
{
  for (const auto& entry : _LINEAR_FFT_DISPATCH)
    if (receptive_field <= entry.max_taps)
      return entry.implementation;
  throw std::runtime_error("No Linear implementation dispatch entry for receptive field");
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
