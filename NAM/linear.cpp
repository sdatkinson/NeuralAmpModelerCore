#include "linear.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cctype>
#include <complex>
#include <cmath>
#include <cstring>
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

// The table keeps model-size policy separate from the DSP implementation.
//
// Chosen on the 99th-percentile callback.
//
// Every transform lands in one callback, so that callback is what decides
// whether a block is late.
//
// Partitions are uniform. A longer partition means fewer partitions to
// multiply but a longer direct head and a larger transform in the callback
// that runs it.
//
// Direct up to 1024 taps. FFT has the lower mean from 513, but with 16- and
// 32-frame callbacks its transform callback is slower than direct convolution's
// flat profile until well past 1024 taps.
//
// Past a second of impulse response, however, uniform partitions cost too many
// multiplies per sample.
constexpr std::array<LinearFFTDispatchEntry, 6> _LINEAR_FFT_DISPATCH{{
  {1024, nam::LinearImplementation::Direct, {256, 256, 0}},
  {2048, nam::LinearImplementation::FFT, {256, 256, 0}},
  {8192, nam::LinearImplementation::FFT, {512, 512, 0}},
  {48000, nam::LinearImplementation::FFT, {1024, 1024, 0}},
  {240000, nam::LinearImplementation::FFT, {1024, 1024, 8192}},
  {std::numeric_limits<int>::max(), nam::LinearImplementation::FFT, {1024, 1024, 16384}},
}};

// Direct-head history holds this many maximum-size callbacks beyond the head.
// Moving the head back to the start of the buffer is uncommon and cheap.
constexpr int _FFT_HISTORY_CALLBACKS = 8;

int _ceil_div(const int numerator, const int denominator)
{
  return (numerator + denominator - 1) / denominator;
}

// Adapted from AudioDSPTools dsp/Resample.h:
// https://github.com/sdatkinson/AudioDSPTools/blob/844680d118f0317565132c3c5e3aca5f5c976e7a/dsp/Resample.h
// Sample by integer output index to avoid accumulated timing error,
// and zero-pad the causal impulse response at both ends for cubic interpolation.
std::vector<float> _resample_impulse_response(const std::vector<float>& inputs, const double original_rate,
                                              const double desired_rate)
{
  const double ratio = desired_rate / original_rate;
  const double length = std::ceil(inputs.size() * ratio);
  if (!std::isfinite(length) || length > std::numeric_limits<int>::max())
    throw std::length_error("Resampled Linear impulse response is too large");
  // Even an IR shorter than one output sample needs its sample at t=0.
  std::vector<float> outputs((size_t)std::max(1.0, length));
  const auto sample = [&inputs](const long long index) -> double {
    return index < 0 || index >= (long long)inputs.size() ? 0.0 : inputs[(size_t)index];
  };
  for (size_t i = 0; i < outputs.size(); ++i)
  {
    const double position = i == 0 ? 0.0 : i / ratio;
    const long long index = (long long)std::floor(position);
    const double x = position - index;
    const double p[4] = {sample(index - 1), sample(index), sample(index + 1), sample(index + 2)};
    const double value =
      p[1]
      + 0.5 * x
          * (p[2] - p[0] + x * (2.0 * p[0] - 5.0 * p[1] + 4.0 * p[2] - p[3] + x * (3.0 * (p[1] - p[2]) + p[3] - p[0])));
    // An IR is integrated by convolution; compensate for the changed tap density.
    outputs[i] = (float)(value * (original_rate / desired_rate));
  }
  return outputs;
}

} // namespace

struct nam::LinearFFTState
{
  using Complex = std::complex<float>;

  struct InputChannelState
  {
    // Contiguous float history.
    //
    // The direct head's window for every sample of a callback lies inside
    // it, and so does each block the transforms read, which is the same
    // window at the block's last sample.
    std::vector<float> history;
    // Where the next callback's first sample goes. Always >= direct_taps.
    size_t history_index = 0;
    // The spectra of the last num_partitions input blocks, as a ring.
    std::vector<std::vector<Complex>> spectra;
  };

  struct OutputChannelState
  {
    // The next transform's spectrum, summed over every path into this output.
    std::vector<Complex> accumulator;
    // The last two inverse transforms, fft_size samples each.
    //
    // Sample input_pos of the block being collected takes sample input_pos of
    // the newer and block_size + input_pos of the older, so each output sample
    // does its own overlap-add as it goes out and no transform pays for all of
    // it at once.
    std::vector<float> ifft_current;
    std::vector<float> ifft_previous;
  };

  Eigen::FFT<float> fft;
  // Samples per partition: also the direct head's length, and how many samples
  // arrive between transforms.
  int block_size = 0;
  int fft_size = 0;
  // Bins stored and multiplied: fft_size / 2 + 1. Everything convolved here is
  // real, so the upper half of every spectrum mirrors the lower half and the
  // real-output inverse transform never reads it.
  int num_bins = 0;
  int direct_taps = 0;
  int num_partitions = 0;

  // A path is one (input, kernel, output) triple; see the Linear class comment.
  int num_paths = 0;
  std::vector<int> path_input;
  std::vector<int> path_kernel;
  std::vector<int> path_output;

  // [kernel][partition][bin]
  std::vector<std::vector<std::vector<Complex>>> kernel_spectra;
  std::vector<InputChannelState> inputs;
  std::vector<OutputChannelState> outputs;
  // One block of input followed by zeros: what every forward transform reads.
  std::vector<float> transform_input;

  // How many samples of the current block have arrived.
  int input_pos = 0;
  // The ring slot the next transform writes.
  int spectrum_write_index = 0;

  // Partitions 1..num_partitions-1 of the next transform multiply spectra of
  // blocks that have already arrived, so that work is done ahead, in the
  // callbacks that run no transform; only partition 0 has to wait for the
  // block that completes.
  //
  // One unit is one (path, partition, bin), ordered by path, then partition,
  // then bin. pending_done is how far this block has got.
  int pending_units = 0;
  int pending_done = 0;

  // The tail tier
  //
  // Partitions of tail.block_size (T) taps from 2T on. Its
  // blocks arrive T / block_size times less often than the uniform ones and
  // each of its transforms covers that many more taps, so a long impulse
  // response costs far fewer multiplies per sample.
  //
  // A tail block's result is first needed T samples after the block completes,
  // and all of its work is spread across that gap: not only the multiplies but
  // the transforms too.
  //
  // A 2T-point transform in one callback would be the large, rare callback
  // this design exists to avoid, so each one is done as M = 2T / L transforms
  // of L = 2 * block_size points, the uniform tier's own size, plus a
  // combining pass (Cooley-Tukey with the L-point transforms as the inner
  // step).
  //
  // The job is a fixed sequence of small steps, and each callback does its
  // share by estimated cost, so no callback does more than one L-point
  // transform beyond its share of the arithmetic.
  //
  // The sequence and the arithmetic in each step do not depend on how the
  // input is split into callbacks, so neither does the output.
  struct TailInputState
  {
    // The block being collected, and the last complete one.
    std::vector<float> collecting;
    std::vector<float> ready;
    // The last num_partitions blocks' spectra, as a ring.
    std::vector<std::vector<Complex>> spectra;
    // [sub][bin]: the L-point transforms of the block's M interleaved phases.
    std::vector<std::vector<Complex>> sub_spectra;
  };
  struct TailOutputState
  {
    std::vector<Complex> accumulator;
    // [sub][bin]: the inverse's M phases, before their L-point transforms.
    std::vector<std::vector<Complex>> sub_spectra;
    // Tail output still to be played, indexed by sample count modulo its size.
    std::vector<float> ring;
  };
  struct Tail
  {
    Eigen::FFT<float> fft;
    int block_size = 0; // T
    int fft_size = 0; // N = 2T
    int num_bins = 0; // T + 1
    int num_partitions = 0;
    int sub_size = 0; // L
    int num_subs = 0; // M = N / L
    int sub_bins = 0; // L / 2 + 1
    // twiddle[k] = exp(-2 pi i k / N)
    std::vector<Complex> twiddle;
    // [kernel][partition][bin]
    std::vector<std::vector<std::vector<Complex>>> kernel_spectra;
    std::vector<TailInputState> inputs;
    std::vector<TailOutputState> outputs;
    std::vector<float> sub_time;
    int collect_pos = 0;
    int ring_size = 0;
    int ring_pos = 0;

    // The job for the last complete block. Steps run in this order, each
    // indexed by a cursor within its stage:
    //   0 forward phase transforms   (input, sub)          one L-point transform each
    //   1 forward combine            (input, bin)          num_subs multiplies each
    //   2 multiplies                 (path, partition, bin) one each
    //   3 inverse split              (output, sub, bin)    num_subs multiplies each
    //   4 inverse phase transforms   (output, sub)         one L-point transform each
    bool job_active = false;
    int stage = 0;
    int cursor = 0;
    long long remaining_cost = 0;
    int job_clock = 0; // samples since the block completed
    int job_output_pos = 0; // ring index where its output starts
    int spectrum_write_index = 0;
    long long transform_cost = 0;
  };
  Tail tail;
  bool has_tail_tier = false;

  int tail_stage_count(const int stage) const
  {
    switch (stage)
    {
      case 0: return (int)tail.inputs.size() * tail.num_subs;
      case 1: return (int)tail.inputs.size() * tail.num_bins;
      case 2: return num_paths * tail.num_partitions * tail.num_bins;
      case 3: return (int)tail.outputs.size() * tail.num_subs * tail.sub_bins;
      case 4: return (int)tail.outputs.size() * tail.num_subs;
    }
    return 0;
  }

  long long tail_step_cost(const int stage) const
  {
    return stage == 0 || stage == 4 ? tail.transform_cost : stage == 2 ? 1 : tail.num_subs;
  }

  long long tail_job_cost() const
  {
    long long cost = 0;
    for (int stage = 0; stage < 5; ++stage)
      cost += (long long)tail_stage_count(stage) * tail_step_cost(stage);
    return cost;
  }

  // X[k] of phase `sub`'s L-point spectrum, for any k: the stored half and its
  // conjugate mirror.
  static Complex half_bin(const std::vector<Complex>& spectrum, const int size, int k)
  {
    k %= size;
    return k <= size / 2 ? spectrum[k] : std::conj(spectrum[size - k]);
  }

  void tail_step(const int stage, const int index)
  {
    const int M = tail.num_subs;
    const int N = tail.fft_size;
    const int L = tail.sub_size;
    switch (stage)
    {
      case 0:
      {
        // Phase `sub` of the zero-padded block: x[M * n + sub].
        auto& input = tail.inputs[index / M];
        const int sub = index % M;
        for (int n = 0; n < L; ++n)
        {
          const int at = M * n + sub;
          tail.sub_time[n] = at < tail.block_size ? input.ready[at] : 0.0f;
        }
        tail.fft.fwd(input.sub_spectra[sub].data(), tail.sub_time.data(), L);
        break;
      }
      case 1:
      {
        auto& input = tail.inputs[index / tail.num_bins];
        const int k = index % tail.num_bins;
        Complex sum = half_bin(input.sub_spectra[0], L, k);
        for (int sub = 1; sub < M; ++sub)
          sum += tail.twiddle[(int)(((long long)sub * k) % N)] * half_bin(input.sub_spectra[sub], L, k);
        input.spectra[tail.spectrum_write_index][k] = sum;
        break;
      }
      case 3:
      {
        auto& output = tail.outputs[index / (M * tail.sub_bins)];
        const int within = index % (M * tail.sub_bins);
        const int sub = within / tail.sub_bins;
        const int k = within % tail.sub_bins;
        Complex sum{};
        for (int q = 0; q < M; ++q)
        {
          const int bin = k + L * q;
          const Complex x = bin <= tail.block_size ? output.accumulator[bin] : std::conj(output.accumulator[N - bin]);
          sum += x * std::conj(tail.twiddle[(int)(((long long)bin * sub) % N)]);
        }
        output.sub_spectra[sub][k] = sum;
        break;
      }
      case 4:
      {
        auto& output = tail.outputs[index / M];
        const int sub = index % M;
        tail.fft.inv(tail.sub_time.data(), output.sub_spectra[sub].data(), L);
        // The L-point inverse divides by L; the whole transform divides by N.
        const float scale = 1.0f / (float)M;
        for (int n = 0; n < L; ++n)
        {
          const int at = M * n + sub;
          // Two T-long pieces convolve to 2T - 1 samples; the last is wrap-around.
          if (at == N - 1)
            continue;
          int pos = tail.job_output_pos + at;
          if (pos >= tail.ring_size)
            pos -= tail.ring_size;
          output.ring[pos] += tail.sub_time[n] * scale;
        }
        break;
      }
    }
  }

  void tail_multiply_add(const int path, const int partition, const int bin_begin, const int bin_end)
  {
    int input_index = tail.spectrum_write_index - partition;
    if (input_index < 0)
      input_index += tail.num_partitions;
    const Complex* x = tail.inputs[path_input[path]].spectra[input_index].data();
    const Complex* h = tail.kernel_spectra[path_kernel[path]][partition].data();
    Complex* accumulator = tail.outputs[path_output[path]].accumulator.data();
    for (int bin = bin_begin; bin < bin_end; ++bin)
      accumulator[bin] += x[bin] * h[bin];
  }

  // Up to `budget` of the job's estimated cost, in order. Always makes
  // progress if there is any work left.
  void tail_work(long long budget)
  {
    while (tail.job_active && budget > 0)
    {
      const int count = tail_stage_count(tail.stage);
      if (tail.cursor >= count)
      {
        if (++tail.stage == 5)
        {
          tail.job_active = false;
          if (++tail.spectrum_write_index == tail.num_partitions)
            tail.spectrum_write_index = 0;
        }
        tail.cursor = 0;
        continue;
      }
      if (tail.stage == 2)
      {
        const int units_per_path = tail.num_partitions * tail.num_bins;
        const int path = tail.cursor / units_per_path;
        const int within = tail.cursor - path * units_per_path;
        const int partition = within / tail.num_bins;
        const int bin = within - partition * tail.num_bins;
        const int bin_end = (int)std::min<long long>(tail.num_bins, bin + budget);
        tail_multiply_add(path, partition, bin, bin_end);
        tail.cursor += bin_end - bin;
        budget -= bin_end - bin;
        tail.remaining_cost -= bin_end - bin;
      }
      else
      {
        tail_step(tail.stage, tail.cursor++);
        const long long cost = tail_step_cost(tail.stage);
        budget -= cost;
        tail.remaining_cost -= cost;
      }
    }
  }

  // One input sample, already in every input channel's history at `index`,
  // has been played. The tail's output for it was read before it arrived.
  void tail_advance(const size_t index)
  {
    for (size_t ch = 0; ch < tail.inputs.size(); ++ch)
      tail.inputs[ch].collecting[tail.collect_pos] = inputs[ch].history[index];
    if (tail.job_active)
      ++tail.job_clock;
    if (++tail.collect_pos == tail.block_size)
    {
      // The previous block's job is due now; with regular callbacks there is
      // nothing left of it.
      tail_work(std::numeric_limits<long long>::max());
      for (auto& input : tail.inputs)
        input.collecting.swap(input.ready);
      tail.collect_pos = 0;
      for (auto& output : tail.outputs)
        std::fill(output.accumulator.begin(), output.accumulator.end(), Complex{});
      tail.job_active = true;
      tail.stage = 0;
      tail.cursor = 0;
      tail.remaining_cost = tail_job_cost();
      tail.job_clock = 0;
      // The block started T - 1 samples ago; its output starts 2T after that.
      tail.job_output_pos = (tail.ring_pos + 1 + tail.block_size) % tail.ring_size;
    }
    if (++tail.ring_pos == tail.ring_size)
      tail.ring_pos = 0;
  }

  // accumulator += input spectrum * kernel spectrum, for one path and one
  // partition over [bin_begin, bin_end), pairing the partition with the input
  // block it multiplies in the transform at spectrum_write_index.
  void multiply_add(const int path, const int partition, const int bin_begin, const int bin_end)
  {
    int input_index = spectrum_write_index - partition;
    if (input_index < 0)
      input_index += num_partitions;
    const Complex* x = inputs[path_input[path]].spectra[input_index].data();
    const Complex* h = kernel_spectra[path_kernel[path]][partition].data();
    Complex* accumulator = outputs[path_output[path]].accumulator.data();
    for (int bin = bin_begin; bin < bin_end; ++bin)
      accumulator[bin] += x[bin] * h[bin];
  }

  // The next `units` of the pending work.
  void do_pending(const int units)
  {
    const int units_per_path = (num_partitions - 1) * num_bins;
    const int end = std::min(pending_units, pending_done + units);
    while (pending_done < end)
    {
      const int path = pending_done / units_per_path;
      const int within_path = pending_done - path * units_per_path;
      const int partition = 1 + within_path / num_bins;
      const int bin = within_path - (partition - 1) * num_bins;
      const int bin_end = std::min(num_bins, bin + (end - pending_done));
      multiply_add(path, partition, bin, bin_end);
      pending_done += bin_end - bin;
    }
  }

  // A block has just completed at sample `frame` of the current callback.
  void transform(const int frame)
  {
    for (auto& input : inputs)
    {
      const float* block = input.history.data() + input.history_index + frame + 1 - block_size;
      std::copy_n(block, block_size, transform_input.begin());
      fft.fwd(input.spectra[spectrum_write_index].data(), transform_input.data(), fft_size);
    }

    // Whatever the callbacks since the last transform left undone, then
    // partition 0, the only one whose input spectrum did not exist until now.
    do_pending(pending_units - pending_done);
    for (int path = 0; path < num_paths; ++path)
      multiply_add(path, 0, 0, num_bins);

    for (auto& output : outputs)
    {
      output.ifft_current.swap(output.ifft_previous);
      fft.inv(output.ifft_current.data(), output.accumulator.data(), fft_size);
      // Two block_size-long pieces convolve to 2 * block_size - 1 samples; the
      // transform's last sample is circular wrap-around.
      output.ifft_current[fft_size - 1] = 0.0f;
      std::fill(output.accumulator.begin(), output.accumulator.end(), Complex{});
    }

    pending_done = 0;
    input_pos = 0;
    if (++spectrum_write_index == num_partitions)
      spectrum_write_index = 0;
  }
};

nam::Linear::Linear(const int in_channels, const int out_channels, const int receptive_field, const bool _bias,
                    const std::vector<float>& weights, const double expected_sample_rate,
                    const LinearImplementation implementation)
: nam::Buffer(in_channels, out_channels, receptive_field, expected_sample_rate)
, _requested_implementation(implementation)
, _active_implementation(LinearImplementation::Direct)
{
  if (in_channels != out_channels && in_channels != 1 && out_channels != 1)
    throw std::runtime_error("Linear requires equal channel counts or one input/output channel");
  const int kernels = in_channels == out_channels ? 1 : std::max(in_channels, out_channels);
  const int biases = in_channels == out_channels ? 1 : out_channels;
  const size_t coefficient_count = (size_t)receptive_field * kernels;
  if (weights.size() != coefficient_count + (_bias ? biases : 0))
    throw std::runtime_error("Linear parameter count does not match impulse responses and output biases");

  this->_impulse_response.resize(kernels);
  for (int k = 0; k < kernels; ++k)
    this->_impulse_response[k].assign(
      weights.begin() + (size_t)k * receptive_field, weights.begin() + (size_t)(k + 1) * receptive_field);
  this->_original_impulse_response = this->_impulse_response;
  this->_configure_weights();
  this->_bias.resize(out_channels, 0.0f);
  if (_bias)
    for (int ch = 0; ch < out_channels; ++ch)
      this->_bias[ch] = weights[coefficient_count + (biases == 1 ? 0 : ch)];

  this->_configure_implementation();
}

nam::Linear::~Linear() = default;

bool nam::Linear::SupportsArbitrarySampleRate()
{
  const double training_rate = GetExpectedSampleRate();
  return std::isfinite(training_rate) && training_rate > 0.0;
}

void nam::Linear::process(NAM_SAMPLE** input, NAM_SAMPLE** output, const int num_frames)
{
  if (this->_active_implementation == LinearImplementation::FFT)
    this->_process_fft(input, output, num_frames);
  else
    this->_process_direct(input, output, num_frames);
}

void nam::Linear::Reset(const double sampleRate, const int maxBufferSize)
{
  if (!std::isfinite(sampleRate) || sampleRate <= 0.0)
    throw std::invalid_argument("Linear processing sample rate must be finite and positive");
  const double training_rate = GetExpectedSampleRate();
  if (training_rate != NAM_UNKNOWN_EXPECTED_SAMPLE_RATE && (!std::isfinite(training_rate) || training_rate <= 0.0))
    throw std::invalid_argument("Linear training sample rate must be finite and positive, or unknown (-1)");
  if (maxBufferSize < 0)
    throw std::invalid_argument("Linear maximum buffer size must be non-negative");

  auto impulse_responses = this->_original_impulse_response;
  if (training_rate != NAM_UNKNOWN_EXPECTED_SAMPLE_RATE && training_rate != sampleRate)
    for (auto& response : impulse_responses)
      response = _resample_impulse_response(response, training_rate, sampleRate);
  if (impulse_responses.front().size() + 32LL * maxBufferSize > std::numeric_limits<int>::max())
    throw std::length_error("Linear input buffer is too large");
  this->_impulse_response = std::move(impulse_responses);
  this->_receptive_field = (int)this->_impulse_response.front().size();
  this->_configure_weights();
  // SetMaxBufferSize rebuilds history and FFT state before any prewarming.
  nam::DSP::Reset(sampleRate, maxBufferSize);
}

void nam::Linear::SetMaxBufferSize(const int maxBufferSize)
{
  const long long input_buffer_size = this->_receptive_field + 32LL * maxBufferSize;
  if (maxBufferSize < 0 || input_buffer_size > std::numeric_limits<int>::max())
    throw std::length_error("Linear input buffer is too large");
  nam::Buffer::SetMaxBufferSize(maxBufferSize);
  // Match Buffer::_update_buffers_ capacity requirements before entering process().
  this->_set_receptive_field(this->_receptive_field, (int)input_buffer_size);
  for (auto& output : this->_output_buffers)
    output.resize(maxBufferSize);
  this->_configure_implementation();
}

void nam::Linear::_configure_weights()
{
  this->_weight.resize(this->_impulse_response.size());
  for (size_t k = 0; k < this->_weight.size(); ++k)
  {
    this->_weight[k].resize(this->_receptive_field);
    for (int i = 0; i < this->_receptive_field; ++i)
      this->_weight[k](i) = this->_impulse_response[k][this->_receptive_field - 1 - i];
  }
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

  // The transforms read their block from the direct head's window, and the
  // first partition's output starts where the head ends.
  assert(plan.direct_taps == plan.max_partition_size);
  const int in_channels = NumInputChannels();
  const int out_channels = NumOutputChannels();
  const int kernels = (int)this->_impulse_response.size();

  state.block_size = plan.max_partition_size;
  state.fft_size = 2 * state.block_size;
  state.num_bins = state.fft_size / 2 + 1;

  // Without this, the forward transform computes the lower half and then copies
  // conjugates into the upper half, which would then be multiplied for nothing.
  state.fft.SetFlag(Eigen::FFT<float>::HalfSpectrum);
  state.direct_taps = std::min(this->_receptive_field, plan.direct_taps);

  // A tail tier takes over at 2 * its partition size, making sense only
  // when there are several of its partitions before that mark.
  const int tail_size = plan.tail_partition_size;
  state.has_tail_tier = tail_size > 0 && this->_receptive_field > 2 * tail_size;
  assert(!state.has_tail_tier || (tail_size % state.block_size == 0 && tail_size >= 2 * state.block_size));

  const int uniform_end = state.has_tail_tier ? 2 * tail_size : this->_receptive_field;
  state.num_partitions =
    uniform_end > state.direct_taps ? _ceil_div(uniform_end - state.direct_taps, state.block_size) : 0;

  state.num_paths = std::max(in_channels, out_channels);
  state.path_input.resize(state.num_paths);
  state.path_kernel.resize(state.num_paths);
  state.path_output.resize(state.num_paths);

  for (int path = 0; path < state.num_paths; ++path)
  {
    state.path_input[path] = in_channels == 1 ? 0 : path;
    state.path_kernel[path] = this->_kernel_index(path);
    state.path_output[path] = out_channels == 1 ? 0 : path;
  }

  const long long pending_units =
    state.num_partitions > 1 ? (long long)state.num_paths * (state.num_partitions - 1) * state.num_bins : 0;

  if (pending_units > std::numeric_limits<int>::max())
    throw std::length_error("Linear impulse response is too large for the FFT path");

  state.pending_units = (int)pending_units;
  state.pending_done = 0;
  state.input_pos = 0;
  state.spectrum_write_index = 0;

  this->_fft_direct_weight.resize(kernels);

  for (int k = 0; k < kernels; ++k)
  {
    this->_fft_direct_weight[k].resize(state.direct_taps);
    for (int i = 0; i < state.direct_taps; ++i)
      this->_fft_direct_weight[k](i) = this->_impulse_response[k][state.direct_taps - 1 - i];
  }

  state.kernel_spectra.resize(kernels);
  std::vector<float> kernel_time(state.fft_size, 0.0f);
  for (int k = 0; k < kernels; ++k)
  {
    auto& spectra = state.kernel_spectra[k];
    spectra.assign(state.num_partitions, std::vector<LinearFFTState::Complex>(state.num_bins));
    for (int partition = 0; partition < state.num_partitions; ++partition)
    {
      std::fill(kernel_time.begin(), kernel_time.end(), 0.0f);
      const int start = state.direct_taps + partition * state.block_size;
      const int partition_size = std::min(state.block_size, uniform_end - start);
      std::copy_n(this->_impulse_response[k].begin() + start, partition_size, kernel_time.begin());
      state.fft.fwd(spectra[partition].data(), kernel_time.data(), state.fft_size);
    }
  }

  const size_t history_size =
    (size_t)state.direct_taps + (size_t)_FFT_HISTORY_CALLBACKS * (size_t)std::max(1, GetMaxBufferSize());
  state.inputs.resize(in_channels);
  for (auto& input : state.inputs)
  {
    input.history.assign(history_size, 0.0f);
    input.history_index = state.direct_taps;
    input.spectra.assign(
      state.num_partitions, std::vector<LinearFFTState::Complex>(state.num_bins, LinearFFTState::Complex{}));
  }
  state.outputs.resize(out_channels);
  for (auto& output : state.outputs)
  {
    output.accumulator.assign(state.num_bins, LinearFFTState::Complex{});
    output.ifft_current.assign(state.fft_size, 0.0f);
    output.ifft_previous.assign(state.fft_size, 0.0f);
  }
  state.transform_input.assign(state.fft_size, 0.0f);

  if (state.has_tail_tier)
  {
    auto& tail = state.tail;
    tail.block_size = tail_size;
    tail.fft_size = 2 * tail_size;
    tail.num_bins = tail_size + 1;
    tail.sub_size = state.fft_size;
    tail.num_subs = tail.fft_size / tail.sub_size;
    tail.sub_bins = tail.sub_size / 2 + 1;
    tail.fft.SetFlag(Eigen::FFT<float>::HalfSpectrum);
    tail.num_partitions = _ceil_div(this->_receptive_field - uniform_end, tail_size);
    if ((long long)state.num_paths * tail.num_partitions * tail.num_bins > std::numeric_limits<int>::max())
      throw std::length_error("Linear impulse response is too large for the FFT path");
    // An L-point real transform, in multiply-equivalents: about (L / 2) log2 L.
    int log2_sub = 0;
    while ((1 << log2_sub) < tail.sub_size)
      ++log2_sub;
    tail.transform_cost = (long long)(tail.sub_size / 2) * log2_sub;
    tail.twiddle.resize(tail.fft_size);
    const double pi = 3.14159265358979323846;
    for (int k = 0; k < tail.fft_size; ++k)
    {
      const double angle = -2.0 * pi * k / tail.fft_size;
      tail.twiddle[k] = LinearFFTState::Complex((float)std::cos(angle), (float)std::sin(angle));
    }
    // Kernel spectra are computed off the audio thread, whole.
    Eigen::FFT<float> whole;
    whole.SetFlag(Eigen::FFT<float>::HalfSpectrum);
    tail.kernel_spectra.resize(kernels);
    std::vector<float> tail_time(tail.fft_size, 0.0f);
    for (int k = 0; k < kernels; ++k)
    {
      auto& spectra = tail.kernel_spectra[k];
      spectra.assign(tail.num_partitions, std::vector<LinearFFTState::Complex>(tail.num_bins));
      for (int partition = 0; partition < tail.num_partitions; ++partition)
      {
        std::fill(tail_time.begin(), tail_time.end(), 0.0f);
        const int start = uniform_end + partition * tail_size;
        const int partition_size = std::min(tail_size, this->_receptive_field - start);
        std::copy_n(this->_impulse_response[k].begin() + start, partition_size, tail_time.begin());
        whole.fwd(spectra[partition].data(), tail_time.data(), tail.fft_size);
      }
    }
    tail.inputs.resize(in_channels);
    for (auto& input : tail.inputs)
    {
      input.collecting.assign(tail_size, 0.0f);
      input.ready.assign(tail_size, 0.0f);
      input.spectra.assign(
        tail.num_partitions, std::vector<LinearFFTState::Complex>(tail.num_bins, LinearFFTState::Complex{}));
      input.sub_spectra.assign(tail.num_subs, std::vector<LinearFFTState::Complex>(tail.sub_bins));
    }
    // Output starts 2T after a block's first sample and lasts 2T - 1.
    tail.ring_size = 4 * tail_size;
    tail.outputs.resize(out_channels);
    for (auto& output : tail.outputs)
    {
      output.accumulator.assign(tail.num_bins, LinearFFTState::Complex{});
      output.sub_spectra.assign(tail.num_subs, std::vector<LinearFFTState::Complex>(tail.sub_bins));
      output.ring.assign(tail.ring_size, 0.0f);
    }
    tail.sub_time.assign(tail.sub_size, 0.0f);
    std::vector<LinearFFTState::Complex> warm_spectrum(tail.sub_bins);
    tail.fft.fwd(warm_spectrum.data(), tail.sub_time.data(), tail.sub_size);
    tail.fft.inv(tail.sub_time.data(), warm_spectrum.data(), tail.sub_size);
  }

  // Create the FFT plans outside the audio callback.
  if (state.num_partitions > 0)
  {
    std::vector<LinearFFTState::Complex> warm_spectrum(state.num_bins);
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

  const int paths = std::max(in_channels, out_channels);
  for (int ch = 0; ch < out_channels; ++ch)
    std::fill_n(output[ch], num_frames, this->_bias[ch]);
  for (int path = 0; path < paths; ++path)
  {
    const int input_channel = in_channels == 1 ? 0 : path;
    const int output_channel = out_channels == 1 ? 0 : path;
    const auto& weight = this->_weight[this->_kernel_index(path)];
    for (int i = 0; i < num_frames; ++i)
    {
      const long offset = this->_input_buffer_offset - this->_receptive_field + i + 1;
      auto input_vec =
        Eigen::Map<const Eigen::VectorXf>(&this->_input_buffers[input_channel][offset], this->_receptive_field);
      output[output_channel][i] += weight.dot(input_vec);
    }
  }

  // Prepare for next call:
  nam::Buffer::_advance_input_buffer_(num_frames);
}

void nam::Linear::_process_fft(NAM_SAMPLE** input, NAM_SAMPLE** output, const int num_frames)
{
  auto& state = *this->_fft_state;
  if (num_frames <= 0)
    return;
  const int direct_taps = state.direct_taps;

  // The whole callback's input goes into history before any output is
  // written, because output may alias input.
  for (int ch = 0; ch < NumInputChannels(); ++ch)
  {
    auto& channel = state.inputs[ch];
    if (channel.history_index + num_frames > channel.history.size())
    {
      const float* head = channel.history.data() + channel.history_index - direct_taps;
      if ((size_t)direct_taps + num_frames > channel.history.size())
      {
        // Only a callback larger than the maximum buffer size gets here.
        std::vector<float> grown((size_t)direct_taps + (size_t)_FFT_HISTORY_CALLBACKS * num_frames, 0.0f);
        std::copy_n(head, direct_taps, grown.begin());
        channel.history.swap(grown);
      }
      else
        std::memmove(channel.history.data(), head, (size_t)direct_taps * sizeof(float));
      channel.history_index = direct_taps;
    }
    for (int i = 0; i < num_frames; ++i)
      channel.history[channel.history_index + i] = (float)input[ch][i];
  }

  const bool has_tail = state.num_partitions > 0;
  bool transformed = false;
  for (int i = 0; i < num_frames; ++i)
  {
    for (int ch = 0; ch < NumOutputChannels(); ++ch)
    {
      NAM_SAMPLE value = this->_bias[ch];
      if (has_tail)
      {
        const auto& tail = state.outputs[ch];
        value += tail.ifft_previous[state.block_size + state.input_pos] + tail.ifft_current[state.input_pos];
      }
      if (state.has_tail_tier)
      {
        float& pending = state.tail.outputs[ch].ring[state.tail.ring_pos];
        value += pending;
        pending = 0.0f;
      }
      output[ch][i] = value;
    }
    for (int path = 0; path < state.num_paths; ++path)
    {
      const auto& channel = state.inputs[state.path_input[path]];
      const auto window =
        Eigen::Map<const Eigen::VectorXf>(channel.history.data() + channel.history_index + i + 1 - direct_taps, direct_taps);
      output[state.path_output[path]][i] += this->_fft_direct_weight[state.path_kernel[path]].dot(window);
    }
    if (has_tail && ++state.input_pos == state.block_size)
    {
      state.transform(i);
      transformed = true;
    }
    if (state.has_tail_tier)
      state.tail_advance(state.inputs.front().history_index + i);
  }

  for (auto& channel : state.inputs)
    channel.history_index += num_frames;

  // A callback that ran no transform takes its share of the pending
  // multiplies: what is left, divided among the callbacks still to come before
  // the one that runs the transform, this one included.
  //
  // With regular callbacks that leaves the transform nothing to catch up on;
  // irregular ones are made whole there. A callback of block_size frames or
  // more always runs a transform, and so does all of the work there.
  if (!transformed && state.pending_done < state.pending_units)
  {
    const int callbacks_left = (state.block_size - state.input_pos + num_frames - 1) / num_frames;
    const int remaining = state.pending_units - state.pending_done;
    state.do_pending((remaining + callbacks_left - 1) / callbacks_left);
  }

  // The tail's job likewise: what is left, shared among the callbacks still to
  // come before the next tail block completes, this one included.
  auto& tail = state.tail;
  if (state.has_tail_tier && tail.job_active)
  {
    const int samples_left = std::max(1, tail.block_size - tail.job_clock);
    const long long callbacks_left = (samples_left + num_frames - 1) / num_frames;
    state.tail_work((tail.remaining_cost + callbacks_left - 1) / callbacks_left);
  }
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
    {
      return entry.plan;
    }
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
