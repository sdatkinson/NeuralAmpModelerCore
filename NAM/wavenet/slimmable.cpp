#include "slimmable.h"
#include "../get_dsp.h"

#include <algorithm>
#include <cmath>
#include <optional>
#include <stdexcept>

namespace nam
{
namespace slimmable_wavenet
{

namespace
{

// ============================================================================
// Weight extraction helpers (groups=1 only)
// ============================================================================

// Extract Conv1x1 weight subset: take first slim_out rows, first slim_in cols.
// Full layout (groups=1): row-major out*in, then optional bias(out).
void extract_conv1x1(std::vector<float>::const_iterator& src, int full_in, int full_out, int slim_in, int slim_out,
                     bool bias, std::vector<float>& dst)
{
  for (int i = 0; i < full_out; i++)
  {
    for (int j = 0; j < full_in; j++)
    {
      float w = *(src++);
      if (i < slim_out && j < slim_in)
        dst.push_back(w);
    }
  }
  if (bias)
  {
    for (int i = 0; i < full_out; i++)
    {
      float b = *(src++);
      if (i < slim_out)
        dst.push_back(b);
    }
  }
}

// Extract Conv1D weight subset: take first slim_out output channels, first slim_in input channels.
// Full layout (groups=1): for each out i, for each in j, for each kernel tap k: weight. Then bias(out).
void extract_conv1d(std::vector<float>::const_iterator& src, int full_in, int full_out, int slim_in, int slim_out,
                    int kernel_size, std::vector<float>& dst)
{
  for (int i = 0; i < full_out; i++)
  {
    for (int j = 0; j < full_in; j++)
    {
      for (int k = 0; k < kernel_size; k++)
      {
        float w = *(src++);
        if (i < slim_out && j < slim_in)
          dst.push_back(w);
      }
    }
  }
  // Bias is always present for conv in WaveNet layers
  for (int i = 0; i < full_out; i++)
  {
    float b = *(src++);
    if (i < slim_out)
      dst.push_back(b);
  }
}

// Copy n weights unchanged
void copy_weights(std::vector<float>::const_iterator& src, int n, std::vector<float>& dst)
{
  for (int i = 0; i < n; i++)
    dst.push_back(*(src++));
}

// Compute slim bottleneck from original params and target channels
int compute_slim_bottleneck(const wavenet::LayerArrayParams& p, int new_channels)
{
  if (!p.layer1x1_params.active)
    return new_channels; // bottleneck must equal channels when layer1x1 inactive
  return std::max(1, p.bottleneck * new_channels / p.channels);
}

// Validate that all convolution groups are 1
void validate_groups(const wavenet::LayerArrayParams& p)
{
  if (p.groups_input != 1)
    throw std::runtime_error("SlimmableWavenet: groups_input > 1 not supported");
  if (p.groups_input_mixin != 1)
    throw std::runtime_error("SlimmableWavenet: groups_input_mixin > 1 not supported");
  if (p.layer1x1_params.active && p.layer1x1_params.groups != 1)
    throw std::runtime_error("SlimmableWavenet: layer1x1 groups > 1 not supported");
  if (p.head1x1_params.active && p.head1x1_params.groups != 1)
    throw std::runtime_error("SlimmableWavenet: head1x1 groups > 1 not supported");
}

// Map ratio [0,1] to a channel count from allowed_channels.
// Matches Python: idx = min(floor(ratio * len), len - 1)
int ratio_to_channels(double ratio, const std::vector<int>& allowed)
{
  int idx = std::min((int)std::floor(ratio * (double)allowed.size()), (int)allowed.size() - 1);
  return allowed[idx];
}

// ============================================================================
// Extract slimmed weights by walking the full weight vector in set_weights_
// order, using typed LayerArrayParams for dimensions.
// ============================================================================

std::vector<float> extract_slimmed_weights(const std::vector<wavenet::LayerArrayParams>& original_params,
                                           const std::vector<float>& full_weights,
                                           const std::vector<int>& new_channels_per_array)
{
  std::vector<float> slim;
  auto src = full_weights.cbegin();
  const int num_arrays = (int)original_params.size();

  for (int arr = 0; arr < num_arrays; arr++)
  {
    const auto& p = original_params[arr];
    if (p.head_kernel_size != 1)
    {
      throw std::runtime_error(
        "SlimmableWavenet: head rechannel kernel_size must be 1 (slimming with head kernel_size > 1 is not "
        "implemented)");
    }
    validate_groups(p);

    const int full_ch = p.channels;
    const int full_bn = p.bottleneck;
    const int num_layers = (int)p.dilations.size();
    const int slim_ch = new_channels_per_array[arr];

    const int slim_bn = compute_slim_bottleneck(p, slim_ch);

    // Input size: first array keeps original, others get previous array's target channels
    const int slim_input_size = (arr == 0) ? p.input_size : new_channels_per_array[arr - 1];
    // Head size: intermediate arrays must match next array's channels; last keeps original
    const int slim_head_size = (arr < num_arrays - 1) ? new_channels_per_array[arr + 1] : p.head_size;

    const int full_head_out = p.head1x1_params.active ? p.head1x1_params.out_channels : full_bn;
    const int slim_head_out = p.head1x1_params.active ? p.head1x1_params.out_channels : slim_bn;

    // ---- rechannel: Conv1x1(input_size -> channels, no bias) ----
    extract_conv1x1(src, p.input_size, full_ch, slim_input_size, slim_ch, false, slim);

    // ---- Per layer ----
    for (int l = 0; l < num_layers; l++)
    {
      const int kernel_size = p.kernel_sizes[l];
      const bool gated = p.gating_modes[l] != wavenet::GatingMode::NONE;
      const int full_bg = gated ? 2 * full_bn : full_bn;
      const int slim_bg = gated ? 2 * slim_bn : slim_bn;

      // conv: Conv1D(channels -> B_g, K, bias=true)
      extract_conv1d(src, full_ch, full_bg, slim_ch, slim_bg, kernel_size, slim);

      // input_mixin: Conv1x1(condition_size -> B_g, no bias)
      extract_conv1x1(src, p.condition_size, full_bg, p.condition_size, slim_bg, false, slim);

      // layer1x1 (optional): Conv1x1(B -> C, bias=true)
      if (p.layer1x1_params.active)
        extract_conv1x1(src, full_bn, full_ch, slim_bn, slim_ch, true, slim);

      // head1x1 (optional): Conv1x1(B -> head1x1_out, bias=true)
      if (p.head1x1_params.active)
        extract_conv1x1(
          src, full_bn, p.head1x1_params.out_channels, slim_bn, p.head1x1_params.out_channels, true, slim);

      // ---- FiLM objects (8, in set_weights_ order) ----

      // conv_pre_film: FiLM(condition_size -> channels)
      if (p.conv_pre_film_params.active)
      {
        int full_out = (p.conv_pre_film_params.shift ? 2 : 1) * full_ch;
        int slim_out = (p.conv_pre_film_params.shift ? 2 : 1) * slim_ch;
        extract_conv1x1(src, p.condition_size, full_out, p.condition_size, slim_out, true, slim);
      }

      // conv_post_film: FiLM(condition_size -> B_g)
      if (p.conv_post_film_params.active)
      {
        int full_out = (p.conv_post_film_params.shift ? 2 : 1) * full_bg;
        int slim_out = (p.conv_post_film_params.shift ? 2 : 1) * slim_bg;
        extract_conv1x1(src, p.condition_size, full_out, p.condition_size, slim_out, true, slim);
      }

      // input_mixin_pre_film: FiLM(condition_size -> condition_size) -- unchanged
      if (p.input_mixin_pre_film_params.active)
      {
        int dim = (p.input_mixin_pre_film_params.shift ? 2 : 1) * p.condition_size;
        copy_weights(src, p.condition_size * dim + dim, slim);
      }

      // input_mixin_post_film: FiLM(condition_size -> B_g)
      if (p.input_mixin_post_film_params.active)
      {
        int full_out = (p.input_mixin_post_film_params.shift ? 2 : 1) * full_bg;
        int slim_out = (p.input_mixin_post_film_params.shift ? 2 : 1) * slim_bg;
        extract_conv1x1(src, p.condition_size, full_out, p.condition_size, slim_out, true, slim);
      }

      // activation_pre_film: FiLM(condition_size -> B_g)
      if (p.activation_pre_film_params.active)
      {
        int full_out = (p.activation_pre_film_params.shift ? 2 : 1) * full_bg;
        int slim_out = (p.activation_pre_film_params.shift ? 2 : 1) * slim_bg;
        extract_conv1x1(src, p.condition_size, full_out, p.condition_size, slim_out, true, slim);
      }

      // activation_post_film: FiLM(condition_size -> B)
      if (p.activation_post_film_params.active)
      {
        int full_out = (p.activation_post_film_params.shift ? 2 : 1) * full_bn;
        int slim_out = (p.activation_post_film_params.shift ? 2 : 1) * slim_bn;
        extract_conv1x1(src, p.condition_size, full_out, p.condition_size, slim_out, true, slim);
      }

      // layer1x1_post_film: FiLM(condition_size -> C)
      if (p._layer1x1_post_film_params.active && p.layer1x1_params.active)
      {
        int full_out = (p._layer1x1_post_film_params.shift ? 2 : 1) * full_ch;
        int slim_out = (p._layer1x1_post_film_params.shift ? 2 : 1) * slim_ch;
        extract_conv1x1(src, p.condition_size, full_out, p.condition_size, slim_out, true, slim);
      }

      // head1x1_post_film: FiLM(condition_size -> head1x1_out) -- unchanged
      if (p.head1x1_post_film_params.active && p.head1x1_params.active)
      {
        int dim = (p.head1x1_post_film_params.shift ? 2 : 1) * p.head1x1_params.out_channels;
        copy_weights(src, p.condition_size * dim + dim, slim);
      }
    }

    // ---- head_rechannel: Conv1x1(head_output_size -> head_size, bias=head_bias) ----
    extract_conv1x1(src, full_head_out, p.head_size, slim_head_out, slim_head_size, p.head_bias, slim);
  }

  // head_scale: 1 float, copy as-is
  slim.push_back(*(src++));

  return slim;
}

// ============================================================================
// Build modified LayerArrayParams with per-array channel counts
// ============================================================================

std::vector<wavenet::LayerArrayParams> modify_params_for_channels(
  const std::vector<wavenet::LayerArrayParams>& original_params, const std::vector<int>& new_channels_per_array)
{
  std::vector<wavenet::LayerArrayParams> modified;
  const int num_arrays = (int)original_params.size();

  for (int i = 0; i < num_arrays; i++)
  {
    const auto& p = original_params[i];
    const int new_ch = new_channels_per_array[i];

    int new_bottleneck = compute_slim_bottleneck(p, new_ch);
    int new_input_size = (i == 0) ? p.input_size : new_channels_per_array[i - 1];
    int new_head_size = (i < num_arrays - 1) ? new_channels_per_array[i + 1] : p.head_size;

    modified.push_back(wavenet::LayerArrayParams(
      new_input_size, p.condition_size, new_head_size, p.head_kernel_size, new_ch, new_bottleneck,
      std::vector<int>(p.kernel_sizes), std::vector<int>(p.dilations),
      std::vector<activations::ActivationConfig>(p.activation_configs),
      std::vector<wavenet::GatingMode>(p.gating_modes), p.head_bias, p.groups_input, p.groups_input_mixin,
      p.layer1x1_params, p.head1x1_params, std::vector<activations::ActivationConfig>(p.secondary_activation_configs),
      p.conv_pre_film_params, p.conv_post_film_params, p.input_mixin_pre_film_params, p.input_mixin_post_film_params,
      p.activation_pre_film_params, p.activation_post_film_params, p._layer1x1_post_film_params,
      p.head1x1_post_film_params));
  }

  return modified;
}

// Check if all per-array channels match full (no slimming needed)
bool is_full_size(const std::vector<wavenet::LayerArrayParams>& params, const std::vector<int>& channels)
{
  for (size_t i = 0; i < params.size(); i++)
  {
    if (channels[i] != params[i].channels)
      return false;
  }
  return true;
}

} // anonymous namespace

#ifdef _LIBCPP_VERSION
void SlimmableWavenet::_pending_clear_release()
{
  std::atomic_store_explicit(&_pending_staged, std::shared_ptr<StagedSlimModel>{}, std::memory_order_release);
}

std::shared_ptr<SlimmableWavenet::StagedSlimModel> SlimmableWavenet::_pending_load_acquire() const
{
  return std::atomic_load_explicit(&_pending_staged, std::memory_order_acquire);
}

void SlimmableWavenet::_pending_store_release(std::shared_ptr<StagedSlimModel> p)
{
  std::atomic_store_explicit(&_pending_staged, std::move(p), std::memory_order_release);
}

std::shared_ptr<SlimmableWavenet::StagedSlimModel> SlimmableWavenet::_pending_exchange_take_acq_rel()
{
  return std::atomic_exchange_explicit(&_pending_staged, std::shared_ptr<StagedSlimModel>{}, std::memory_order_acq_rel);
}
#else
void SlimmableWavenet::_pending_clear_release()
{
  _pending_staged.store({}, std::memory_order_release);
}

std::shared_ptr<SlimmableWavenet::StagedSlimModel> SlimmableWavenet::_pending_load_acquire() const
{
  return _pending_staged.load(std::memory_order_acquire);
}

void SlimmableWavenet::_pending_store_release(std::shared_ptr<StagedSlimModel> p)
{
  _pending_staged.store(std::move(p), std::memory_order_release);
}

std::shared_ptr<SlimmableWavenet::StagedSlimModel> SlimmableWavenet::_pending_exchange_take_acq_rel()
{
  return _pending_staged.exchange({}, std::memory_order_acq_rel);
}
#endif

// ============================================================================
// SlimmableWavenet
// ============================================================================

SlimmableWavenet::SlimmableWavenet(std::vector<wavenet::LayerArrayParams> original_params,
                                   std::vector<std::vector<int>> per_array_allowed_channels, int in_channels,
                                   float head_scale, bool with_head, nlohmann::json condition_dsp_json,
                                   std::vector<float> full_weights, double expected_sample_rate)
: DSP(in_channels, original_params.back().head_size, expected_sample_rate)
, _original_params(std::move(original_params))
, _per_array_allowed_channels(std::move(per_array_allowed_channels))
, _in_channels(in_channels)
, _head_scale(head_scale)
, _with_head(with_head)
, _condition_dsp_json(std::move(condition_dsp_json))
, _full_weights(std::move(full_weights))
{
  if (_per_array_allowed_channels.size() != _original_params.size())
    throw std::runtime_error("SlimmableWavenet: per_array_allowed_channels size must match number of layer arrays");

  // Validate: at least one array must be slimmable
  bool any_slimmable = false;
  for (size_t i = 0; i < _per_array_allowed_channels.size(); i++)
  {
    const auto& allowed = _per_array_allowed_channels[i];
    if (!allowed.empty())
    {
      any_slimmable = true;
      // Validate sorted
      for (size_t j = 1; j < allowed.size(); j++)
      {
        if (allowed[j] <= allowed[j - 1])
          throw std::runtime_error("SlimmableWavenet: allowed_channels must be sorted ascending");
      }
      // Validate last entry matches full channel count
      if (allowed.back() != _original_params[i].channels)
        throw std::runtime_error(
          "SlimmableWavenet: last allowed_channels entry must equal the full channel count for that array");
    }
  }
  if (!any_slimmable)
    throw std::runtime_error("SlimmableWavenet: at least one layer array must have allowed_channels");

  if (with_head)
    throw std::runtime_error("SlimmableWavenet: post-stack head is not supported");

  // Build with full channel counts as default (ratio=1.0)
  std::vector<int> full_channels(_original_params.size());
  for (size_t i = 0; i < _original_params.size(); i++)
    full_channels[i] = _original_params[i].channels;
  _rebuild_model(full_channels);
}

std::unique_ptr<DSP> SlimmableWavenet::_create_wavenet_for_channels(const std::vector<int>& target_channels)
{
  std::vector<float> weights;
  std::vector<wavenet::LayerArrayParams> modified_params;
  const std::vector<wavenet::LayerArrayParams>* params_ptr;

  if (is_full_size(_original_params, target_channels))
  {
    weights = _full_weights;
    params_ptr = &_original_params;
  }
  else
  {
    weights = extract_slimmed_weights(_original_params, _full_weights, target_channels);
    modified_params = modify_params_for_channels(_original_params, target_channels);
    params_ptr = &modified_params;
  }

  // Rebuild condition_dsp if present (WaveNet takes ownership each time)
  std::unique_ptr<DSP> condition_dsp;
  if (!_condition_dsp_json.is_null())
    condition_dsp = get_dsp(_condition_dsp_json);

  double sampleRate = _current_sample_rate > 0 ? _current_sample_rate : GetExpectedSampleRate();
  return std::make_unique<wavenet::WaveNet>(_in_channels, *params_ptr, _head_scale, _with_head, std::nullopt,
                                            std::move(weights), std::move(condition_dsp), sampleRate);
}

void SlimmableWavenet::_rebuild_model(const std::vector<int>& target_channels)
{
  if (target_channels == _current_channels && _active_model)
    return;

  _pending_clear_release();

  _active_model = std::shared_ptr<DSP>(_create_wavenet_for_channels(target_channels));
  _current_channels = target_channels;

  if (_current_buffer_size > 0)
    _active_model->Reset(_current_sample_rate, _current_buffer_size);
}

void SlimmableWavenet::_stage_rebuild_model(const std::vector<int>& target_channels)
{
  if (target_channels == _current_channels && _active_model)
  {
    _pending_clear_release();
    return;
  }

  if (auto pending = _pending_load_acquire())
  {
    if (pending->channels == target_channels)
      return;
  }

  auto pack = std::make_shared<StagedSlimModel>();
  pack->model = std::shared_ptr<DSP>(_create_wavenet_for_channels(target_channels));
  pack->channels = target_channels;

  if (_current_buffer_size > 0)
    pack->model->Reset(_current_sample_rate, _current_buffer_size);

  _pending_store_release(std::move(pack));
}

void SlimmableWavenet::process(NAM_SAMPLE** input, NAM_SAMPLE** output, const int num_frames)
{
  if (auto pack = _pending_exchange_take_acq_rel())
  {
    _active_model = std::move(pack->model);
    _current_channels = std::move(pack->channels);
  }
  if (_active_model)
    _active_model->process(input, output, num_frames);
}

void SlimmableWavenet::prewarm()
{
  if (_active_model)
    _active_model->prewarm();
  if (auto pending = _pending_load_acquire())
    pending->model->prewarm();
}

void SlimmableWavenet::Reset(const double sampleRate, const int maxBufferSize)
{
  _current_sample_rate = sampleRate;
  _current_buffer_size = maxBufferSize;
  if (_active_model)
    _active_model->Reset(sampleRate, maxBufferSize);
  if (auto pending = _pending_load_acquire())
    pending->model->Reset(sampleRate, maxBufferSize);
}

void SlimmableWavenet::SetSlimmableSize(const double val)
{
  const size_t num_arrays = _original_params.size();
  std::vector<int> target(num_arrays);

  for (size_t i = 0; i < num_arrays; i++)
  {
    const auto& allowed = _per_array_allowed_channels[i];
    if (allowed.empty())
      target[i] = _original_params[i].channels; // Non-slimmable: keep full
    else
      target[i] = ratio_to_channels(val, allowed);
  }

  _stage_rebuild_model(target);
}

// ============================================================================
// Config / factory / registration
// ============================================================================

std::unique_ptr<DSP> SlimmableWavenetConfig::create(std::vector<float> weights, double sampleRate)
{
  // Parse the WaveNet model config — support both wrapped {"model": {...}} and flat config
  nlohmann::json model_json = raw_config.contains("model") ? raw_config["model"] : raw_config;
  auto wc = wavenet::parse_config_json(model_json, sampleRate);

  // Extract per-array allowed_channels from slimmable config fields
  const auto& layers_json = model_json["layers"];
  std::vector<std::vector<int>> per_array_allowed;
  for (size_t i = 0; i < layers_json.size(); i++)
  {
    const auto& lc = layers_json[i];
    std::vector<int> allowed;
    if (lc.find("slimmable") != lc.end() && lc["slimmable"].is_object())
    {
      const auto& slim_cfg = lc["slimmable"];
      const std::string method = slim_cfg.value("method", "");
      if (method != "slice_channels_uniform")
        throw std::runtime_error("SlimmableWavenet: unsupported slimmable method '" + method + "'");
      if (slim_cfg.find("kwargs") != slim_cfg.end()
          && slim_cfg["kwargs"].find("allowed_channels") != slim_cfg["kwargs"].end())
      {
        for (const auto& ch : slim_cfg["kwargs"]["allowed_channels"])
          allowed.push_back(ch.get<int>());
      }
      else
      {
        // Missing allowed_channels: assume [1, 2, ..., channels] for slice_channels_uniform
        const int channels = lc["channels"].get<int>();
        for (int c = 1; c <= channels; c++)
          allowed.push_back(c);
      }
    }
    per_array_allowed.push_back(std::move(allowed));
  }

  // Extract condition_dsp JSON for future rebuilds (in model config)
  nlohmann::json condition_dsp_json;
  if (model_json.find("condition_dsp") != model_json.end() && !model_json["condition_dsp"].is_null())
    condition_dsp_json = model_json["condition_dsp"];

  return std::make_unique<SlimmableWavenet>(std::move(wc.layer_array_params), std::move(per_array_allowed),
                                            wc.in_channels, wc.head_scale, wc.with_head, std::move(condition_dsp_json),
                                            std::move(weights), sampleRate);
}

std::unique_ptr<ModelConfig> create_config(const nlohmann::json& config, double sampleRate)
{
  auto sc = std::make_unique<SlimmableWavenetConfig>();
  sc->raw_config = config;
  sc->sample_rate = sampleRate;
  return sc;
}

} // namespace slimmable_wavenet
} // namespace nam
