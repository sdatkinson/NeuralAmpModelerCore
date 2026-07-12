#if defined(NAM_ENABLE_FUSED)

  #include "fused.h"

  #include <algorithm>
  #include <cassert>
  #include <cmath>
  #include <cstddef>
  #include <cstring>
  #include <memory>
  #include <stdexcept>
  #include <string>
  #include <utility>
  #include <vector>

  #include "../activations.h"
  #include "../dsp.h"

  #if defined(__aarch64__) || defined(_M_ARM64)
    #define NAM_FUSED_KERNELS 1
    #include <arm_neon.h>
  #endif

namespace nam
{
namespace wavenet
{
namespace fused
{

namespace
{

// =============================================================================
// Shape spec: everything the engine needs, parsed once from the .nam JSON.
// =============================================================================

struct LayerSpec
{
  int kernel_size = 0;
  int dilation = 0;
  activations::ActivationConfig activation{};
};

struct ArraySpec
{
  int input_size = 0;
  int channels = 0;
  int head_size = 0;
  int head_kernel_size = 1;
  bool head_bias = false;
  bool layer1x1_active = true;
  std::vector<LayerSpec> layers;
};

struct ShapeSpec
{
  std::vector<ArraySpec> arrays;
};

constexpr int kMaxChannels = 32;
// Cap on conv/head kernel sizes (bounds the per-block tap pointer table).
constexpr int kMaxKernelSize = 64;

bool film_inactive(const nlohmann::json& layer, const char* key)
{
  auto it = layer.find(key);
  if (it == layer.end() || it->is_null())
    return true;
  if (it->is_boolean())
    return !it->get<bool>();
  if (it->is_object())
    return !it->value("active", false);
  return false;
}

bool gating_all_none(const nlohmann::json& la, size_t num_layers)
{
  auto gm_it = la.find("gating_mode");
  if (gm_it != la.end() && !gm_it->is_null())
  {
    if (gm_it->is_string())
      return gm_it->get<std::string>() == "none";
    if (!gm_it->is_array() || gm_it->size() != num_layers)
      return false;
    for (const auto& e : *gm_it)
    {
      if (!e.is_string() || e.get<std::string>() != "none")
        return false;
    }
    return true;
  }
  auto g_it = la.find("gated");
  if (g_it != la.end() && !g_it->is_null())
    return g_it->is_boolean() && !g_it->get<bool>();
  return true; // absent defaults to NONE
}

/// Strict parser. Returns false (never throws) if the config doesn't match a
/// supported shape. On success fills *spec when non-null.
bool parse_spec(const nlohmann::json& config, ShapeSpec* spec) noexcept
{
  try
  {
    // Mono in, no condition DSP, no post-stack head.
    if (config.value("in_channels", 1) != 1)
      return false;
    if (auto it = config.find("condition_dsp"); it != config.end() && !it->is_null())
      return false;
    if (auto it = config.find("head"); it != config.end() && !it->is_null())
      return false;
    if (auto it = config.find("head_scale"); it == config.end() || !it->is_number())
      return false;

    auto layers_it = config.find("layers");
    if (layers_it == config.end() || !layers_it->is_array() || layers_it->empty())
      return false;

    ShapeSpec local;
    int prev_channels = -1;
    int prev_head_size = -1;
    for (const auto& la : *layers_it)
    {
      ArraySpec as;

      as.input_size = la.value("input_size", 0);
      if (local.arrays.empty())
      {
        if (as.input_size != 1)
          return false;
      }
      else if (as.input_size != prev_channels)
        return false;

      if (la.value("condition_size", 0) != 1)
        return false;

      as.channels = la.value("channels", 0);
      const int bottleneck = la.value("bottleneck", as.channels);
      if (as.channels <= 0 || bottleneck != as.channels)
        return false;
      if (as.channels % 4 != 0 || as.channels > kMaxChannels)
        return false;
      if (!local.arrays.empty() && as.channels != prev_head_size)
        return false;

      if (la.value("groups_input", 1) != 1 || la.value("groups_input_mixin", 1) != 1)
        return false;

      // layer1x1: default active with groups=1
      if (auto it = la.find("layer1x1"); it != la.end() && !it->is_null())
      {
        if (!it->is_object())
          return false;
        as.layer1x1_active = it->value("active", false);
        if (as.layer1x1_active && it->value("groups", 1) != 1)
          return false;
        // Generic path requires bottleneck == channels when layer1x1 inactive;
        // that already holds here.
      }

      // head1x1 must be inactive
      if (auto it = la.find("head1x1"); it != la.end() && it->is_object() && it->value("active", false))
        return false;

      // No FiLM anywhere
      for (const char* key : {"conv_pre_film", "conv_post_film", "input_mixin_pre_film", "input_mixin_post_film",
                              "activation_pre_film", "activation_post_film", "layer1x1_post_film", "head1x1_post_film"})
      {
        if (!film_inactive(la, key))
          return false;
      }

      // Not slimmable
      if (auto it = la.find("slimmable"); it != la.end() && !it->is_null())
        return false;

      // Dilations & kernel sizes
      auto dil_it = la.find("dilations");
      if (dil_it == la.end() || !dil_it->is_array() || dil_it->empty())
        return false;
      const size_t num_layers = dil_it->size();

      std::vector<int> kernel_sizes;
      const bool has_ks = la.find("kernel_size") != la.end();
      const bool has_kss = la.find("kernel_sizes") != la.end();
      if (has_ks == has_kss) // exactly one must be present
        return false;
      if (has_ks)
      {
        if (!la["kernel_size"].is_number_integer())
          return false;
        kernel_sizes.assign(num_layers, la["kernel_size"].get<int>());
      }
      else
      {
        if (!la["kernel_sizes"].is_array() || la["kernel_sizes"].size() != num_layers)
          return false;
        for (const auto& k : la["kernel_sizes"])
        {
          if (!k.is_number_integer())
            return false;
          kernel_sizes.push_back(k.get<int>());
        }
      }

      // Activations (single or per-layer). Any config that
      // ActivationConfig::from_json accepts is fine: the engine falls back to
      // the generic Activation object for types without NEON kernels.
      std::vector<activations::ActivationConfig> act_configs;
      const auto& act_json = la.at("activation");
      if (act_json.is_array())
      {
        if (act_json.size() != num_layers)
          return false;
        for (const auto& aj : act_json)
          act_configs.push_back(activations::ActivationConfig::from_json(aj));
      }
      else
      {
        act_configs.assign(num_layers, activations::ActivationConfig::from_json(act_json));
      }

      if (!gating_all_none(la, num_layers))
        return false;

      // Head rechannel: nested "head" object or legacy head_size/head_bias.
      if (auto it = la.find("head"); it != la.end() && !it->is_null())
      {
        if (!it->is_object())
          return false;
        as.head_size = it->value("out_channels", 0);
        as.head_kernel_size = it->value("kernel_size", 0);
        as.head_bias = it->value("bias", false);
      }
      else if (la.find("head_size") != la.end())
      {
        as.head_size = la["head_size"].get<int>();
        as.head_kernel_size = 1;
        as.head_bias = la.at("head_bias").get<bool>();
      }
      else
        return false;
      if (as.head_size <= 0 || as.head_kernel_size < 1 || as.head_kernel_size > kMaxKernelSize)
        return false;

      for (size_t i = 0; i < num_layers; i++)
      {
        LayerSpec ls;
        ls.kernel_size = kernel_sizes[i];
        ls.dilation = (*dil_it)[i].get<int>();
        ls.activation = act_configs[i];
        if (ls.kernel_size < 1 || ls.kernel_size > kMaxKernelSize || ls.dilation < 1)
          return false;
        as.layers.push_back(std::move(ls));
      }

      prev_channels = as.channels;
      prev_head_size = as.head_size;
      local.arrays.push_back(std::move(as));
    }

    if (spec)
      *spec = std::move(local);
    return true;
  }
  catch (const std::exception&)
  {
    return false;
  }
}

} // namespace

  #if defined(NAM_FUSED_KERNELS)

namespace
{

// =============================================================================
// NEON helpers
// =============================================================================

/// Vectorized version of activations::fast_tanh (same rational approximation).
/// Uses vdivq_f32: on Apple M-series the FP divider is a separate unit, so the
/// division overlaps with the FMA work and measures faster than a
/// reciprocal-refinement sequence (which competes with the FMAs) — and it's
/// exact, matching the scalar implementation up to FMA contraction.
inline float32x4_t vfast_tanh_f32(float32x4_t x)
{
  const float32x4_t a = vdupq_n_f32(2.45550750702956f);
  const float32x4_t b = vdupq_n_f32(0.893229853513558f);
  const float32x4_t c = vdupq_n_f32(0.821226666969744f);
  const float32x4_t d = vdupq_n_f32(2.44506634652299f);
  const float32x4_t e = vdupq_n_f32(0.814642734961073f);
  const float32x4_t ax = vabsq_f32(x);
  const float32x4_t x2 = vmulq_f32(x, x);
  const float32x4_t num = vmulq_f32(x, vaddq_f32(vfmaq_f32(a, a, ax), vmulq_f32(vfmaq_f32(b, c, ax), x2)));
  const float32x4_t den = vfmaq_f32(d, vaddq_f32(d, x2), vabsq_f32(vfmaq_f32(x, e, vmulq_f32(x, ax))));
  return vdivq_f32(num, den);
}

// =============================================================================
// Activation dispatch
// =============================================================================

enum class ActKind
{
  FastTanh,
  ReLU,
  LeakyReLU,
  Hardtanh,
  Softsign,
  Other // exact Tanh, Sigmoid, PReLU, LUT, ... -> Activation::apply()
};

struct ResolvedActivation
{
  ActKind kind = ActKind::Other;
  float leaky_slope = 0.01f;
  activations::Activation::Ptr ptr; // always valid; used for ActKind::Other
};

/// Resolve the activation exactly the way the generic Layer would (through the
/// registry, so enable_fast_tanh()/LUT substitutions are honored), then pick a
/// NEON kernel only when the resolved object is a known implementation.
ResolvedActivation resolve_activation(const activations::ActivationConfig& config)
{
  ResolvedActivation out;
  out.ptr = activations::Activation::get_activation(config);
  if (out.ptr == nullptr)
    throw std::runtime_error("FusedWaveNet: unsupported activation");

  using activations::ActivationType;
  switch (config.type)
  {
    case ActivationType::Tanh:
    case ActivationType::Fasttanh:
      // "Tanh" resolves to the fast-tanh singleton when enable_fast_tanh() is
      // active, or to a LUT if one was installed. Only claim the NEON kernel
      // when the resolved object really is the fast-tanh implementation.
      if (dynamic_cast<activations::ActivationFastTanh*>(out.ptr.get()) != nullptr)
        out.kind = ActKind::FastTanh;
      break;
    case ActivationType::ReLU: out.kind = ActKind::ReLU; break;
    case ActivationType::LeakyReLU:
      out.kind = ActKind::LeakyReLU;
      out.leaky_slope = config.negative_slope.value_or(0.01f);
      break;
    case ActivationType::Hardtanh: out.kind = ActKind::Hardtanh; break;
    case ActivationType::Softsign: out.kind = ActKind::Softsign; break;
    default: break;
  }
  return out;
}

/// Apply activation to a contiguous buffer. n is always a multiple of 4
/// (channels are multiples of 4).
void apply_activation(const ResolvedActivation& act, float* p, int n)
{
  assert(n % 4 == 0);
  switch (act.kind)
  {
    case ActKind::FastTanh:
      for (int i = 0; i < n; i += 4)
        vst1q_f32(p + i, vfast_tanh_f32(vld1q_f32(p + i)));
      break;
    case ActKind::ReLU:
    {
      const float32x4_t zero = vdupq_n_f32(0.0f);
      for (int i = 0; i < n; i += 4)
        vst1q_f32(p + i, vmaxq_f32(vld1q_f32(p + i), zero));
      break;
    }
    case ActKind::LeakyReLU:
    {
      const float32x4_t zero = vdupq_n_f32(0.0f);
      const float32x4_t slope = vdupq_n_f32(act.leaky_slope);
      for (int i = 0; i < n; i += 4)
      {
        const float32x4_t x = vld1q_f32(p + i);
        vst1q_f32(p + i, vbslq_f32(vcltq_f32(x, zero), vmulq_f32(x, slope), x));
      }
      break;
    }
    case ActKind::Hardtanh:
    {
      const float32x4_t lo = vdupq_n_f32(-1.0f);
      const float32x4_t hi = vdupq_n_f32(1.0f);
      for (int i = 0; i < n; i += 4)
        vst1q_f32(p + i, vminq_f32(vmaxq_f32(vld1q_f32(p + i), lo), hi));
      break;
    }
    case ActKind::Softsign:
    {
      const float32x4_t one = vdupq_n_f32(1.0f);
      for (int i = 0; i < n; i += 4)
      {
        const float32x4_t x = vld1q_f32(p + i);
        vst1q_f32(p + i, vdivq_f32(x, vaddq_f32(one, vabsq_f32(x))));
      }
      break;
    }
    case ActKind::Other: act.ptr->apply(p, n); break;
  }
}

// =============================================================================
// Ring buffer: power-of-2 capacity with a mirrored tail so any read of up to
// max_buffer_size frames starting inside [0, pow2) is one contiguous span.
// Same scheme as a2_fast ring mode 1 (constant per-block work, no rewind
// spikes on the audio thread).
// =============================================================================

int next_pow2(int v)
{
  int p = 1;
  while (p < v)
    p <<= 1;
  return p;
}

struct Ring
{
  std::vector<float> data;
  int channels = 0;
  int pow2 = 0;
  int mask = 0;
  int wpos = 0;
  int mbs = 0;

  void reset(int channels_, int max_lookback, int max_buffer)
  {
    channels = channels_;
    mbs = max_buffer;
    pow2 = next_pow2(max_lookback + max_buffer);
    mask = pow2 - 1;
    data.assign(static_cast<size_t>(channels) * (pow2 + max_buffer), 0.0f);
    wpos = max_lookback & mask;
  }

  void write(const float* src, int n)
  {
    float* h = data.data();
    const int first = std::min(n, pow2 - wpos);
    std::memcpy(h + static_cast<size_t>(wpos) * channels, src, static_cast<size_t>(first) * channels * sizeof(float));
    if (first < n)
    {
      std::memcpy(h, src + static_cast<size_t>(first) * channels,
                  static_cast<size_t>(n - first) * channels * sizeof(float));
    }
    wpos = (wpos + n) & mask;
  }

  /// Refresh the mirrored tail so that reads of up to `cols` frames past the
  /// end of the ring see cols [0, cols). Called by tap() only for the blocks
  /// where a read actually wraps, which keeps the common case copy-free.
  void mirror(int cols)
  {
    float* h = data.data();
    std::memcpy(h + static_cast<size_t>(pow2) * channels, h, static_cast<size_t>(cols) * channels * sizeof(float));
  }

  /// Pointer to the first frame of an n-frame read looking back
  /// `lookback_frames` from the start of the block just written. Ensures the
  /// mirrored tail covers the read when it wraps past the end of the ring.
  const float* tap(int lookback_frames, int n)
  {
    const int base = (wpos - n - lookback_frames) & mask;
    const int overflow = base + n - pow2;
    if (overflow > 0)
      mirror(overflow);
    return data.data() + static_cast<size_t>(base) * channels;
  }
};

// =============================================================================
// Register-tiled NEON kernels, templated on Q = channels / 4.
//
// Layouts (all column-major, channels fastest):
//   conv weights: per tap k, (C out x C in): w[k*C*C + j*C + i]
//   1x1 weights:  (C out x C in): w[j*C + i]
//   buffers:      (C x frames): buf[f*C + c]
// =============================================================================

/// One tile of the dilated conv: acc = bias; for each tap, acc += W_k * x_k;
/// then acc += mixin * cond. T frames per tile, accumulators live in registers
/// across all taps.
template <int Q, int T, int LANE>
inline void conv_lane(float32x4_t (&acc)[Q][T], const float32x4_t (&iv)[T], const float* wc)
{
  constexpr int C = 4 * Q;
  float32x4_t w[Q];
  for (int q = 0; q < Q; q++)
    w[q] = vld1q_f32(wc + LANE * C + 4 * q);
  for (int t = 0; t < T; t++)
    for (int q = 0; q < Q; q++)
      acc[q][t] = vfmaq_laneq_f32(acc[q][t], w[q], iv[t], LANE);
}

template <int Q, int T>
inline void conv_tile(const float* const* tap_src, int num_taps, const float* W, const float* bias,
                      const float* mixin, const float* cond, float* z, int f0)
{
  constexpr int C = 4 * Q;
  float32x4_t acc[Q][T];
  for (int q = 0; q < Q; q++)
  {
    const float32x4_t b = vld1q_f32(bias + 4 * q);
    for (int t = 0; t < T; t++)
      acc[q][t] = b;
  }
  for (int k = 0; k < num_taps; k++)
  {
    const float* src = tap_src[k] + static_cast<size_t>(f0) * C;
    const float* wk = W + static_cast<size_t>(k) * C * C;
    for (int j4 = 0; j4 < Q; j4++)
    {
      float32x4_t iv[T];
      for (int t = 0; t < T; t++)
        iv[t] = vld1q_f32(src + t * C + j4 * 4);
      const float* wc = wk + static_cast<size_t>(j4 * 4) * C;
      conv_lane<Q, T, 0>(acc, iv, wc);
      conv_lane<Q, T, 1>(acc, iv, wc);
      conv_lane<Q, T, 2>(acc, iv, wc);
      conv_lane<Q, T, 3>(acc, iv, wc);
    }
  }
  for (int t = 0; t < T; t++)
  {
    const float32x4_t cf = vdupq_n_f32(cond[f0 + t]);
    float* zc = z + static_cast<size_t>(f0 + t) * C;
    for (int q = 0; q < Q; q++)
      vst1q_f32(zc + 4 * q, vfmaq_f32(acc[q][t], vld1q_f32(mixin + 4 * q), cf));
  }
}

template <int Q>
void conv_block(const float* const* tap_src, int num_taps, const float* W, const float* bias, const float* mixin,
                const float* cond, float* z, int num_frames)
{
  constexpr int T = (Q <= 4) ? 4 : 2;
  int f = 0;
  for (; f + T <= num_frames; f += T)
    conv_tile<Q, T>(tap_src, num_taps, W, bias, mixin, cond, z, f);
  for (; f < num_frames; f++)
    conv_tile<Q, 1>(tap_src, num_taps, W, bias, mixin, cond, z, f);
}

/// Post-activation tail: head_sum += a; if layer1x1 active,
/// lin += L * a + l1x1_bias (in-place residual for the next layer).
template <int Q, int T, int LANE>
inline void tail_lane(float32x4_t (&out)[Q][T], const float32x4_t (&a)[Q][T], const float* lc, int j4)
{
  constexpr int C = 4 * Q;
  float32x4_t w[Q];
  for (int q = 0; q < Q; q++)
    w[q] = vld1q_f32(lc + LANE * C + 4 * q);
  for (int t = 0; t < T; t++)
    for (int q = 0; q < Q; q++)
      out[q][t] = vfmaq_laneq_f32(out[q][t], w[q], a[j4][t], LANE);
}

template <int Q, int T>
inline void tail_tile(const float* z, const float* L, const float* lb, bool has_l1x1, float* head_sum, float* lin,
                      int f0)
{
  constexpr int C = 4 * Q;
  float32x4_t a[Q][T];
  for (int t = 0; t < T; t++)
  {
    const float* zc = z + static_cast<size_t>(f0 + t) * C;
    for (int q = 0; q < Q; q++)
      a[q][t] = vld1q_f32(zc + 4 * q);
  }
  for (int t = 0; t < T; t++)
  {
    float* hc = head_sum + static_cast<size_t>(f0 + t) * C;
    for (int q = 0; q < Q; q++)
      vst1q_f32(hc + 4 * q, vaddq_f32(vld1q_f32(hc + 4 * q), a[q][t]));
  }
  if (!has_l1x1)
    return;
  float32x4_t out[Q][T];
  for (int q = 0; q < Q; q++)
  {
    const float32x4_t b = vld1q_f32(lb + 4 * q);
    for (int t = 0; t < T; t++)
      out[q][t] = b;
  }
  for (int j4 = 0; j4 < Q; j4++)
  {
    const float* lc = L + static_cast<size_t>(j4 * 4) * C;
    tail_lane<Q, T, 0>(out, a, lc, j4);
    tail_lane<Q, T, 1>(out, a, lc, j4);
    tail_lane<Q, T, 2>(out, a, lc, j4);
    tail_lane<Q, T, 3>(out, a, lc, j4);
  }
  for (int t = 0; t < T; t++)
  {
    float* lc = lin + static_cast<size_t>(f0 + t) * C;
    for (int q = 0; q < Q; q++)
      vst1q_f32(lc + 4 * q, vaddq_f32(vld1q_f32(lc + 4 * q), out[q][t]));
  }
}

template <int Q>
void tail_block(const float* z, const float* L, const float* lb, bool has_l1x1, float* head_sum, float* lin,
                int num_frames)
{
  constexpr int T = (Q <= 4) ? 2 : 1;
  int f = 0;
  for (; f + T <= num_frames; f += T)
    tail_tile<Q, T>(z, L, lb, has_l1x1, head_sum, lin, f);
  for (; f < num_frames; f++)
    tail_tile<Q, 1>(z, L, lb, has_l1x1, head_sum, lin, f);
}

/// Rechannel (1x1, no bias): out(C x N) = W(C x Cin) * x(Cin x N).
/// Cin is a runtime value (1 for the first array, previous channels after).
template <int Q>
void rechannel_block(const float* W, const float* x, int c_in, float* out, int num_frames)
{
  constexpr int C = 4 * Q;
  for (int f = 0; f < num_frames; f++)
  {
    const float* xc = x + static_cast<size_t>(f) * c_in;
    float32x4_t acc[Q];
    for (int q = 0; q < Q; q++)
      acc[q] = vdupq_n_f32(0.0f);
    for (int j = 0; j < c_in; j++)
    {
      const float32x4_t s = vdupq_n_f32(xc[j]);
      const float* wc = W + static_cast<size_t>(j) * C;
      for (int q = 0; q < Q; q++)
        acc[q] = vfmaq_f32(acc[q], vld1q_f32(wc + 4 * q), s);
    }
    float* oc = out + static_cast<size_t>(f) * C;
    for (int q = 0; q < Q; q++)
      vst1q_f32(oc + 4 * q, acc[q]);
  }
}

/// Head rechannel conv: out(H x N) = sum_k W_k(H x C) * x_k + bias.
/// Weights row-major per tap: w[k*H*C + h*C + j]. H is small (often 8 or 1).
void head_conv_block(const float* const* tap_src, int num_taps, const float* W, const float* bias, int H, int C,
                     float* out, int num_frames)
{
  const int Cq = C / 4;
  for (int f = 0; f < num_frames; f++)
  {
    float* oc = out + static_cast<size_t>(f) * H;
    for (int h = 0; h < H; h++)
    {
      float32x4_t acc = vdupq_n_f32(0.0f);
      for (int k = 0; k < num_taps; k++)
      {
        const float* src = tap_src[k] + static_cast<size_t>(f) * C;
        const float* w = W + (static_cast<size_t>(k) * H + h) * C;
        for (int q = 0; q < Cq; q++)
          acc = vfmaq_f32(acc, vld1q_f32(w + 4 * q), vld1q_f32(src + 4 * q));
      }
      float y = vaddvq_f32(acc);
      if (bias != nullptr)
        y += bias[h];
      oc[h] = y;
    }
  }
}

// Per-channel-count dispatch table.
struct KernelFns
{
  void (*conv)(const float* const*, int, const float*, const float*, const float*, const float*, float*, int) =
    nullptr;
  void (*tail)(const float*, const float*, const float*, bool, float*, float*, int) = nullptr;
  void (*rechannel)(const float*, const float*, int, float*, int) = nullptr;
};

template <int Q>
KernelFns make_kernel_fns()
{
  KernelFns fns;
  fns.conv = &conv_block<Q>;
  fns.tail = &tail_block<Q>;
  fns.rechannel = &rechannel_block<Q>;
  return fns;
}

KernelFns kernel_fns_for_channels(int channels)
{
  switch (channels / 4)
  {
    case 1: return make_kernel_fns<1>();
    case 2: return make_kernel_fns<2>();
    case 3: return make_kernel_fns<3>();
    case 4: return make_kernel_fns<4>();
    case 5: return make_kernel_fns<5>();
    case 6: return make_kernel_fns<6>();
    case 7: return make_kernel_fns<7>();
    case 8: return make_kernel_fns<8>();
    default: throw std::runtime_error("FusedWaveNet: unsupported channel count " + std::to_string(channels));
  }
}

// =============================================================================
// FusedWaveNet
// =============================================================================

class FusedWaveNet : public DSP
{
public:
  FusedWaveNet(const ShapeSpec& spec, std::vector<float> weights, double expected_sample_rate)
  : DSP(/*in_channels=*/1, /*out_channels=*/spec.arrays.back().head_size, expected_sample_rate)
  {
    for (const auto& as : spec.arrays)
    {
      Array arr;
      arr.c_in = as.input_size;
      arr.channels = as.channels;
      arr.head_size = as.head_size;
      arr.head_kernel_size = as.head_kernel_size;
      arr.head_bias_active = as.head_bias;
      arr.fns = kernel_fns_for_channels(as.channels);
      for (const auto& ls : as.layers)
      {
        Layer layer;
        layer.kernel_size = ls.kernel_size;
        layer.dilation = ls.dilation;
        layer.max_lookback = (ls.kernel_size - 1) * ls.dilation;
        layer.has_l1x1 = as.layer1x1_active;
        layer.act = resolve_activation(ls.activation);
        arr.layers.push_back(std::move(layer));
      }
      _arrays.push_back(std::move(arr));
    }

    // The residual output of the very last layer feeds nothing (the generic
    // path computes and then never reads it), so skip its layer1x1 GEMM. Its
    // weights are still consumed by _load_weights.
    if (!_arrays.empty() && !_arrays.back().layers.empty())
      _arrays.back().layers.back().skip_l1x1_output = true;

    _load_weights(weights);

    int prewarm = 1;
    for (const auto& arr : _arrays)
    {
      for (const auto& layer : arr.layers)
        prewarm += layer.max_lookback;
      prewarm += arr.head_kernel_size - 1;
    }
    _prewarm_samples = prewarm;
  }

  ~FusedWaveNet() override = default;

  void process(NAM_SAMPLE** input, NAM_SAMPLE** output, int num_frames) override
  {
    if (num_frames > GetMaxBufferSize())
      SetMaxBufferSize(num_frames);
    const int N = num_frames;

    // Condition buffer (float copy of the input).
    float* cond = _cond.data();
    const NAM_SAMPLE* in0 = input[0];
    for (int f = 0; f < N; f++)
      cond[f] = static_cast<float>(in0[f]);

    float* lin = _lin_a.data();
    float* alt = _lin_b.data();
    const float* prev_head_out = nullptr;
    int prev_channels = 0;

    const float* tap_src[kMaxTaps];

    for (auto& arr : _arrays)
    {
      const int C = arr.channels;

      // Rechannel into the layer input buffer.
      if (prev_head_out == nullptr)
      {
        arr.fns.rechannel(arr.rechannel_w.data(), cond, 1, lin, N);
      }
      else
      {
        arr.fns.rechannel(arr.rechannel_w.data(), lin, prev_channels, alt, N);
        std::swap(lin, alt);
      }

      // Head accumulator: zero for the first array, copy of the previous
      // array's head output after (its head_size == this array's channels).
      if (prev_head_out == nullptr)
        std::memset(_head_sum.data(), 0, static_cast<size_t>(C) * N * sizeof(float));
      else
        std::memcpy(_head_sum.data(), prev_head_out, static_cast<size_t>(C) * N * sizeof(float));

      for (auto& layer : arr.layers)
      {
        const int K = layer.kernel_size;
        if (layer.max_lookback > 0)
        {
          layer.ring.write(lin, N);
          for (int k = 0; k < K; k++)
            tap_src[k] = layer.ring.tap((K - 1 - k) * layer.dilation, N);
        }
        else
        {
          tap_src[0] = lin; // K == 1: reads only the current block
        }
        arr.fns.conv(tap_src, K, layer.conv_w.data(), layer.conv_b.data(), layer.mixin_w.data(), cond, _z.data(), N);
        apply_activation(layer.act, _z.data(), C * N);
        arr.fns.tail(_z.data(), layer.l1x1_w.data(), layer.l1x1_b.data(),
                     layer.has_l1x1 && !layer.skip_l1x1_output, _head_sum.data(), lin, N);
      }

      // Head rechannel conv over the accumulated head signal.
      const int Kh = arr.head_kernel_size;
      if (Kh > 1)
      {
        arr.head_ring.write(_head_sum.data(), N);
        for (int k = 0; k < Kh; k++)
          tap_src[k] = arr.head_ring.tap(Kh - 1 - k, N);
      }
      else
      {
        tap_src[0] = _head_sum.data();
      }
      head_conv_block(tap_src, Kh, arr.head_w.data(), arr.head_bias_active ? arr.head_b.data() : nullptr,
                      arr.head_size, C, arr.head_out.data(), N);

      prev_head_out = arr.head_out.data();
      prev_channels = C;
    }

    // Output: head_scale * last head output.
    const int out_channels = NumOutputChannels();
    const float scale = _head_scale;
    const float* ho = _arrays.back().head_out.data();
    if (out_channels == 1)
    {
      NAM_SAMPLE* out0 = output[0];
      for (int f = 0; f < N; f++)
        out0[f] = static_cast<NAM_SAMPLE>(scale * ho[f]);
    }
    else
    {
      for (int ch = 0; ch < out_channels; ch++)
      {
        NAM_SAMPLE* oc = output[ch];
        for (int f = 0; f < N; f++)
          oc[f] = static_cast<NAM_SAMPLE>(scale * ho[static_cast<size_t>(f) * out_channels + ch]);
      }
    }
  }

protected:
  void SetMaxBufferSize(int maxBufferSize) override
  {
    DSP::SetMaxBufferSize(maxBufferSize);
    int max_channels = 0;
    for (auto& arr : _arrays)
    {
      max_channels = std::max(max_channels, arr.channels);
      for (auto& layer : arr.layers)
      {
        if (layer.max_lookback > 0)
          layer.ring.reset(arr.channels, layer.max_lookback, maxBufferSize);
      }
      if (arr.head_kernel_size > 1)
        arr.head_ring.reset(arr.channels, arr.head_kernel_size - 1, maxBufferSize);
      arr.head_out.assign(static_cast<size_t>(arr.head_size) * maxBufferSize, 0.0f);
    }
    const size_t buf = static_cast<size_t>(max_channels) * maxBufferSize;
    _cond.assign(static_cast<size_t>(maxBufferSize), 0.0f);
    _lin_a.assign(buf, 0.0f);
    _lin_b.assign(buf, 0.0f);
    _z.assign(buf, 0.0f);
    _head_sum.assign(buf, 0.0f);
  }

  int GetPrewarmSamples() override { return _prewarm_samples; }

private:
  static constexpr int kMaxTaps = kMaxKernelSize;

  struct Layer
  {
    int kernel_size = 0;
    int dilation = 0;
    int max_lookback = 0;
    bool has_l1x1 = true;
    // True for the final layer of the final array: its residual output is
    // never consumed, so the layer1x1 GEMM is skipped (weights still loaded).
    bool skip_l1x1_output = false;
    ResolvedActivation act;
    // Dilated conv (C -> C), column-major per tap + bias.
    std::vector<float> conv_w;
    std::vector<float> conv_b;
    // Input mixin (1 -> C), no bias.
    std::vector<float> mixin_w;
    // layer1x1 (C -> C) column-major + bias (empty when !has_l1x1).
    std::vector<float> l1x1_w;
    std::vector<float> l1x1_b;
    Ring ring;
  };

  struct Array
  {
    int c_in = 0;
    int channels = 0;
    int head_size = 0;
    int head_kernel_size = 1;
    bool head_bias_active = false;
    KernelFns fns;
    // Rechannel (c_in -> C), column-major, no bias.
    std::vector<float> rechannel_w;
    std::vector<Layer> layers;
    // Head rechannel (C -> head_size), row-major per tap + optional bias.
    std::vector<float> head_w;
    std::vector<float> head_b;
    Ring head_ring;
    std::vector<float> head_out;
  };

  /// Consumes weights in exactly the order the generic WaveNet does:
  /// per array: rechannel; per layer: conv (+bias), mixin, layer1x1 (+bias);
  /// head rechannel (+bias). Trailing value: head_scale.
  void _load_weights(std::vector<float>& weights)
  {
    auto it = weights.begin();
    const auto end = weights.end();
    auto take = [&]() -> float {
      if (it == end)
        throw std::runtime_error("FusedWaveNet: weight stream exhausted");
      return *it++;
    };

    for (auto& arr : _arrays)
    {
      const int C = arr.channels;
      // Rechannel Conv1x1 (c_in -> C, no bias): for i in C: for j in c_in.
      arr.rechannel_w.assign(static_cast<size_t>(C) * arr.c_in, 0.0f);
      for (int i = 0; i < C; i++)
        for (int j = 0; j < arr.c_in; j++)
          arr.rechannel_w[static_cast<size_t>(j) * C + i] = take();

      for (auto& layer : arr.layers)
      {
        const int K = layer.kernel_size;
        // Conv1D (C -> C, K taps, bias): for i: for j: for k; then bias.
        layer.conv_w.assign(static_cast<size_t>(K) * C * C, 0.0f);
        layer.conv_b.assign(static_cast<size_t>(C), 0.0f);
        for (int i = 0; i < C; i++)
          for (int j = 0; j < C; j++)
            for (int k = 0; k < K; k++)
              layer.conv_w[(static_cast<size_t>(k) * C + j) * C + i] = take();
        for (int i = 0; i < C; i++)
          layer.conv_b[i] = take();
        // Input mixin Conv1x1 (1 -> C, no bias).
        layer.mixin_w.assign(static_cast<size_t>(C), 0.0f);
        for (int i = 0; i < C; i++)
          layer.mixin_w[i] = take();
        // layer1x1 (C -> C, bias) when active.
        if (layer.has_l1x1)
        {
          layer.l1x1_w.assign(static_cast<size_t>(C) * C, 0.0f);
          layer.l1x1_b.assign(static_cast<size_t>(C), 0.0f);
          for (int i = 0; i < C; i++)
            for (int j = 0; j < C; j++)
              layer.l1x1_w[static_cast<size_t>(j) * C + i] = take();
          for (int i = 0; i < C; i++)
            layer.l1x1_b[i] = take();
        }
        else
        {
          // Kernels never read these when has_l1x1 is false, but keep them
          // allocated so .data() stays valid.
          layer.l1x1_w.assign(static_cast<size_t>(C) * C, 0.0f);
          layer.l1x1_b.assign(static_cast<size_t>(C), 0.0f);
        }
      }

      // Head rechannel Conv1D (C -> head_size, Kh taps): for i: for j: for k.
      const int H = arr.head_size;
      const int Kh = arr.head_kernel_size;
      arr.head_w.assign(static_cast<size_t>(Kh) * H * C, 0.0f);
      for (int i = 0; i < H; i++)
        for (int j = 0; j < C; j++)
          for (int k = 0; k < Kh; k++)
            arr.head_w[(static_cast<size_t>(k) * H + i) * C + j] = take();
      arr.head_b.assign(static_cast<size_t>(H), 0.0f);
      if (arr.head_bias_active)
        for (int i = 0; i < H; i++)
          arr.head_b[i] = take();
    }

    _head_scale = take();

    if (it != end)
    {
      throw std::runtime_error("FusedWaveNet: weight stream has " + std::to_string(std::distance(it, end))
                               + " unconsumed values");
    }
  }

  std::vector<Array> _arrays;
  float _head_scale = 1.0f;
  int _prewarm_samples = 0;

  // Working buffers (sized in SetMaxBufferSize; no allocation in process()).
  std::vector<float> _cond;
  std::vector<float> _lin_a;
  std::vector<float> _lin_b;
  std::vector<float> _z;
  std::vector<float> _head_sum;
};

} // namespace

  #endif // NAM_FUSED_KERNELS

// =============================================================================
// Public API
// =============================================================================

namespace
{

struct FusedConfig : public ModelConfig
{
  ShapeSpec spec;

  std::unique_ptr<DSP> create(std::vector<float> weights, double sampleRate) override
  {
  #if defined(NAM_FUSED_KERNELS)
    return std::make_unique<FusedWaveNet>(spec, std::move(weights), sampleRate);
  #else
    (void)weights;
    (void)sampleRate;
    throw std::runtime_error("FusedConfig::create called on a build without NEON kernels");
  #endif
  }
};

} // namespace

bool available()
{
  #if defined(NAM_FUSED_KERNELS)
  return true;
  #else
  return false;
  #endif
}

bool is_fused_shape(const nlohmann::json& config)
{
  if (!available())
    return false;
  return parse_spec(config, nullptr);
}

std::unique_ptr<ModelConfig> create_fused_config(const nlohmann::json& config, double sampleRate)
{
  (void)sampleRate;
  auto out = std::make_unique<FusedConfig>();
  if (!available() || !parse_spec(config, &out->spec))
    throw std::runtime_error("create_fused_config: config does not match a fused-engine shape");
  return out;
}

} // namespace fused
} // namespace wavenet
} // namespace nam

#endif // NAM_ENABLE_FUSED
