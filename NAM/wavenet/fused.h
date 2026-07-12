#pragma once

// Fused WaveNet fast path for AArch64 (Apple Silicon and other ARMv8+ CPUs).
//
// Covers the common "standard shape" WaveNet family — including the classic
// A1 standard/lite/feather models (channels 16/8/4, kernel 3) and the A2
// standard model (channels 8) — with hand-written NEON kernels:
//
//   - The dilated conv, input mixin, bias, activation, head accumulation,
//     layer1x1 and residual add are fused into three passes over an
//     L1-resident block buffer (vs ~8 buffer passes + 4 Eigen GEMM calls per
//     layer in the generic path).
//   - GEMMs are register-tiled with vfmaq_laneq_f32 and compile-time channel
//     counts; measured at ~90% of fp32 FMA peak on Apple M-series for the
//     16-channel A1 layer (Eigen's dynamic GEMM reaches ~40% and allocates).
//   - Conv history lives in power-of-2 ring buffers with a mirrored tail so
//     every tap read is one contiguous span (no rewind memmove spikes).
//   - Activations use vectorized NEON implementations where a known
//     activation is in effect (fast tanh, ReLU, LeakyReLU, Hardtanh,
//     Softsign); anything else (exact Tanh, Sigmoid, PReLU, LUTs) falls back
//     to the exact same Activation object the generic path would call, so
//     semantics never diverge.
//
// When NAM_ENABLE_FUSED is defined at build time, wavenet::create_config
// consults is_fused_shape() on every incoming WaveNet config and, on match,
// instantiates the fused engine instead of the generic WaveNet. The detector
// only matches on AArch64 builds; other platforms always use the generic
// path.
//
// Shape requirements (checked strictly by is_fused_shape):
//   - in_channels == 1, no condition DSP, no post-stack head
//   - each layer array: condition_size == 1, bottleneck == channels,
//     channels a multiple of 4 (max 32), all groups == 1, gating "none",
//     no FiLM, head1x1 inactive, layer1x1 groups == 1
//   - arbitrary kernel sizes, dilations, activations, head kernel size/bias

#if defined(NAM_ENABLE_FUSED)

  #include <memory>

  #include "../model_config.h"
  #include "json.hpp"

namespace nam
{
namespace wavenet
{
namespace fused
{

/// \brief Whether this build has the fused NEON kernels (AArch64 only).
bool available();

/// \brief Strict detector: returns true iff the config matches a shape the
///        fused engine supports *and* the kernels are available in this build.
/// \param config The "config" sub-object from a .nam WaveNet entry.
bool is_fused_shape(const nlohmann::json& config);

/// \brief Build a ModelConfig that instantiates the fused engine.
/// \pre is_fused_shape(config) returned true.
std::unique_ptr<ModelConfig> create_fused_config(const nlohmann::json& config, double sampleRate);

} // namespace fused
} // namespace wavenet
} // namespace nam

#endif // NAM_ENABLE_FUSED
