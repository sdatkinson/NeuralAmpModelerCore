#pragma once

// Planar NEON kernels for the A2 fast path (AArch64 only).
//
// These are drop-in replacements for A2FastModel<3> and A2FastModel<8> that
// produce **bit-identical** output: not "within a tolerance", not "below the
// noise floor" -- the same float32 bits, sample for sample.
//
// The idea in one line: a2_fast keeps the channels of a frame adjacent in
// memory and vectorises across channels; these kernels keep each channel in its
// own plane and vectorise across *frames*, so one NEON lane runs a2_fast's
// per-frame scalar reduction verbatim. Nothing is reassociated, which is what
// makes the bit-identity claim hold rather than being a lucky accident.
//
// Availability is decided here rather than at the call site: NAM_A2_PLANAR is
// defined only when the A2 fast path is built for AArch64. Everywhere else this
// header declares nothing and a2_fast keeps its existing behaviour. Define
// NAM_DISABLE_A2_PLANAR to opt out on AArch64 too (useful for A/B measurement).

#if defined(NAM_ENABLE_A2_FAST)

  #if (defined(__aarch64__) || defined(_M_ARM64)) && !defined(NAM_DISABLE_A2_PLANAR)
    #define NAM_A2_PLANAR 1
  #endif

  #if defined(NAM_A2_PLANAR)

    #include <memory>
    #include <vector>

    #include "../dsp.h"

namespace nam
{
namespace wavenet
{
namespace a2_fast
{

/// \brief Build the planar NEON model for an A2 submodel.
/// \param channels 3 (A2 nano) or 8 (A2 standard); anything else yields nullptr.
/// \param weights The A2 weight stream, consumed in A2FastModel's order.
/// \param expected_sample_rate Passed through to DSP.
/// \return The model, or nullptr when this channel count has no planar kernel
///         (the caller then falls back to A2FastModel).
std::unique_ptr<DSP> create_a2_planar_model(int channels, std::vector<float> weights, double expected_sample_rate);

} // namespace a2_fast
} // namespace wavenet
} // namespace nam

  #endif // NAM_A2_PLANAR
#endif // NAM_ENABLE_A2_FAST
