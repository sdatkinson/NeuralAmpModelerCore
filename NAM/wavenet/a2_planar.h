#pragma once

// Planar NEON kernels for the A2 fast path (AArch64).
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
// -----------------------------------------------------------------------------
// Where this is active, and where it is not
//
// NAM_A2_PLANAR is defined only when the A2 fast path is being built for
// AArch64. On every other target -- x86 above all -- this header declares
// nothing, a2_planar.cpp compiles to an object with no symbols, the call site in
// a2_fast.cpp is preprocessed away, and the A2 path is byte for byte the code
// that is there today. There is nothing to regress.
//
// The gate was __APPLE__ && __aarch64__ at first, because Apple Silicon was the
// only place these had been built and measured. It has since been widened to
// AArch64 generally, on evidence rather than optimism:
//
//   * Bit-identity holds off Apple. The property it leans on is the compiler
//     contracting a*b+c into an FMA inside a2_fast's *own* 3-channel branch,
//     which is a toolchain behaviour, not an architectural one. Checked on a
//     Cortex-A76 (Raspberry Pi 500, Ubuntu 24.04) under GCC 13, and on Neoverse
//     N2 under GCC 14 and Clang 18: both submodels bit-identical to a2_fast,
//     max|diff| exactly zero, over a full render.
//
//   * The speed holds too, though the shape of the win is not the same. On an
//     M2: 2.47x on A2 standard and 2.00x on A2 nano. On a Cortex-A76: 2.13x and
//     2.94x. Faster on both parts, on both submodels.
//
// __aarch64__ specifically, rather than a spelling that would also catch MSVC's
// _M_ARM64. That is deliberate and is the one part of the old gate worth
// keeping: MSVC at /fp:precise does not contract a*b+c into an FMA, so the
// reference branch it would be compared against computes something else, and
// bit-identity -- the whole claim -- would not hold. clang-cl on ARM64 defines
// __aarch64__ and is fine.
//
// The tile widths remain M2 measurements. They affect speed only, never output,
// and the Cortex-A76's rather different profile suggests re-tuning them per part
// would be worth someone's time.
//
// NAM_DISABLE_A2_PLANAR opts out anywhere, which is what makes an A/B
// measurement against the reference a one-flag change.
// -----------------------------------------------------------------------------

#if defined(NAM_ENABLE_A2_FAST)

  #if defined(__aarch64__) && !defined(NAM_DISABLE_A2_PLANAR)
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
