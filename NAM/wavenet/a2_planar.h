#pragma once

// Planar NEON kernels for the A2 fast path (Apple Silicon only).
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
// NAM_A2_PLANAR is defined only when the A2 fast path is being built for Apple
// Silicon. On every other target -- x86, and also every *other* AArch64 target
// -- this header declares nothing, a2_planar.cpp compiles to an object with no
// symbols, the call site in a2_fast.cpp is preprocessed away, and the A2 path
// is byte for byte the code that is there today. There is nothing to regress.
//
// The gate is __APPLE__ rather than plain __aarch64__ on purpose. The kernels
// are almost certainly correct and probably faster on any AArch64 part, but
// they have only been built and measured on Apple Silicon, and two things there
// are toolchain-dependent rather than architectural: the tile widths are M2
// measurements, and bit-identity relies on the compiler contracting a*b+c into
// an FMA in a2_fast's own 3-channel branch, which clang and gcc do by default
// and MSVC at /fp:precise does not. Rather than claim a target nobody has run,
// the gate stops at the one that has been.
//
// NAM_DISABLE_A2_PLANAR opts out on Apple Silicon too, which is what makes an
// A/B measurement against the reference a one-flag change.
// -----------------------------------------------------------------------------

#if defined(NAM_ENABLE_A2_FAST)

  #if defined(__APPLE__) && defined(__aarch64__) && !defined(NAM_DISABLE_A2_PLANAR)
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
