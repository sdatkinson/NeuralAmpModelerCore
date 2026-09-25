#pragma once

// Planar NEON kernels for the A2 fast path (AArch64 and ARMv7).
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
// NAM_A2_PLANAR is defined only when the A2 fast path is being built for an ARM
// target that has fused multiply-add. On every other target -- x86 above all --
// this header declares nothing, a2_planar.cpp compiles to an object with no
// symbols, the call site in a2_fast.cpp is preprocessed away, and the A2 path is
// byte for byte the code that is there today. There is nothing to regress.
//
// The gate was __APPLE__ && __aarch64__ at first, because Apple Silicon was the
// only place these had been built and measured. It has since been widened
// twice, each time on evidence rather than optimism.
//
// -----------------------------------------------------------------------------
// At a glance
//
// The promoted kernel is a2_planar in every case: one engine, four
// configurations. What varies is the tile width, how a weight reaches the
// multiplier, and -- at C=8 -- the shape of the conv loop.
//
//   target     submodel      C  tile  weights    C=8 conv loop
//   ------------------------------------------------------------------
//   AArch64    A2 nano       3    32  by-lane    --
//   AArch64    A2 standard   8     8  by-lane    fold over lambdas
//   ARMv7      A2 nano       3     8  broadcast  --
//   ARMv7      A2 standard   8     8  broadcast  plain loop nest
//
// "by-lane" is vfmaq_laneq_f32, which encodes the lane in the instruction;
// "broadcast" is vld1q_dup_f32 at the point of use, because ARMv7 has no
// by-element FMA at all. Both sections below say why the remaining two columns
// differ, and neither difference was inherited -- each was measured on the part.
//
// The speed, all of it bit-identical to a2_fast over a full render:
//
//   part                  submodel      C  vs a2_fast  conditions
//   ------------------------------------------------------------------
//   Apple M2              A2 nano       3       2.00x  see note
//   Apple M2              A2 standard   8       2.47x  see note
//   Cortex-A76 (Pi 500)   A2 nano       3       2.87x  block 32
//   Cortex-A76 (Pi 500)   A2 standard   8       2.40x  block 32
//   Cortex-A17 (RK3288)   A2 nano       3       1.47x  block 32, 1416 MHz
//   Cortex-A17 (RK3288)   A2 standard   8       1.42x  block 32, 1416 MHz
//
// The M2 rows are carried from the Apple Silicon campaign these kernels came
// out of, and have not been re-measured since; every other row is a direct
// measurement of the code as it stands. The A76 rows are at a 32-frame block,
// which is why they do not match the 2.13x/2.94x quoted just below -- same
// code, different block size, not a discrepancy.
//
// Note the inversion. On the M2 the wide model wins bigger; on both ARM parts
// the narrow one does. That is the same architectural story as the tile widths
// further down, and the reason none of this travels between targets by
// assumption.
// -----------------------------------------------------------------------------
//
// AArch64 generally:
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
//     2.94x. Faster on both parts, on both submodels. (At a 32-frame block the
//     A76 reads 2.40x and 2.87x; see the table above.)
//
// ARMv7-A with NEON and VFPv4 (see the caveats below, which are load-bearing):
//
//   * Measured on a Rockchip RK3288 (quad Cortex-A17, ASUS Tinker Board), GCC
//     13.3, clock-pinned to 1416 MHz, over the same 523,808-frame render. Both
//     submodels bit-identical to a2_fast, max|diff| exactly zero.
//
//   * A2 standard 78.79% -> 55.63% of one core and A2 nano 12.29% -> 8.36% at
//     32-frame blocks (1.416x and 1.470x). Smaller than the AArch64 wins, and
//     on this part that is the difference between one instance per core with
//     nothing left over and one with room for the rest of a signal chain.
//
// __aarch64__ specifically, rather than a spelling that would also catch MSVC's
// _M_ARM64. That is deliberate and is the one part of the old gate worth
// keeping: MSVC at /fp:precise does not contract a*b+c into an FMA, so the
// reference branch it would be compared against computes something else, and
// bit-identity -- the whole claim -- would not hold. clang-cl on ARM64 defines
// __aarch64__ and is fine.
//
// The ARMv7 arm of the gate additionally requires __ARM_NEON and
// __ARM_FEATURE_FMA, and both are load-bearing rather than defensive:
//
//   * Without NEON there is no kernel at all.
//   * Without FMA, Eigen stops defining EIGEN_VECTORIZE_FMA and selects the
//     non-fused vmlaq_f32 for every pmadd, which changes a2_fast's *own* C=8
//     arithmetic. The kernels would then be bit-identical to a reference that
//     is no longer there. -mfpu=neon alone reaches that state; -mfpu=neon-vfpv4
//     is what makes the comparison meaningful.
//
// -----------------------------------------------------------------------------
// Three caveats on ARMv7, stated here rather than buried
//
//   1. **Bit-identity rests on FPSCR.FZ.** AArch32 Advanced SIMD is
//      *unconditionally* flush-to-zero for single precision, while the VFP
//      scalar code these kernels are compared against honours the FZ bit. The
//      two agree bit-for-bit only when the host has FZ set -- which audio hosts
//      usually do, and which a host application is under no obligation to do. If
//      FZ is clear and a denormal reaches a layer, the NEON kernel and the
//      scalar reference will differ. This is the one condition under which the
//      claim above is false. AArch64 has no such caveat: there FPCR.FZ applies
//      to scalar and vector alike, so both sides move together.
//
//   2. **The C=8 claim is a GCC claim.** Under clang 18.1.3 on this target
//      Eigen's gebp_kernel emits non-fused vmla.f32, so a2_fast itself computes
//      different bits (reference checksum -17.478711597881365 under GCC against
//      -17.478718637490147 under clang) and these kernels land 127.5 dB from it
//      rather than at zero. The general honest form of the claim is
//      "bit-identical wherever the reference contracts its own FMAs".
//
//   3. **The C=3 speed is a GCC claim**, for an unrelated reason: clang declines
//      to inline the tile helpers and spills the accumulators the kernel exists
//      to keep in registers, which costs most of the win. Correctness is
//      unaffected; only the speed is.
//
// -----------------------------------------------------------------------------
// What is per-architecture here, and why none of it was inherited
//
// Two things vary by target, and neither changes the output -- only the speed.
//
// **Tile widths**, and the difference is not a small one. They were swept
// separately on each architecture, and the AArch64 values are actively wrong on
// ARMv7: the C=3 ladder peaks at 32 frames on an M2 and at 8 on a Cortex-A17,
// with tile 32 running *slower than a2_fast* there. ARMv7 has 16 Q registers
// against AArch64's 32, and tile 8 is the last rung whose accumulators fit.
//
// **The shape of the C=8 conv loop**, which is the less obvious one and cost
// more to find. The AArch64 form unrolls the input and output channels with a
// fold over generic lambdas, because it needs each index as a compile-time
// value for the by-lane FMA encoding. ARMv7 has no by-lane FMA, so it gains
// nothing from that -- and loses a great deal: with 16 registers the
// accumulators cannot stay resident, and GCC schedules the unavoidable spill
// traffic far better for a plain loop nest than for the fold. The two forms are
// 68.7% and 55.4% of one core on the same part. See the comment at the branch
// itself for the instruction and memory-access counts.
//
// The general lesson, stated because it was learned the expensive way: porting
// these kernels is not a matter of widening the gate and re-sweeping the tiles.
// The first ARMv7 build of this file was bit-identical, passed every
// conformance check, and was still slower than the kernel it was ported from by
// enough to lose most of the win -- and nothing but a measurement on the part
// said so. Re-measure against the reference on the target itself, and do not
// trust a figure carried over from a lab kernel of the same shape.
//
// See A32-PATH.md in the NAMBench repository for the full ladder and the spill
// counts underneath it.
//
// NAM_DISABLE_A2_PLANAR opts out anywhere, which is what makes an A/B
// measurement against the reference a one-flag change.
// -----------------------------------------------------------------------------

#if defined(NAM_ENABLE_A2_FAST)

  #if !defined(NAM_DISABLE_A2_PLANAR)
    #if defined(__aarch64__)
      #define NAM_A2_PLANAR 1
    #elif defined(__arm__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_FMA)
      #define NAM_A2_PLANAR 1
      /// 32-bit ARM: 16 Q registers, no by-element FMA, narrower NEON datapath.
      /// Selects the ARMv7 weight delivery and tile widths in a2_planar.cpp.
      #define NAM_A2_PLANAR_A32 1
    #endif
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
