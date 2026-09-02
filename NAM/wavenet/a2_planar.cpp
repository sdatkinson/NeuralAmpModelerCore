#if defined(NAM_ENABLE_A2_FAST)

  #include "a2_planar.h"

  #if defined(NAM_A2_PLANAR)

    #include <array>
    #include <cstddef>
    #include <cstring>
    #include <iterator>
    #include <memory>
    #include <sstream>
    #include <stdexcept>
    #include <string>
    #include <utility>
    #include <vector>

    #include <arm_neon.h>

    #include "a2_fast.h"

// =============================================================================
// Planar NEON kernels for the A2 fast path.
//
// Both kernels are bit-identical to the A2FastModel<Channels> they replace, over
// the whole of a 523,808-frame test render. That is the point of the design, so
// it is worth being precise about how it is achieved, because the two channel
// counts get there by reproducing two *different* orders of arithmetic --
// a2_fast itself branches, and so does this.
//
// The shared idea: planar layout. a2_fast stores history column-major, the
// channels of one frame adjacent, and a SIMD register naturally spans channels.
// Here each channel gets its own plane, so a register holds four consecutive
// *frames* of one channel and each lane independently runs a2_fast's per-frame
// scalar chain. No cross-lane reduction ever happens, so no reassociation
// happens, so the bits do not move.
//
// Channels == 3 reproduces a2_fast's hand-written scalar 3x3 GEMV: bias first,
// then per tap the three input channels in increasing order, all contracted into
// FMAs, mixin contracted too.
//
// Channels == 8 reproduces what a2_fast's Eigen expressions actually compute:
//
//     per tap k:  t_i = 0;  for j = 0..7:  t_i = fma(W_k(i,j), x_k(j), t_i)
//     then:       z_i = 0;  for k:         z_i = z_i + t_i^(k)
//                 z_i = z_i + bias_i
//                 z_i = z_i + (mixin_i * cond)     <- product then add, two roundings
//                 z_i = LeakyReLU(z_i)
//                 head_sum_i = head_sum_i + z_i
//                 u_i = 0;  for j:  u_i = fma(L(i,j), z_j, u_i)
//                 lin_i = (lin_i + u_i) + l1x1_b_i
//
// That per-tap partial `t`, summed into `z` only at the end of the tap, is the
// part that is easy to get wrong: folding the taps into one chain (the obvious
// thing to write) is a different association and does move the bits. The order
// above was established by comparing candidate orderings bit-for-bit against
// Eigen's own output across 224 (block size, kernel size, trial) combinations,
// not by assumption.
//
// The tuning constants below (frame tile widths, head tile, ring strategy) were
// each swept independently on an Apple M2 against the full-length render; the
// values chosen are the measured optima and the comments say what they trade.
// They do not affect the output, only the speed.
// =============================================================================

namespace nam
{
namespace wavenet
{
namespace a2_fast
{

namespace
{

// -----------------------------------------------------------------------------
// The two instruction sets, reconciled
//
// AArch64 multiply-accumulates against a *lane* of a loaded weight vector --
// vfmaq_laneq_f32 and friends. ARMv7's VFPv4 has no VFMA-by-scalar encoding at
// all, so none of the by-element forms exist. The replacement is not the same
// thing spelled differently: on a machine with 16 Q registers rather than 32,
// the right move is not to hold the weight vector at all, but to broadcast each
// weight straight out of memory as it is used (vld1q_dup_f32, one instruction).
//
// Both spellings are exact, for three reasons:
//
//   1. A splat is a bit-copy. The multiplicand the FMA sees is the identical
//      float32 the lane-addressed form would have seen; no rounding is added.
//   2. vfmaq_f32 is VFMA.F32 -- one rounding, exactly as vfmaq_laneq_f32 is.
//   3. Lane independence is untouched: each lane still runs one frame's chain in
//      a2_fast's order, so nothing is reassociated.
//
// LaneWeights is what keeps the two shapes behind one call site, so the kernels
// below are written once. On AArch64 it holds the weight vectors and indexes
// them by lane, compiling to exactly the instructions this file emitted before
// ARMv7 was added -- that path is byte for byte what it was. On ARMv7 it holds
// only the pointer.
// -----------------------------------------------------------------------------

    #if defined(NAM_A2_PLANAR_A32)
// The kernel's premise is that the accumulators stay in registers across the tap
// loop, which stops being true the moment a tile helper is left out of line: the
// accumulator arrays are passed by reference, so a call boundary spills them.
// GCC inlines these on its own. Clang, measured on this target, does not -- it
// leaves the helpers out of line and spends 164 loads and 39 stores to do 87
// FMAs, which costs most of the win. Forcing the decision is free where the
// compiler was going to make it anyway.
      #define NAM_A2_PLANAR_INLINE inline __attribute__((always_inline))
    #else
      #define NAM_A2_PLANAR_INLINE inline
    #endif

/// A run of 4*NV weights, delivered to the multiplier the way the target
/// prefers. Every index is a template parameter, never a runtime value: on
/// AArch64 the lane is encoded in the instruction, and on ARMv7 the offset is
/// folded into the load.
template <int NV>
struct LaneWeights
{
  static constexpr int kFloats = 4 * NV;

  const float* p;
    #if !defined(NAM_A2_PLANAR_A32)
  float32x4_t v[NV];
    #endif

  NAM_A2_PLANAR_INLINE explicit LaneWeights(const float* q)
  : p(q)
  {
    #if !defined(NAM_A2_PLANAR_A32)
    for (int u = 0; u < NV; u++)
      v[u] = vld1q_f32(q + 4 * u);
    #endif
  }

  /// Weight I broadcast to every lane.
  template <int I>
  NAM_A2_PLANAR_INLINE float32x4_t dup() const
  {
    static_assert(I >= 0 && I < kFloats, "weight index out of range");
    #if defined(NAM_A2_PLANAR_A32)
    return vld1q_dup_f32(p + I);
    #else
    return vdupq_laneq_f32(v[I / 4], I % 4);
    #endif
  }

  /// acc + s * w[I], one rounding.
  template <int I>
  NAM_A2_PLANAR_INLINE float32x4_t fma(float32x4_t acc, float32x4_t s) const
  {
    static_assert(I >= 0 && I < kFloats, "weight index out of range");
    #if defined(NAM_A2_PLANAR_A32)
    return vfmaq_f32(acc, s, vld1q_dup_f32(p + I));
    #else
    return vfmaq_laneq_f32(acc, s, v[I / 4], I % 4);
    #endif
  }

  /// s * w[I], one rounding. Seeds a chain where an FMA against zero would
  /// otherwise be needed; both are one rounding of w*x, so this is exact either
  /// way and just saves the init.
  template <int I>
  NAM_A2_PLANAR_INLINE float32x4_t mul(float32x4_t s) const
  {
    static_assert(I >= 0 && I < kFloats, "weight index out of range");
    #if defined(NAM_A2_PLANAR_A32)
    return vmulq_f32(s, vld1q_dup_f32(p + I));
    #else
    return vmulq_laneq_f32(s, v[I / 4], I % 4);
    #endif
  }
};

/// Force `v` to be a rounded value before it is used again.
///
/// The C=8 layer body computes the mixin as a product and then a separate add --
/// two roundings, because that is what Eigen does. Under -ffp-contract=fast
/// (GCC's default) nothing stops the compiler folding that pair into one vfma,
/// which is one rounding and different bits.
///
/// This is measured rather than defensive, and only on ARMv7: an ordering probe
/// against Eigen under arm-linux-gnueabihf-g++ matches 224/224 at
/// -ffp-contract=off and 11/224 at =fast, with the conv stage unaffected either
/// way. AArch64 was checked and does not contract here, so the barrier is not
/// applied there -- those kernels are measured as they stand.
///
/// Turning contraction off globally would be the wrong fix: a2_fast's C=3 branch
/// *depends* on the compiler contracting a*b+c, which is the whole bit-identity
/// premise at that width. So contraction stays on and is blocked here, locally,
/// at exactly the point a2_fast rounds twice.
///
/// Compiles to nothing; "w" is the VFP/NEON register constraint on ARM.
NAM_A2_PLANAR_INLINE float32x4_t round_now(float32x4_t v)
{
    #if defined(NAM_A2_PLANAR_A32)
  __asm__("" : "+w"(v));
    #endif
  return v;
}

// -----------------------------------------------------------------------------
// Weights
//
// Parsed in exactly A2FastModel::_load_weights' order -- which is the generic
// WaveNet's order -- into exactly its layout, so the two engines are fed
// identical numbers and only the kernel differs.
// -----------------------------------------------------------------------------
template <int C>
struct PlanarLayerWeights
{
  int kernel_size = 0;
  int dilation = 0;
  int max_lookback = 0; // (kernel_size - 1) * dilation

  /// kernel_size * C * C floats. Tap k, output i, input j lives at [k*C*C + j*C + i].
  std::vector<float> conv_w;
  std::array<float, C> conv_b{};
  /// Input mixin (condition size 1 -> C), no bias.
  std::array<float, C> mixin_w{};
  /// layer1x1 (C -> C), column-major: [j*C + i] is bottleneck j to output i.
  std::array<float, C * C> l1x1_w{};
  std::array<float, C> l1x1_b{};
};

template <int C>
struct PlanarWeights
{
  /// Rechannel (input size 1 -> C), no bias.
  std::array<float, C> rechannel_w{};
  std::array<PlanarLayerWeights<C>, kNumLayers> layers;
  /// Head rechannel (C -> 1), kernel 16. At tap k the matrix is 1 x C.
  std::array<std::array<float, C>, kHeadKernelSize> head_w{};
  float head_b = 0.0f;
  /// The trailing float of the stream, which overrides the JSON head_scale.
  float head_scale = 1.0f;
};

template <int C>
PlanarWeights<C> parse_weights(const std::vector<float>& weights)
{
  PlanarWeights<C> out;

  auto it = weights.begin();
  const auto end = weights.end();
  auto take = [&]() -> float {
    if (it == end)
      throw std::runtime_error("A2PlanarModel: weight stream exhausted");
    return *it++;
  };

  for (int i = 0; i < C; i++)
    out.rechannel_w[i] = take();

  for (int li = 0; li < kNumLayers; li++)
  {
    PlanarLayerWeights<C>& L = out.layers[li];
    L.kernel_size = kKernelSizes[li];
    L.dilation = kDilations[li];
    L.max_lookback = (L.kernel_size - 1) * L.dilation;
    const int K = L.kernel_size;

    // Conv1D read order: for i in out, for j in in, for k in taps.
    L.conv_w.assign(static_cast<size_t>(K) * C * C, 0.0f);
    for (int i = 0; i < C; i++)
      for (int j = 0; j < C; j++)
        for (int k = 0; k < K; k++)
          L.conv_w[static_cast<size_t>(k) * C * C + static_cast<size_t>(j) * C + i] = take();
    for (int i = 0; i < C; i++)
      L.conv_b[i] = take();

    for (int i = 0; i < C; i++)
      L.mixin_w[i] = take();

    // Conv1x1 read order: for i in out, for j in in.
    for (int i = 0; i < C; i++)
      for (int j = 0; j < C; j++)
        L.l1x1_w[static_cast<size_t>(j) * C + i] = take();
    for (int i = 0; i < C; i++)
      L.l1x1_b[i] = take();
  }

  for (int j = 0; j < C; j++)
    for (int k = 0; k < kHeadKernelSize; k++)
      out.head_w[k][j] = take();
  out.head_b = take();
  out.head_scale = take();

  if (it != end)
  {
    std::stringstream ss;
    ss << "A2PlanarModel: weight stream has " << std::distance(it, end) << " trailing values";
    throw std::runtime_error(ss.str());
  }

  return out;
}

/// A product that is rounded before it is used.
///
/// The C=8 path's mixin is `z += mixin * cond` with the multiply and the add
/// rounded separately -- that is what Eigen does, and a fused multiply-add there
/// would move the result by one ulp. In the vector paths that is spelled out in
/// intrinsics; in the scalar tails, writing `a = a + m * cf` would let the
/// compiler contract it. Routing the product through a NEON register keeps the
/// two roundings whatever the compiler decides.
inline float mul_rounded(float a, float b)
{
  float p = vget_lane_f32(vmul_f32(vdup_n_f32(a), vdup_n_f32(b)), 0);
    #if defined(NAM_A2_PLANAR_A32)
  // Routing through a vector register is enough on AArch64, where the pair is
  // not contracted anyway. On ARMv7 the same barrier the vector path needs is
  // applied here too, for the same reason and with the same cost: none.
  __asm__("" : "+w"(p));
    #endif
  return p;
}

/// Receptive field, counted exactly as A2FastModel counts it, so the planar
/// models warm up over the same number of samples as the code they replace.
int planar_prewarm_samples()
{
  int prewarm = 1;
  for (int li = 0; li < kNumLayers; li++)
    prewarm += (kKernelSizes[li] - 1) * kDilations[li];
  prewarm += kHeadKernelSize - 1;
  return prewarm;
}

// -----------------------------------------------------------------------------
// Planar history ring: C channel planes side by side, one linear buffer each,
// written forward and memmoved back when it runs out.
//
// This is a2_fast's NAM_A2_RING_MODE=0 strategy, in planar layout. Four
// strategies were measured -- power-of-two with an eagerly mirrored tail
// (a2_fast's shipped default), power-of-two with a lazy mirror, exactly-sized
// with a lazy mirror, and this one -- and linear+rewind won for both channel
// counts once the residual writes go straight into the next layer's ring. It
// has no mirror to maintain, no masking on reads, and every read is contiguous;
// it pays instead with an occasional large memmove, amortised over many blocks
// by the 2*lookback sizing.
//
// ARMv7 re-ran that sweep and agrees at C=3, emphatically: linear+rewind is
// 1.378x against a2_fast where the eager mirror is 1.322x, the largest single
// gain available after the tile width. On a 32 KB L1D the mirror's per-block
// memcpy across all 23 layers is not merely overhead, it is eviction.
//
// At C=8 the same sweep put the lazy mirror marginally ahead -- 1.312x against
// linear's 1.309x -- and that difference is kept out of this file deliberately.
// It is 0.2%, against a run-to-run spread of 0.5% on the same board, so it is
// below what the measurement can resolve; carrying a second ring implementation
// into shipped code to chase it would be buying complexity with noise.
// -----------------------------------------------------------------------------
template <int C>
struct PlanarRing
{
  std::vector<float> data;
  int cap = 0; ///< columns per plane
  int stride = 0; ///< distance between channel planes (== cap here)
  int wpos = 0;
  int lookback = 0;

  void reset(int max_lookback, int max_buffer)
  {
    lookback = max_lookback;
    cap = 2 * max_lookback + max_buffer;
    stride = cap;
    data.assign(static_cast<size_t>(C) * stride, 0.0f);
    wpos = max_lookback;
  }

  float* plane(int c) { return data.data() + static_cast<size_t>(c) * stride; }
  const float* plane(int c) const { return data.data() + static_cast<size_t>(c) * stride; }

  /// Make room for an n-frame write, rewinding if it would not fit.
  void prepare(int n)
  {
    if (wpos + n > cap)
    {
      for (int c = 0; c < C; c++)
        std::memmove(plane(c), plane(c) + (wpos - lookback), static_cast<size_t>(lookback) * sizeof(float));
      wpos = lookback;
    }
  }

  /// Where an n-frame block is written. Always one contiguous run.
  float* write_ptr(int c) { return plane(c) + wpos; }

  void commit(int n) { wpos += n; }

  /// First column of an n-frame read looking `lookback_frames` further back than
  /// the block just written.
  int tap(int lookback_frames, int n) const { return wpos - n - lookback_frames; }
};

// =============================================================================
// Channels == 3 (A2 nano)
//
// Reproduces a2_fast's `if constexpr (Channels == 3)` branch: the fully
// unrolled scalar 3x3 GEMV, bias-seeded at tap 0, inputs in increasing order,
// mixin contracted into an FMA, then LeakyReLU, head_sum, layer1x1 residual.
// Each NEON lane runs that chain for its own frame.
//
// Per conv tap this is 3 loads and 9 vfmaq_laneq_f32 per 4 frames, against
// a2_fast's 9 scalar FMAs per frame.
// =============================================================================

/// Frames per tile, swept separately on each architecture because the answer is
/// not close.
///
/// AArch64: twelve accumulator registers at 32 (3 channels x 8 vectors), which
/// is where the sweep peaked -- 8/16/32/64 measured 1.39x/1.56x/1.75x/1.46x
/// against a2_fast. 64 spills.
///
/// ARMv7: the ladder inverts. 2/4/8/12/16/32 measured
/// 0.888x/1.116x/1.242x/1.086x/1.006x/0.954x, so AArch64's 32 is *slower than
/// a2_fast* here. Counting registers says why. The live set is
///
///     z accumulators C*T/lanes + t accumulators C*T/lanes + input T/lanes + 1
///
/// which at C=3, tile 8, 4 lanes is 6 + 6 + 2 + 1 = 15 of the 16 Q registers
/// this machine has; tile 12 needs 22. The measured spill counts turn over at
/// exactly that step: 0.48 memory operations per FMA at tile 8, 1.13 at 12.
/// Tile 8 is the last rung that fits, and it is the peak.
    #if defined(NAM_A2_PLANAR_A32)
constexpr int kNanoTile = 8;
    #else
constexpr int kNanoTile = 32;
    #endif
constexpr int kNanoVecs = kNanoTile / 4;

/// One layer's weights, padded to four lanes so each group of three is a single
/// vector load addressed by lane.
struct NanoLayer
{
  int kernel_size = 0;
  int dilation = 0;
  int max_lookback = 0;
  /// kernel_size x 12 floats: nine weights then three pad, per tap.
  std::vector<float> conv_w;
  std::array<float, 4> conv_b{};
  std::array<float, 4> mixin_w{};
  /// Twelve floats: nine layer1x1 weights then three pad.
  std::array<float, 12> l1x1_w{};
  std::array<float, 4> l1x1_b{};
};

class A2PlanarNano : public DSP
{
  static constexpr int C = 3;
  using Ring = PlanarRing<C>;

public:
  A2PlanarNano(const std::vector<float>& weights, double expected_sample_rate)
  : DSP(/*in_channels=*/1, /*out_channels=*/1, expected_sample_rate)
  , _w(parse_weights<C>(weights))
  , _prewarm_samples(planar_prewarm_samples())
  {
    for (int li = 0; li < kNumLayers; li++)
    {
      const PlanarLayerWeights<C>& L = _w.layers[li];
      NanoLayer& P = _p[li];
      P.kernel_size = L.kernel_size;
      P.dilation = L.dilation;
      P.max_lookback = L.max_lookback;

      P.conv_w.assign(static_cast<size_t>(L.kernel_size) * 12, 0.0f);
      for (int k = 0; k < L.kernel_size; k++)
        for (int e = 0; e < 9; e++)
          P.conv_w[static_cast<size_t>(k) * 12 + e] = L.conv_w[static_cast<size_t>(k) * 9 + e];

      for (int i = 0; i < C; i++)
      {
        P.conv_b[i] = L.conv_b[i];
        P.mixin_w[i] = L.mixin_w[i];
        P.l1x1_b[i] = L.l1x1_b[i];
      }
      for (int e = 0; e < 9; e++)
        P.l1x1_w[e] = L.l1x1_w[e];
    }

    _head_w4.assign(static_cast<size_t>(kHeadKernelSize) * 4, 0.0f);
    for (int k = 0; k < kHeadKernelSize; k++)
      for (int j = 0; j < C; j++)
        _head_w4[static_cast<size_t>(k) * 4 + j] = _w.head_w[k][j];
  }

  ~A2PlanarNano() override = default;

  int GetPrewarmSamples() override { return _prewarm_samples; }

  void process(NAM_SAMPLE** input, NAM_SAMPLE** output, const int num_frames) override
  {
    if (num_frames > GetMaxBufferSize())
      SetMaxBufferSize(num_frames);
    const int N = num_frames;

    const NAM_SAMPLE* in0 = input[0];
    NAM_SAMPLE* out0 = output[0];

    // Rechannel straight into layer 0's ring: there is no scratch buffer for a
    // later pass to copy in.
    float* cond = _cond.data();
    _rings[0].prepare(N);
    float* const r0 = _rings[0].write_ptr(0);
    float* const r1 = _rings[0].write_ptr(1);
    float* const r2 = _rings[0].write_ptr(2);
    const float rw0 = _w.rechannel_w[0], rw1 = _w.rechannel_w[1], rw2 = _w.rechannel_w[2];
    for (int f = 0; f < N; f++)
    {
      const float x = static_cast<float>(in0[f]);
      cond[f] = x;
      r0[f] = rw0 * x;
      r1[f] = rw1 * x;
      r2[f] = rw2 * x;
    }
    _rings[0].commit(N);

    // No memset of head_sum: layer 0 writes it rather than accumulating onto it.
    for (int li = 0; li < kNumLayers; li++)
      dispatch_layer(li, N);

    head_forward(N);

    const float* head_out = _head_out.data();
    for (int f = 0; f < N; f++)
      out0[f] = static_cast<NAM_SAMPLE>(head_out[f]);
  }

protected:
  void SetMaxBufferSize(const int maxBufferSize) override
  {
    DSP::SetMaxBufferSize(maxBufferSize);

    _stride = maxBufferSize;
    _layer_in.assign(static_cast<size_t>(C) * _stride, 0.0f);
    _head_sum.assign(static_cast<size_t>(C) * _stride, 0.0f);
    _cond.assign(static_cast<size_t>(maxBufferSize), 0.0f);
    _head_out.assign(static_cast<size_t>(maxBufferSize), 0.0f);

    for (int li = 0; li < kNumLayers; li++)
      _rings[li].reset(_p[li].max_lookback, maxBufferSize);
    _head_ring.reset(kHeadKernelSize - 1, maxBufferSize);
  }

private:
  float* lin(int c) { return _layer_in.data() + static_cast<size_t>(c) * _stride; }
  float* hsum(int c) { return _head_sum.data() + static_cast<size_t>(c) * _stride; }

  void dispatch_layer(int li, int N)
  {
    // The last layer's layer1x1 residual is read by nothing -- there is no layer
    // 24, and the head reads head_sum -- so it is not computed. The first
    // layer's head_sum accumulate has nothing to accumulate onto, so it stores
    // instead, which is what makes the per-block memset unnecessary.
    const bool store_head = (li == 0);
    const bool do_l1x1 = (li != kNumLayers - 1);

    if (_p[li].kernel_size == 6)
    {
      if (store_head)
        layer_forward<6, true, true>(li, N);
      else if (do_l1x1)
        layer_forward<6, false, true>(li, N);
      else
        layer_forward<6, false, false>(li, N);
    }
    else
    {
      if (store_head)
        layer_forward<15, true, true>(li, N);
      else if (do_l1x1)
        layer_forward<15, false, true>(li, N);
      else
        layer_forward<15, false, false>(li, N);
    }
  }

  template <int K, bool StoreHead, bool DoL1x1>
  void layer_forward(int li, int N)
  {
    Ring& R = _rings[li];
    const NanoLayer& P = _p[li];

    const float* h[C] = {R.plane(0), R.plane(1), R.plane(2)};
    int tapb[K];
    for (int k = 0; k < K; k++)
      tapb[k] = R.tap((K - 1 - k) * P.dilation, N);

    // Residual destination: the next layer's ring, so no layer ever copies its
    // input in from a scratch buffer.
    Ring* next = nullptr;
    float* d[C];
    if (li + 1 < kNumLayers)
    {
      next = &_rings[li + 1];
      next->prepare(N);
      for (int c = 0; c < C; c++)
        d[c] = next->write_ptr(c);
    }
    else
    {
      for (int c = 0; c < C; c++)
        d[c] = lin(c);
    }

    float* hs[C] = {hsum(0), hsum(1), hsum(2)};

    int f = 0;
    for (; f + kNanoTile <= N; f += kNanoTile)
      tile<K, kNanoVecs, StoreHead, DoL1x1>(P, h, tapb, f, d, hs);
    for (; f + 4 <= N; f += 4)
      tile<K, 1, StoreHead, DoL1x1>(P, h, tapb, f, d, hs);
    for (; f < N; f++)
      frame_scalar<K, StoreHead, DoL1x1>(P, h, tapb, f, d, hs);

    if (next != nullptr)
      next->commit(N);
  }

  /// NVEC x 4 frames, three channel accumulators per vector. z stays in
  /// registers across every tap instead of making a round trip to memory per
  /// tap per frame.
  template <int K, int NVEC, bool StoreHead, bool DoL1x1>
  NAM_A2_PLANAR_INLINE void tile(const NanoLayer& P, const float* const* h, const int (&tapb)[K], int f0,
                                 float* const* d, float* const* hs)
  {
    const LaneWeights<1> cb(P.conv_b.data());
    float32x4_t a0[NVEC], a1[NVEC], a2[NVEC];
    for (int v = 0; v < NVEC; v++)
    {
      a0[v] = cb.dup<0>();
      a1[v] = cb.dup<1>();
      a2[v] = cb.dup<2>();
    }

    const float* cw = P.conv_w.data();
    for (int k = 0; k < K; k++)
    {
      // Nine weights then three pad, per tap: [out i][in j] at j * 3 + i.
      const LaneWeights<3> w(cw + static_cast<size_t>(k) * 12);
      const int b = tapb[k] + f0;
      for (int v = 0; v < NVEC; v++)
      {
        const float32x4_t s0 = vld1q_f32(h[0] + b + 4 * v);
        const float32x4_t s1 = vld1q_f32(h[1] + b + 4 * v);
        const float32x4_t s2 = vld1q_f32(h[2] + b + 4 * v);
        a0[v] = w.fma<0>(a0[v], s0);
        a1[v] = w.fma<1>(a1[v], s0);
        a2[v] = w.fma<2>(a2[v], s0);
        a0[v] = w.fma<3>(a0[v], s1);
        a1[v] = w.fma<4>(a1[v], s1);
        a2[v] = w.fma<5>(a2[v], s1);
        a0[v] = w.fma<6>(a0[v], s2);
        a1[v] = w.fma<7>(a1[v], s2);
        a2[v] = w.fma<8>(a2[v], s2);
      }
    }

    post<K, NVEC, StoreHead, DoL1x1>(P, h, tapb[K - 1], f0, a0, a1, a2, d, hs);
  }

  /// Everything after the conv: mixin, LeakyReLU, head_sum, layer1x1 residual.
  /// `last_tap` is the base of the offset-0 tap, i.e. this block's own input --
  /// reloading it here is cheaper than carrying it through the tap loop, which
  /// is what decides how wide the tile can usefully get.
  template <int K, int NVEC, bool StoreHead, bool DoL1x1>
  NAM_A2_PLANAR_INLINE void post(const NanoLayer& P, const float* const* h, int last_tap, int f0,
                                 float32x4_t (&a0)[NVEC], float32x4_t (&a1)[NVEC], float32x4_t (&a2)[NVEC],
                                 float* const* d, float* const* hs)
  {
    const LaneWeights<1> M(P.mixin_w.data());
    const float32x4_t zero = vdupq_n_f32(0.0f);
    const float32x4_t slope = vdupq_n_f32(kLeakySlope);
    const float* cond = _cond.data();

    for (int v = 0; v < NVEC; v++)
    {
      const float32x4_t cf = vld1q_f32(cond + f0 + 4 * v);
      // Contracted, unlike the C=8 mixin: a2_fast's 3-channel branch writes
      // `a += mixin * cond` as one FMA, so one rounding is the correct order.
      a0[v] = M.fma<0>(a0[v], cf);
      a1[v] = M.fma<1>(a1[v], cf);
      a2[v] = M.fma<2>(a2[v], cf);
      a0[v] = vbslq_f32(vcltq_f32(a0[v], zero), vmulq_f32(a0[v], slope), a0[v]);
      a1[v] = vbslq_f32(vcltq_f32(a1[v], zero), vmulq_f32(a1[v], slope), a1[v]);
      a2[v] = vbslq_f32(vcltq_f32(a2[v], zero), vmulq_f32(a2[v], slope), a2[v]);
    }

    for (int v = 0; v < NVEC; v++)
    {
      const int o = f0 + 4 * v;
      if constexpr (StoreHead)
      {
        // Kept as an add against +0.0 rather than a plain store: 0.0f + (-0.0f)
        // is +0.0f, so storing would differ from a2_fast on a signed zero. One
        // vector add per four frames in one layer, and the exactness is then a
        // fact rather than an argument about whether that case can arise.
        vst1q_f32(hs[0] + o, vaddq_f32(zero, a0[v]));
        vst1q_f32(hs[1] + o, vaddq_f32(zero, a1[v]));
        vst1q_f32(hs[2] + o, vaddq_f32(zero, a2[v]));
      }
      else
      {
        vst1q_f32(hs[0] + o, vaddq_f32(vld1q_f32(hs[0] + o), a0[v]));
        vst1q_f32(hs[1] + o, vaddq_f32(vld1q_f32(hs[1] + o), a1[v]));
        vst1q_f32(hs[2] + o, vaddq_f32(vld1q_f32(hs[2] + o), a2[v]));
      }
    }

    if constexpr (DoL1x1)
    {
      // Nine weights then three pad: [out i][in j] at j * 3 + i, as the conv.
      const LaneWeights<3> L(P.l1x1_w.data());
      const LaneWeights<1> LBias(P.l1x1_b.data());

      for (int v = 0; v < NVEC; v++)
      {
        const int o = f0 + 4 * v;
        float32x4_t o0 = LBias.dup<0>();
        float32x4_t o1 = LBias.dup<1>();
        float32x4_t o2 = LBias.dup<2>();
        o0 = L.fma<0>(o0, a0[v]);
        o0 = L.fma<3>(o0, a1[v]);
        o0 = L.fma<6>(o0, a2[v]);
        o1 = L.fma<1>(o1, a0[v]);
        o1 = L.fma<4>(o1, a1[v]);
        o1 = L.fma<7>(o1, a2[v]);
        o2 = L.fma<2>(o2, a0[v]);
        o2 = L.fma<5>(o2, a1[v]);
        o2 = L.fma<8>(o2, a2[v]);
        vst1q_f32(d[0] + o, vaddq_f32(vld1q_f32(h[0] + last_tap + o), o0));
        vst1q_f32(d[1] + o, vaddq_f32(vld1q_f32(h[1] + last_tap + o), o1));
        vst1q_f32(d[2] + o, vaddq_f32(vld1q_f32(h[2] + last_tap + o), o2));
      }
    }
  }

  /// The last few frames of a block that is not a multiple of four. Same
  /// operation order as the vector path, one frame at a time.
  template <int K, bool StoreHead, bool DoL1x1>
  void frame_scalar(const NanoLayer& P, const float* const* h, const int (&tapb)[K], int f, float* const* d,
                    float* const* hs)
  {
    float a[C] = {P.conv_b[0], P.conv_b[1], P.conv_b[2]};
    for (int k = 0; k < K; k++)
    {
      const float* wk = P.conv_w.data() + static_cast<size_t>(k) * 12;
      const float s0 = h[0][tapb[k] + f];
      const float s1 = h[1][tapb[k] + f];
      const float s2 = h[2][tapb[k] + f];
      // Spelled as fused rather than written `a += w * s` and left to
      // -ffp-contract. Both compile to the same instruction where contraction is
      // on, and this way the tail's exactness is a property of the source
      // instead of a property of a flag.
      a[0] = __builtin_fmaf(wk[0], s0, a[0]);
      a[1] = __builtin_fmaf(wk[1], s0, a[1]);
      a[2] = __builtin_fmaf(wk[2], s0, a[2]);
      a[0] = __builtin_fmaf(wk[3], s1, a[0]);
      a[1] = __builtin_fmaf(wk[4], s1, a[1]);
      a[2] = __builtin_fmaf(wk[5], s1, a[2]);
      a[0] = __builtin_fmaf(wk[6], s2, a[0]);
      a[1] = __builtin_fmaf(wk[7], s2, a[1]);
      a[2] = __builtin_fmaf(wk[8], s2, a[2]);
    }

    const float cf = _cond[f];
    for (int c = 0; c < C; c++)
    {
      a[c] = __builtin_fmaf(P.mixin_w[c], cf, a[c]);
      a[c] = (a[c] < 0.0f) ? a[c] * kLeakySlope : a[c];
      if constexpr (StoreHead)
        hs[c][f] = 0.0f + a[c];
      else
        hs[c][f] = hs[c][f] + a[c];
    }

    if constexpr (DoL1x1)
    {
      for (int c = 0; c < C; c++)
      {
        float o = P.l1x1_b[c];
        o = __builtin_fmaf(P.l1x1_w[0 + c], a[0], o);
        o = __builtin_fmaf(P.l1x1_w[3 + c], a[1], o);
        o = __builtin_fmaf(P.l1x1_w[6 + c], a[2], o);
        d[c][f] = h[c][tapb[K - 1] + f] + o;
      }
    }
  }

  /// Head rechannel: K=16, dilation 1, three channels down to one, plus bias and
  /// scale. In planar layout this is 48 vector FMAs per four frames where
  /// a2_fast does 48 scalar FMAs per frame.
  void head_forward(int N)
  {
    _head_ring.prepare(N);
    for (int c = 0; c < C; c++)
      std::memcpy(_head_ring.write_ptr(c), hsum(c), static_cast<size_t>(N) * sizeof(float));
    _head_ring.commit(N);

    int hb[kHeadKernelSize];
    for (int k = 0; k < kHeadKernelSize; k++)
      hb[k] = _head_ring.tap(kHeadKernelSize - 1 - k, N);

    const float* p[C] = {_head_ring.plane(0), _head_ring.plane(1), _head_ring.plane(2)};
    const float* hw = _head_w4.data();
    const float scale = _w.head_scale;
    float* out = _head_out.data();

    int f = 0;
    for (; f + 4 <= N; f += 4)
    {
      float32x4_t y = vdupq_n_f32(_w.head_b);
      for (int k = 0; k < kHeadKernelSize; k++)
      {
        const LaneWeights<1> W(hw + static_cast<size_t>(k) * 4);
        y = W.fma<0>(y, vld1q_f32(p[0] + hb[k] + f));
        y = W.fma<1>(y, vld1q_f32(p[1] + hb[k] + f));
        y = W.fma<2>(y, vld1q_f32(p[2] + hb[k] + f));
      }
      vst1q_f32(out + f, vmulq_n_f32(y, scale));
    }
    for (; f < N; f++)
    {
      float y = _w.head_b;
      for (int k = 0; k < kHeadKernelSize; k++)
      {
        const float* w = hw + static_cast<size_t>(k) * 4;
        y = __builtin_fmaf(w[0], p[0][hb[k] + f], y);
        y = __builtin_fmaf(w[1], p[1][hb[k] + f], y);
        y = __builtin_fmaf(w[2], p[2][hb[k] + f], y);
      }
      out[f] = y * scale;
    }
  }

  PlanarWeights<C> _w;
  int _prewarm_samples = 0;

  std::array<NanoLayer, kNumLayers> _p;
  std::vector<float> _head_w4;

  std::array<Ring, kNumLayers> _rings;
  Ring _head_ring;

  std::vector<float> _layer_in;
  std::vector<float> _head_sum;
  std::vector<float> _cond;
  std::vector<float> _head_out;
  int _stride = 0;
};

// =============================================================================
// Channels == 8 (A2 standard)
//
// Reproduces what a2_fast's Eigen expressions compute, including the per-tap
// partial that is summed into the running total only at the end of the tap, and
// the mixin's separate multiply and add. See the header comment for the full
// order.
// =============================================================================

/// Frames per conv tile. a2_fast's association needs the running total `z` and
/// the current tap's partial `t` live at once, which is 2 x 8 x (tile/4) vector
/// registers; the measured curve peaks at 8 and falls off at 16.
///
/// Both architectures agree here, which is worth saying because at C=3 they do
/// not. The ARMv7 ladder over 2/4/8/12/16/32 is
/// 1.049x/1.126x/1.255x/1.135x/0.899x/0.822x -- the same peak, reached from a
/// very different place: tile 8 at C=8 needs 35 registers on a machine with 16
/// and already spends 1.96 memory operations per FMA. It does not fit by a
/// factor of two and still wins, so what is being measured past that point is
/// how gracefully the spilling degrades, which no register count predicts.
constexpr int kFullTile = 8;
constexpr int kFullVecs = kFullTile / 4;

/// Independent head chains. The head is a 128-deep serial FMA chain per frame,
/// so it is latency-bound rather than throughput-bound; running eight chains at
/// once costs nothing in registers and nothing in exactness, because each chain
/// still covers its own frames in a2_fast's own order.
///
/// "Costs nothing in registers" is an AArch64 statement. Eight chains is eight
/// live accumulators plus the two weight vectors, which on ARMv7 is ten of
/// sixteen before a single input is loaded, and the sweep there chose one chain.
    #if defined(NAM_A2_PLANAR_A32)
constexpr int kFullHeadVecs = 1;
    #else
constexpr int kFullHeadVecs = 8;
    #endif

class A2PlanarFull : public DSP
{
  static constexpr int C = 8;
  using Ring = PlanarRing<C>;

public:
  A2PlanarFull(const std::vector<float>& weights, double expected_sample_rate)
  : DSP(/*in_channels=*/1, /*out_channels=*/1, expected_sample_rate)
  , _w(parse_weights<C>(weights))
  , _prewarm_samples(planar_prewarm_samples())
  {
    // Head weights as C contiguous floats per tap, so a tap's whole weight row
    // is two vector loads addressed by lane.
    _head_w.assign(static_cast<size_t>(kHeadKernelSize) * C, 0.0f);
    for (int k = 0; k < kHeadKernelSize; k++)
      for (int b = 0; b < C; b++)
        _head_w[static_cast<size_t>(k) * C + b] = _w.head_w[k][b];

    // layer1x1 transposed: [i*C + j] is the weight from bottleneck j to output i,
    // so one output's whole row is two contiguous vector loads instead of eight
    // scalar broadcasts. Identical FMAs in an identical order -- only the route
    // the weight takes to the instruction changes.
    _l1x1t.assign(static_cast<size_t>(kNumLayers) * C * C, 0.0f);
    for (int li = 0; li < kNumLayers; li++)
      for (int i = 0; i < C; i++)
        for (int j = 0; j < C; j++)
          _l1x1t[(static_cast<size_t>(li) * C + i) * C + j] = _w.layers[li].l1x1_w[static_cast<size_t>(j) * C + i];
  }

  ~A2PlanarFull() override = default;

  int GetPrewarmSamples() override { return _prewarm_samples; }

  void process(NAM_SAMPLE** input, NAM_SAMPLE** output, const int num_frames) override
  {
    if (num_frames > GetMaxBufferSize())
      SetMaxBufferSize(num_frames);
    const int N = num_frames;

    const NAM_SAMPLE* in0 = input[0];
    NAM_SAMPLE* out0 = output[0];

    float* cond = _cond.data();
    for (int f = 0; f < N; f++)
      cond[f] = static_cast<float>(in0[f]);

    // Rechannel straight into layer 0's ring.
    _rings[0].prepare(N);
    for (int c = 0; c < C; c++)
    {
      float* dst = _rings[0].write_ptr(c);
      const float wc = _w.rechannel_w[c];
      int f = 0;
      for (; f + 4 <= N; f += 4)
        vst1q_f32(dst + f, vmulq_n_f32(vld1q_f32(cond + f), wc));
      for (; f < N; f++)
        dst[f] = wc * cond[f];
    }
    _rings[0].commit(N);

    // The layers accumulate head_sum straight into the head ring's write window,
    // which removes the block-sized copy the head would otherwise make out of a
    // scratch buffer -- the same trick as writing each residual into the next
    // layer's ring.
    _head_ring.prepare(N);
    float* hs[C];
    for (int c = 0; c < C; c++)
      hs[c] = _head_ring.write_ptr(c);

    // No memset: layer 0 writes head_sum rather than accumulating onto it.
    for (int li = 0; li < kNumLayers; li++)
      dispatch_layer(li, N, hs);

    _head_ring.commit(N);

    head_forward(N);

    const float* head_out = _head_out.data();
    for (int f = 0; f < N; f++)
      out0[f] = static_cast<NAM_SAMPLE>(head_out[f]);
  }

protected:
  void SetMaxBufferSize(const int maxBufferSize) override
  {
    DSP::SetMaxBufferSize(maxBufferSize);

    _stride = maxBufferSize;
    _layer_in.assign(static_cast<size_t>(C) * _stride, 0.0f);
    _cond.assign(static_cast<size_t>(maxBufferSize), 0.0f);
    _head_out.assign(static_cast<size_t>(maxBufferSize), 0.0f);

    for (int li = 0; li < kNumLayers; li++)
      _rings[li].reset(_w.layers[li].max_lookback, maxBufferSize);
    _head_ring.reset(kHeadKernelSize - 1, maxBufferSize);
  }

private:
  float* lin(int c) { return _layer_in.data() + static_cast<size_t>(c) * _stride; }

  void dispatch_layer(int li, int N, float* const* hs)
  {
    const bool store_head = (li == 0);
    const bool do_l1x1 = (li != kNumLayers - 1);

    if (_w.layers[li].kernel_size == 6)
    {
      if (store_head)
        layer_forward<6, true, true>(li, N, hs);
      else if (do_l1x1)
        layer_forward<6, false, true>(li, N, hs);
      else
        layer_forward<6, false, false>(li, N, hs);
    }
    else
    {
      if (store_head)
        layer_forward<15, true, true>(li, N, hs);
      else if (do_l1x1)
        layer_forward<15, false, true>(li, N, hs);
      else
        layer_forward<15, false, false>(li, N, hs);
    }
  }

  template <int K, bool StoreHead, bool DoL1x1>
  void layer_forward(int li, int N, float* const* hs)
  {
    Ring& R = _rings[li];
    const PlanarLayerWeights<C>& L = _w.layers[li];

    const float* h[C];
    for (int c = 0; c < C; c++)
      h[c] = R.plane(c);

    int tapb[K];
    for (int k = 0; k < K; k++)
      tapb[k] = R.tap((K - 1 - k) * L.dilation, N);

    Ring* next = nullptr;
    float* d[C];
    if (li + 1 < kNumLayers)
    {
      next = &_rings[li + 1];
      next->prepare(N);
      for (int c = 0; c < C; c++)
        d[c] = next->write_ptr(c);
    }
    else
    {
      for (int c = 0; c < C; c++)
        d[c] = lin(c);
    }

    const float* lt = _l1x1t.data() + static_cast<size_t>(li) * C * C;

    int f = 0;
    for (; f + kFullTile <= N; f += kFullTile)
      tile<K, kFullVecs, StoreHead, DoL1x1>(L, lt, h, tapb, f, d, hs);
    for (; f + 4 <= N; f += 4)
      tile<K, 1, StoreHead, DoL1x1>(L, lt, h, tapb, f, d, hs);
    for (; f < N; f++)
      frame_scalar<K, StoreHead, DoL1x1>(L, h, tapb, f, d, hs);

    if (next != nullptr)
      next->commit(N);
  }

  /// NVEC x 4 frames. `z` is the running total across taps and `t` the current
  /// tap's partial -- both live at once, because that separation *is* a2_fast's
  /// association. z never reaches memory.
  template <int K, int NVEC, bool StoreHead, bool DoL1x1>
  NAM_A2_PLANAR_INLINE void tile(const PlanarLayerWeights<C>& L, const float* lt, const float* const* h,
                                 const int (&tapb)[K], int f0, float* const* d, float* const* hs)
  {
    float32x4_t z[C][NVEC];
    const float32x4_t zero = vdupq_n_f32(0.0f);
    for (int i = 0; i < C; i++)
      for (int v = 0; v < NVEC; v++)
        z[i][v] = zero;

    const float* cw = L.conv_w.data();
    for (int k = 0; k < K; k++)
    {
      const float* wk = cw + static_cast<size_t>(k) * C * C;
      const int base = tapb[k] + f0;
      float32x4_t t[C][NVEC];

    #if defined(NAM_A2_PLANAR_A32)
      // ARMv7: plain loops over j and i, and the weight broadcast from memory
      // at the point of use.
      //
      // This is the same arithmetic in the same order as the AArch64 form
      // below, written the way a 16-register machine wants it, and the
      // difference it makes is not marginal. Over the reference render, with
      // the fold form here: 12.12 G instructions, 8.33 G memory accesses,
      // 10.58 G cycles, 68.7% of one core. With this one: 9.79 G, 6.34 G,
      // 8.53 G, 55.4%. Nearly a third of the memory traffic was the fold.
      //
      // The accumulators are the whole reason. 16 Q registers cannot hold
      // z[C][NVEC] and t[C][NVEC] at once at any useful tile width, so what
      // decides this kernel is not whether it spills -- it must -- but how well
      // the compiler schedules the traffic it cannot avoid. GCC does that far
      // better for a loop nest it can see the shape of than for a fold of
      // generic lambdas it has to decide to inline first. On AArch64, where the
      // accumulators fit, the same fold costs nothing and buys the by-lane
      // encoding, which is why both forms are here rather than one.
      //
      // Seeding j == 0 with a multiply instead of an FMA against zero is exact
      // either way and saves the zeroing pass, but it is not done here: it
      // needs j as a compile-time value, and having j as a loop counter is
      // worth more than the pass it would save.
      for (int i = 0; i < C; i++)
        for (int v = 0; v < NVEC; v++)
          t[i][v] = zero;

      for (int j = 0; j < C; j++)
        for (int v = 0; v < NVEC; v++)
        {
          const float32x4_t s = vld1q_f32(h[j] + base + 4 * v);
          for (int i = 0; i < C; i++)
            t[i][v] = vfmaq_f32(t[i][v], s, vld1q_dup_f32(&wk[j * C + i]));
        }
    #else
      // Input channel j is unrolled at compile time so that j == 0 can seed the
      // partial with a multiply instead of an FMA against zero. Both are one
      // rounding of w*x, so this is exact either way; it just saves the init.
      const auto do_j = [&](auto jc) {
        constexpr int j = decltype(jc)::value;
        const LaneWeights<C / 4> wv(wk + j * C); // W(0..C-1, j), contiguous
        const float* hp = h[j] + base;
        for (int v = 0; v < NVEC; v++)
        {
          const float32x4_t s = vld1q_f32(hp + 4 * v);
          const auto do_i = [&](auto ic) {
            constexpr int i = decltype(ic)::value;
            if constexpr (j == 0)
              t[i][v] = wv.template mul<i>(s);
            else
              t[i][v] = wv.template fma<i>(t[i][v], s);
          };
          [&]<int... Is>(std::integer_sequence<int, Is...>) {
            (do_i(std::integral_constant<int, Is>{}), ...);
          }(std::make_integer_sequence<int, C>{});
        }
      };
      [&]<int... Js>(std::integer_sequence<int, Js...>) {
        (do_j(std::integral_constant<int, Js>{}), ...);
      }(std::make_integer_sequence<int, C>{});
    #endif

      for (int i = 0; i < C; i++)
        for (int v = 0; v < NVEC; v++)
          z[i][v] = vaddq_f32(z[i][v], t[i][v]);
    }

    post<K, NVEC, StoreHead, DoL1x1>(L, lt, h, tapb[K - 1], f0, z, d, hs);
  }

  /// Everything after the conv: bias, mixin, LeakyReLU, head_sum, layer1x1
  /// residual -- in a2_fast's order, with the mixin's multiply and add rounded
  /// separately as Eigen rounds them.
  template <int K, int NVEC, bool StoreHead, bool DoL1x1>
  NAM_A2_PLANAR_INLINE void post(const PlanarLayerWeights<C>& L, const float* lt, const float* const* h, int last_tap,
                                 int f0, float32x4_t (&z)[C][NVEC], float* const* d, float* const* hs)
  {
    const float32x4_t zero = vdupq_n_f32(0.0f);
    const float32x4_t slope = vdupq_n_f32(kLeakySlope);
    const float* cond = _cond.data();

    float32x4_t cf[NVEC];
    for (int v = 0; v < NVEC; v++)
      cf[v] = vld1q_f32(cond + f0 + 4 * v);

    for (int i = 0; i < C; i++)
    {
      const float32x4_t b = vdupq_n_f32(L.conv_b[i]);
      const float m = L.mixin_w[i];
      for (int v = 0; v < NVEC; v++)
      {
        float32x4_t a = vaddq_f32(z[i][v], b);
        // Product then add, two roundings, as Eigen rounds them. round_now is
        // what stops the pair being contracted straight back into one vfma; see
        // its definition for which target needs it and why.
        a = vaddq_f32(a, round_now(vmulq_n_f32(cf[v], m)));
        z[i][v] = vbslq_f32(vcltq_f32(a, zero), vmulq_f32(a, slope), a);
      }
    }

    for (int i = 0; i < C; i++)
    {
      float* p = hs[i] + f0;
      for (int v = 0; v < NVEC; v++)
      {
        // The add against +0.0 in the store case is deliberate; see the note in
        // the nano kernel's post().
        if constexpr (StoreHead)
          vst1q_f32(p + 4 * v, vaddq_f32(zero, z[i][v]));
        else
          vst1q_f32(p + 4 * v, vaddq_f32(vld1q_f32(p + 4 * v), z[i][v]));
      }
    }

    if constexpr (DoL1x1)
    {
      for (int i = 0; i < C; i++)
      {
        const float32x4_t bi = vdupq_n_f32(L.l1x1_b[i]);
        const LaneWeights<C / 4> lw(lt + static_cast<size_t>(i) * C); // L(i, 0..C-1)
        for (int v = 0; v < NVEC; v++)
        {
          // u_i = sum over j in increasing order, from zero.
          float32x4_t u = lw.mul<0>(z[0][v]);
          u = lw.fma<1>(u, z[1][v]);
          u = lw.fma<2>(u, z[2][v]);
          u = lw.fma<3>(u, z[3][v]);
          u = lw.fma<4>(u, z[4][v]);
          u = lw.fma<5>(u, z[5][v]);
          u = lw.fma<6>(u, z[6][v]);
          u = lw.fma<7>(u, z[7][v]);
          const float32x4_t prev = vld1q_f32(h[i] + last_tap + f0 + 4 * v);
          vst1q_f32(d[i] + f0 + 4 * v, vaddq_f32(vaddq_f32(prev, u), bi));
        }
      }
    }
  }

  /// The last few frames of a block that is not a multiple of four. Same
  /// operation order as the vector path, one frame at a time.
  template <int K, bool StoreHead, bool DoL1x1>
  void frame_scalar(const PlanarLayerWeights<C>& L, const float* const* h, const int (&tapb)[K], int f, float* const* d,
                    float* const* hs)
  {
    float z[C];
    for (int i = 0; i < C; i++)
      z[i] = 0.0f;

    for (int k = 0; k < K; k++)
    {
      const float* wk = L.conv_w.data() + static_cast<size_t>(k) * C * C;
      float t[C];
      for (int i = 0; i < C; i++)
        t[i] = 0.0f;
      for (int j = 0; j < C; j++)
      {
        const float s = h[j][tapb[k] + f];
        for (int i = 0; i < C; i++)
          t[i] = __builtin_fmaf(wk[static_cast<size_t>(j) * C + i], s, t[i]);
      }
      for (int i = 0; i < C; i++)
        z[i] += t[i];
    }

    const float cf = _cond[f];
    for (int i = 0; i < C; i++)
    {
      float a = z[i] + L.conv_b[i];
      a = a + mul_rounded(L.mixin_w[i], cf); // product then add, two roundings
      z[i] = (a < 0.0f) ? a * kLeakySlope : a;
      if constexpr (StoreHead)
        hs[i][f] = 0.0f + z[i];
      else
        hs[i][f] = hs[i][f] + z[i];
    }

    if constexpr (DoL1x1)
    {
      for (int i = 0; i < C; i++)
      {
        float u = 0.0f;
        for (int j = 0; j < C; j++)
          u = __builtin_fmaf(L.l1x1_w[static_cast<size_t>(j) * C + i], z[j], u);
        d[i][f] = (h[i][tapb[K - 1] + f] + u) + L.l1x1_b[i];
      }
    }
  }

  /// Head rechannel: K=16, dilation 1, eight channels down to one, plus bias and
  /// scale. a2_fast runs 128 sequential FMAs per frame; here the same 128 FMAs
  /// cover four frames at a time, and kFullHeadVecs of those chains run
  /// independently.
  void head_forward(int N)
  {
    int hb[kHeadKernelSize];
    for (int k = 0; k < kHeadKernelSize; k++)
      hb[k] = _head_ring.tap(kHeadKernelSize - 1 - k, N);

    const float* p[C];
    for (int c = 0; c < C; c++)
      p[c] = _head_ring.plane(c);

    const float* hw = _head_w.data();
    const float scale = _w.head_scale;
    const float32x4_t bias = vdupq_n_f32(_w.head_b);
    float* out = _head_out.data();

    int f = 0;
    for (; f + 4 * kFullHeadVecs <= N; f += 4 * kFullHeadVecs)
    {
      float32x4_t y[kFullHeadVecs];
      for (int u = 0; u < kFullHeadVecs; u++)
        y[u] = bias;
      for (int k = 0; k < kHeadKernelSize; k++)
      {
        const LaneWeights<C / 4> w(hw + static_cast<size_t>(k) * C);
        const int base = hb[k] + f;
        for (int u = 0; u < kFullHeadVecs; u++)
        {
          const int o = base + 4 * u;
          y[u] = w.fma<0>(y[u], vld1q_f32(p[0] + o));
          y[u] = w.fma<1>(y[u], vld1q_f32(p[1] + o));
          y[u] = w.fma<2>(y[u], vld1q_f32(p[2] + o));
          y[u] = w.fma<3>(y[u], vld1q_f32(p[3] + o));
          y[u] = w.fma<4>(y[u], vld1q_f32(p[4] + o));
          y[u] = w.fma<5>(y[u], vld1q_f32(p[5] + o));
          y[u] = w.fma<6>(y[u], vld1q_f32(p[6] + o));
          y[u] = w.fma<7>(y[u], vld1q_f32(p[7] + o));
        }
      }
      for (int u = 0; u < kFullHeadVecs; u++)
        vst1q_f32(out + f + 4 * u, vmulq_n_f32(y[u], scale));
    }
    for (; f + 4 <= N; f += 4)
    {
      float32x4_t y = bias;
      for (int k = 0; k < kHeadKernelSize; k++)
      {
        const LaneWeights<C / 4> w(hw + static_cast<size_t>(k) * C);
        const int o = hb[k] + f;
        y = w.fma<0>(y, vld1q_f32(p[0] + o));
        y = w.fma<1>(y, vld1q_f32(p[1] + o));
        y = w.fma<2>(y, vld1q_f32(p[2] + o));
        y = w.fma<3>(y, vld1q_f32(p[3] + o));
        y = w.fma<4>(y, vld1q_f32(p[4] + o));
        y = w.fma<5>(y, vld1q_f32(p[5] + o));
        y = w.fma<6>(y, vld1q_f32(p[6] + o));
        y = w.fma<7>(y, vld1q_f32(p[7] + o));
      }
      vst1q_f32(out + f, vmulq_n_f32(y, scale));
    }
    for (; f < N; f++)
    {
      float y = _w.head_b;
      for (int k = 0; k < kHeadKernelSize; k++)
      {
        const float* wk = hw + static_cast<size_t>(k) * C;
        for (int b = 0; b < C; b++)
          y = __builtin_fmaf(wk[b], p[b][hb[k] + f], y);
      }
      out[f] = y * scale;
    }
  }

  PlanarWeights<C> _w;
  int _prewarm_samples = 0;

  std::vector<float> _head_w;
  std::vector<float> _l1x1t;

  std::array<Ring, kNumLayers> _rings;
  Ring _head_ring;

  std::vector<float> _layer_in;
  std::vector<float> _cond;
  std::vector<float> _head_out;
  int _stride = 0;
};

} // namespace

std::unique_ptr<DSP> create_a2_planar_model(int channels, std::vector<float> weights, double expected_sample_rate)
{
  if (channels == 3)
    return std::make_unique<A2PlanarNano>(weights, expected_sample_rate);
  if (channels == 8)
    return std::make_unique<A2PlanarFull>(weights, expected_sample_rate);
  return nullptr;
}

} // namespace a2_fast
} // namespace wavenet
} // namespace nam

  #endif // NAM_A2_PLANAR
#endif // NAM_ENABLE_A2_FAST
