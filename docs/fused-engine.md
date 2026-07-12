# Fused NEON WaveNet engine

`NAM/wavenet/fused.{h,cpp}` — an AArch64 (Apple Silicon, and any ARMv8+ CPU)
fast path for the common "standard shape" WaveNet family. Enabled with the
`NAM_ENABLE_FUSED` compile definition (CMake option `NAM_ENABLE_FUSED`, ON by
default; also defined by the plugin's mac/iOS/win build configs).

## Why

Profiling the generic WaveNet on an Apple M2 (`wavenet_a1_standard.nam`,
48 kHz, 64-frame blocks) showed:

- **~65%** of wall time inside Eigen's dynamic-size GEMM machinery — and over
  a quarter of *that* was packing/blocking overhead rather than math. Eigen's
  GEMM also heap-allocates its packing buffers on the audio thread.
- **~8%** scalar fast-tanh, ~20% spread across ring-buffer writes,
  `setZero`, and the many per-layer buffer-to-buffer passes
  (conv → mixin-add → activation → head accumulate → 1x1 → residual add).

The matrices are tiny (16×16 at most per tap), so a general-purpose GEMM is
the wrong tool: a register-tiled NEON microkernel with compile-time channel
counts reaches ~90% of the core's fp32 FMA peak on these shapes, where Eigen
reaches ~40%. Two further strategies were measured and rejected:

- **Accelerate / AMX** (`cblas_sgemm`): ties the NEON kernel at these sizes
  (932 vs 934 ns for a 16-ch, k=3, 64-frame layer conv including the im2col
  gather) — the AMX only pulls ahead on much larger matrices. Not worth the
  framework dependency or the (unspecified) real-time behavior of BLAS.
- **BNNS / Metal / ANE**: dispatch latency and/or conversion overhead make
  them unsuitable for 64-sample (1.3 ms) real-time blocks.

## What it does

Per layer, the engine runs three passes over one L1-resident block buffer
instead of the generic path's ~8 passes + 4 Eigen GEMM calls:

1. **conv pass** — all K dilated-conv taps, conv bias, and the input mixin
   fused into one register-tiled kernel (`conv_block<Q>`, Q = channels/4).
   Accumulators for a 4-frame tile stay in NEON registers across all taps;
   weights stream through `vfmaq_laneq_f32`.
2. **activation pass** — vectorized NEON implementations for fast-tanh
   (same rational approximation as `activations::fast_tanh`; `vdivq_f32`
   measured *faster* than reciprocal-refinement on M-series because the FP
   divider is a separate unit), ReLU, LeakyReLU, Hardtanh, Softsign. Any
   other activation (exact Tanh, Sigmoid, PReLU, LUTs) calls the exact same
   `Activation` object the generic path would use, so semantics never
   diverge — including `enable_fast_tanh()` and LUT substitutions, which are
   resolved through the registry at model-construction time in both paths.
3. **tail pass** — head accumulation, layer1x1 GEMM + bias, and the residual
   add into the (in-place) layer input buffer, again register-tiled.

Additional structure:

- Conv history lives in **power-of-2 ring buffers with a mirrored tail**, so
  every tap read is a single contiguous span. The mirror is only refreshed
  for the (rare) blocks where a read actually wraps — constant, small
  per-block cost, no rewind `memmove` latency spikes.
- The layer1x1 of the final layer of the final array is skipped (its output
  is provably never consumed; the generic path computes and discards it).
- Everything is preallocated in `SetMaxBufferSize`; `process()` performs no
  heap allocation (verified by an allocation-tracking test).

## Supported shapes

The strict detector (`is_fused_shape`) routes a model here only when:

- `in_channels == 1`, no condition DSP, no post-stack head
- every layer array has `condition_size == 1`, `bottleneck == channels`,
  channels a multiple of 4 (≤ 32), all groups == 1, gating `none`, no FiLM,
  `head1x1` inactive, `layer1x1` groups == 1 (active or inactive)
- arbitrary kernel sizes (≤ 64), dilations, activations, head kernel
  size/bias, any number of layer arrays

This covers the classic A1 standard/lite family (16/8 and 8/4 channels,
kernel 3) and the A2 standard model (8 channels). A2 nano (3 channels) is
not a multiple of 4 and falls through to the existing `a2_fast` scalar path,
which is already excellent for it. Anything else (feather/nano 2-channel
arrays, gated/FiLM/grouped models) uses the generic path unchanged — the
fused engine never regresses unsupported models, it just declines them.

Dispatch order in `wavenet::create_config`: slimmable → **fused** → a2_fast
→ generic.

## Correctness

- `tools/test/test_fused.cpp` compares fused vs generic outputs (tolerance
  5e-5, the same as the a2_fast tests) across: A1 standard 16/8 with exact
  tanh and fast tanh, 8/4, channels 12/20/32, ReLU / LeakyReLU / Hardtanh /
  Softsign / Sigmoid, A2-like mixed kernel sizes with head kernel 16,
  layer1x1-inactive arrays, and block sizes 64/256/23 (the odd size
  exercises the remainder tiles and ring wrap paths). Plus detector
  negative tests and a zero-allocation real-time-safety test.
- End-to-end `render` comparison on the real trained models
  (`wavenet_a1_standard.nam`, `my_model.nam`): max sample difference ~1e-6,
  RMS error −127 dB relative to the signal. The only sources of deviation
  are FMA contraction/ordering inside the conv sums.

## Measured results (Apple M2, macOS, buffer = 64, 48 kHz)

| Model | generic | fused | speedup |
|---|---|---|---|
| `wavenet_a1_standard.nam` (16/8) | ~86 ms / 2 s (23x RT) | ~39 ms / 2 s (~51x RT) | **~2.2x** |
| `a2_standard.nam` (8 ch) | ~83 ms / 2 s (24x RT) | ~40 ms / 2 s (~50x RT) | **~2.1x** (1.8x vs `a2_fast`) |

The conv kernel itself runs at ~105 GFLOP/s fp32 — ~93% of a single M2
P-core's theoretical FMA peak — so the remaining per-block time is dominated
by the (irreducible at fp32) tanh evaluation and the memory traffic floor.

## Future directions considered

- **fp16 / bf16**: `FMLAL`-style fp16 multiplies with fp32 accumulation give
  the same 4 MACs/instruction as fp32 FMA on Apple cores — no compute win,
  only bandwidth (we are compute-bound). Full fp16 accumulation doubles
  throughput but the 11-bit mantissa is a real audio-quality risk across a
  48-term accumulation; not pursued.
- **SME/SME2 (M4+)**: the streaming-SVE matrix unit could substantially beat
  NEON on these GEMMs; needs M4 hardware to develop/validate.
- **Multithreading**: layer arrays are sequential, so parallelism would have
  to split channels within a layer; with 64-sample deadlines the sync jitter
  is a poor trade for a plugin. Apple's audio-workgroup API would be the
  right vehicle if ever needed.
