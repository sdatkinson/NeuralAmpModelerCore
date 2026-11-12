# Linear convolution benchmark

`bench_linear` measures callback-time distribution for synthetic linear models. It intentionally reports callback
times rather than only throughput: a convolution implementation can have a low average cost and still cause audio
dropouts when periodic work exceeds the callback deadline.

Build and run a Release benchmark with:

```sh
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release --target bench_linear -j
build-release/tools/bench_linear
```

The optional arguments are `taps`, `callback size`, `seconds`, `fft|direct`, and `cold|verify`. The `cold` mode
touches a 64 MiB buffer before each timed callback to expose cache-sensitive plans. `verify` renders an impulse
through the entire requested filter—including a one-minute filter—and compares every output sample with its tap.

## Dispatch tuning

The FFT path partitions uniformly. The first partition's worth of taps is convolved directly, which keeps the latency
at zero, and every partition behind it is the same size, so every transform is the same size and none of them is ever
the large one. Partitions after the first multiply the spectra of blocks that have already arrived, so that work is
spread across the callbacks that run no transform: only the forward transform, the first partition's multiplies and
the inverse transform land in the callback where a block completes.

| Taps | Implementation | Direct head | Partition |
|---:|---|---:|---:|
| up to 1,024 | direct | | |
| up to 2,048 | FFT | 256 | 256 |
| up to 8,192 | FFT | 512 | 512 |
| up to 48,000 | FFT | 1,024 | 1,024 |
| up to 240,000 | FFT | 1,024 | 1,024, then 8,192 from 16,384 taps |
| more | FFT | 1,024 | 1,024, then 16,384 from 32,768 taps |

Past a second of impulse response, uniform partitions would cost too many multiplies per sample, so a tail tier of
larger partitions takes over at twice its partition size. A tail block's result is first needed one tail block after it
completes, and all its work is spread across that gap. That includes its transforms: each is done as several transforms
of the uniform tier's size plus a combining pass, so no callback runs a large transform.

The FFT path keeps only the direct head's history: a few maximum-size callbacks beyond the head, moved back to the
start of its buffer when it fills. The direct implementation keeps the whole impulse response's worth of history, and
copies it back to the start of its buffer every 32 maximum-size callbacks.

The table was chosen on the 99th-percentile callback rather than the mean, with
[NAMBench](https://github.com/rikkus/NAMBench)'s `nam_ir_benchmark` on an Apple M2 and a Raspberry Pi 500 (Cortex-A76)
at 16- to 256-sample callbacks:

- Direct convolution through 1,024 taps. Forced to FFT, the uniform plan has the lower mean from 513 taps, but at 16-
  and 32-sample callbacks its transform callback costs more than direct convolution's flat profile until well past
  1,024 taps.
- A 256-tap partition up to 2,048 taps and 512 up to 8,192 keep the transform callback short while leaving few enough
  partitions that their spread multiplies stay cheap.
- Beyond 48,000 taps, the tail tier: without it, the uniform tier's multiplies per sample grow with the impulse
  response's length.

These are `bench_linear` results, FFT forced, 5 seconds each, on a Raspberry Pi 500 pinned to one core at 2.4 GHz.
"Uniform only" is this plan without its tail tier, for comparison:

| Taps | Callback | Uniform only mean / p99 / max (us) | This plan mean / p99 / max (us) |
|---:|---:|---|---|
| 240,000 | 32 | 25.74 / 46.63 / 50.20 | 13.30 / 51.70 / 74.63 |
| 1,200,000 | 32 | 118.05 / 145.94 / 165.19 | 20.23 / 60.30 / 84.22 |
| 2,880,000 | 16 | 156.86 / 190.78 / 248.83 | 15.30 / 56.11 / 97.00 |
| 2,880,000 | 32 | 295.66 / 367.61 / 457.45 | 27.52 / 64.00 / 88.70 |
| 2,880,000 | 256 | 2,348.98 / 3,303.18 / 3,379.68 | 211.22 / 370.96 / 390.61 |

On Apple silicon, run `bench_linear` with nothing else busy: it does not ask for a high-priority thread, so macOS may
schedule it on an efficiency core, which roughly doubles every figure in a run.

These numbers are hardware-specific. When changing the FFT backend, data layout, or dispatch table, rerun warm and
cold-cache tests on every supported architecture and optimize for the worst callback distribution, not only mean CPU.
