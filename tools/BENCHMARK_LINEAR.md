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

The dispatch table was tuned on an Apple M1 at 48 kHz using the same `-O3` optimization level as the Release plugin.
The main constraint was the 666.67 microsecond deadline of a 32-sample callback. Candidate plans varied direct-head
sizes from 64 through 256 samples and maximum FFT partitions from 1,024 through 32,768 samples.

The selected FFT plans produced these representative 10-second results (the default dispatcher uses direct
convolution for the 1,024-tap case):

| Taps | Direct head | Maximum partition | Mean (us) | p99 (us) | Max (us) | Overruns |
|---:|---:|---:|---:|---:|---:|---:|
| 1,024 | 128 | 256 | 2.37 | 10.62 | 55.67 | 0 |
| 2,048 | 128 | 512 | 2.90 | 14.54 | 63.38 | 0 |
| 4,096 | 128 | 1,024 | 3.65 | 23.79 | 58.33 | 0 |
| 8,192 | 128 | 2,048 | 4.63 | 49.12 | 81.79 | 0 |
| 48,000 | 64 | 4,096 | 6.45 | 48.83 | 123.42 | 0 |
| 240,000 | 64 | 8,192 | 8.21 | 91.12 | 218.33 | 0 |
| 1,200,000 | 64 | 8,192 | 14.41 | 99.79 | 297.58 | 0 |
| 2,880,000 | 64 | 8,192 | 24.22 | 112.00 | 324.62 | 0 |

The old 1,024-sample uniform implementation measured about 2.7--3.1 ms at p99 for the 1,208,121-tap atmospheric
model and overran 32- and 64-sample callback deadlines. The non-uniform plan reduces that periodic burst by more than
an order of magnitude.

The table favors bounded callback time over the absolute minimum mean:

- A 64-sample direct head had lower p99 times for long filters; a 128-sample head reduced overhead for 2K--8K filters.
- A 16,384-sample maximum partition slightly reduced mean time for multi-second filters but roughly doubled p99 and
  produced deadline outliers. A 4,096 maximum kept transforms smaller but nearly doubled one-minute steady-state cost.
- Direct convolution through 1,024 taps had a flatter callback profile than FFT convolution at essentially the same
  mean cost, so `Auto` selects it for that range.

These numbers are hardware-specific. When changing the FFT backend, data layout, or dispatch table, rerun warm and
cold-cache tests on every supported architecture and optimize for the worst callback distribution, not only mean CPU.
