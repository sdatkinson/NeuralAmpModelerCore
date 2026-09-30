Linear convolution and channel mapping
======================================

``Linear`` supports positive ``in_channels`` and ``out_channels`` counts when
those counts are equal or either count is one. Both fields default to one when
omitted from the model configuration. Unequal counts greater than one (for
example, 2 inputs and 3 outputs) are rejected when constructing/loading the
model.

.. versionchanged:: 0.6.0

   Breaking change: one-to-many and many-to-one models require separate
   impulse responses in the layout documented below. Models with unequal
   channel counts using a single shared response must be updated. Unequal
   channel counts greater than one are rejected instead of partially
   processing channels. Mono and equal-channel model layouts are unchanged.

Channel mapping
---------------

* **1 input, 1 output:** mono-to-mono convolution.
* **N inputs, N outputs:** each input feeds its corresponding output. All
  channels share the same impulse response and, if enabled, bias. There is no
  cross-channel mixing.
* **1 input, M outputs:** each output has its own impulse response applied to
  the single input: ``y[j] = convolve(h[j], x[0]) + b[j]``.
* **N inputs, 1 output:** each input has its own impulse response, and the
  filtered signals sum: ``y[0] = sum(convolve(h[i], x[i])) + b[0]``.
  There is no averaging or normalization. The output bias is added once.

Serialized weights
------------------

The configuration field ``receptive_field`` specifies the number of taps in
each impulse response at the training sample rate. The boolean field ``bias``
controls whether bias parameters are included.

For equal input/output counts, the weights contain one impulse response
followed by one optional shared bias. The parameter count is
``receptive_field + (bias ? 1 : 0)``.

For unequal supported counts, concatenate the impulse responses in channel
order, then append one bias per output if ``bias`` is true. For 1-to-M mapping,
the responses are in output-channel order; for N-to-1 mapping, they are in
input-channel order. Taps within each response are in causal order, starting
with the coefficient applied to the current sample. The parameter count is
``max(in_channels, out_channels) * receptive_field + (bias ? out_channels : 0)``.
An incorrect parameter count is rejected; a single shared response is not
implicitly broadcast for unequal counts.

For example, with two taps per response and bias enabled:

* **1 to 2:** ``[h0[0], h0[1], h1[0], h1[1], b0, b1]``.
* **2 to 1:** ``[h0[0], h0[1], h1[0], h1[1], b0]``.
* **2 to 2:** ``[h[0], h[1], b]``.

Both direct and FFT convolution use these rules. Resetting to a new processing
sample rate independently resamples each response from its original training
coefficients, leaves biases unchanged, and clears all convolution history.
Processing uses preallocated state; construction and reset may allocate.

Convolution engines
-------------------

The optional configuration field ``implementation`` selects how the
convolution is computed: ``"auto"`` (the default), ``"direct"`` or ``"fft"``.
Both engines have zero latency and produce the same result up to floating-point
rounding. ``"auto"`` chooses from the impulse-response length at the processing
sample rate, so resetting to a new sample rate can change the engine:

.. list-table::
   :header-rows: 1

   * - Taps
     - Engine
     - Partition
     - Tail partition
   * - up to 1,024
     - direct
     -
     -
   * - up to 2,048
     - FFT
     - 256
     -
   * - up to 8,192
     - FFT
     - 512
     -
   * - up to 48,000
     - FFT
     - 1,024
     -
   * - up to 240,000
     - FFT
     - 1,024
     - 8,192
   * - more
     - FFT
     - 1,024
     - 16,384

**Direct** convolution computes every output sample as a dot product over the
whole impulse response. Its cost per sample is proportional to the length and
the same in every callback.

**FFT** convolution splits the impulse response into three parts:

* **Direct head.** The first partition's worth of taps is convolved directly,
  sample by sample. This is what keeps the latency at zero.
* **Uniform partitions.** The taps after the head are cut into partitions of
  the same size ``B``, each convolved with a ``2B``-point real FFT of the
  input, overlap-added as the output is played. A transform runs once every
  ``B`` samples. When a block completes, only its forward transform, the first
  partition's multiplies and the inverse transform run in that callback; the
  other partitions multiply spectra of blocks that have already arrived, and
  that work is shared out across the callbacks in between.
* **Tail tier.** Beyond 48,000 taps, the uniform partitions stop at twice the
  tail partition size ``T``, and the rest of the impulse response is cut into
  partitions of ``T`` taps. A tail block's output is first needed ``T`` samples
  after the block completes, and all of its work is spread across that gap,
  including its transforms: each ``2T``-point transform is computed as several
  ``2B``-point transforms plus a combining pass, so no callback runs a large
  transform. Work is shared out by estimated cost among the callbacks that
  remain before the result is due; anything left when it is due is done then.

The FFT engine's output does not depend on how the input is divided into
callbacks. It keeps only the direct head's input history, plus a few
maximum-size callbacks. The direct engine keeps the whole impulse response's
input history and copies it back to the start of its buffer every 32
maximum-size callbacks.

Neither engine allocates while processing callbacks no longer than the
maximum buffer size given to ``Reset``. A longer callback makes the input
history grow, which allocates.

``tools/BENCHMARK_LINEAR.md`` records how the table was chosen and how to
measure it with ``tools/bench_linear``.
