Linear convolution and channel mapping
======================================

``Linear`` supports positive ``in_channels`` and ``out_channels`` counts when
those counts are equal or either count is one. Both fields default to one when
omitted from the model configuration. Unequal counts greater than one (for
example, 2 inputs and 3 outputs) are rejected when constructing/loading the
model.

.. versionchanged:: 0.5.5

   Added one-to-many and many-to-one convolution with separate impulse
   responses. Unequal channel counts greater than one are rejected.

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
