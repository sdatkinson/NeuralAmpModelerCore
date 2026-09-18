DSP API
=======

Sample rates
------------

``DSP::GetExpectedSampleRate()`` reports the sample rate used to train the model
(in Hz), or ``-1.0`` when it is unknown. It does not change when ``Reset()`` is
called and is not necessarily the only rate at which a model can run.

``DSP::SupportsArbitrarySampleRate()`` returns ``false`` by default. Hosts can
query it before choosing whether to process directly at their current sample
rate or use external sample-rate conversion. A ``false`` result preserves the
existing assumption that the model should run at its training rate.

``Linear`` returns ``true`` and adapts its impulse response in
``Reset(sampleRate, maxBufferSize)``. Call this outside the audio callback before
processing at a new rate. Both direct and FFT convolution use the adapted
response; automatic engine selection uses its new length. Each reset starts
from the original trained coefficients and clears processing history, including
when returning to the training rate. Bias is unchanged.

Conversion uses AudioDSPTools' cubic interpolation with zero padding and a gain
factor of training rate divided by processing rate. This approximates the
original response; cubic interpolation is not a band-limited resampler, so
response accuracy, especially near Nyquist or when downsampling, can degrade.
If the training rate is unknown, Linear retains the original coefficients
because no conversion ratio can be determined. Processing rates must be finite
and positive. Other model architectures retain their existing behavior.

.. doxygenclass:: nam::DSP
   :project: NeuralAmpModelerCore
   :members:

.. doxygenclass:: nam::Buffer
   :project: NeuralAmpModelerCore
   :members:

.. doxygenclass:: nam::Linear
   :project: NeuralAmpModelerCore
   :members:

.. doxygenclass:: nam::Conv1x1
   :project: NeuralAmpModelerCore
   :members:

.. doxygenstruct:: nam::dspData
   :project: NeuralAmpModelerCore
   :members:

.. doxygenenum:: nam::EArchitectures
   :project: NeuralAmpModelerCore
