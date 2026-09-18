Model Loading API
==================

WAV impulse responses
---------------------

The file-path overloads of ``nam::get_dsp`` also accept ``.wav`` files
(case-insensitive extension). A mono impulse response becomes a ``Linear``
model with one input, one output, no bias, and one coefficient per WAV sample.
A stereo impulse response creates a mono-to-stereo model: one input feeds two
independent filters, using the left and right WAV channels respectively.
Its receptive field is the number of WAV frames, and returned weights contain
all left-channel coefficients followed by all right-channel coefficients.
The loader preserves sample order and gain; it does not normalize, attenuate,
or trim the impulse response.

Supported files are little-endian RIFF/WAVE with 16-, 24-, or 32-bit signed PCM,
or 32-bit IEEE floating-point samples, including WAVE_FORMAT_EXTENSIBLE variants.
Files with more than two channels, compressed formats, empty or truncated data,
and non-finite samples are rejected with ``std::runtime_error``.

The WAV sample rate becomes the model's expected sample rate. Call ``Reset``
with the processing sample rate and maximum block size before processing audio;
``Linear`` resamples the impulse response when the processing rate differs.

.. code-block:: cpp

   auto ir = nam::get_dsp(std::filesystem::path("cabinet.wav"));
   ir->Reset(48000.0, 128);

The overload taking ``dspData&`` returns a reusable ``Linear`` configuration
and the decoded weights, with null metadata and the current supported NAM
file version. Loading options apply in the same way as for ``.nam`` files.

.. doxygennamespace:: nam
   :project: NeuralAmpModelerCore
   :members:
