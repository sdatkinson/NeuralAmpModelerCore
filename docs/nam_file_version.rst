``.nam`` file versions
======================

As more features are added to NAM, the version of the ``.nam`` file format will be incremented.
The general rules try to follow semantic versioning.
That means that any changes to file contents where an older version of 
NeuralAmpModelerCore is either not able to understand the contents or might
misunderstand them (e.g. new fields that old code is not looking for) will trigger a 
version bump communicating a breaking change (e.g. minor version while pre-v1.0.0; 
later, a major version bump).
Improvements where the model will be loaded correctly, but possibly with some incomplete
functionality will trigger a version bump communicating a non-breaking change 
(e.g. minor version or patch pre-v1.0.0).

Version history
---------------

The following table shows which versions of NeuralAmpModelerCore support which model file versions:

.. list-table:: Core Version Support Matrix
   :header-rows: 1
   :widths: 30 70

   * - Core Version
     - Latest fully-supported ``.nam`` file version
   * - 0.0.0
     - 0.5.1
   * - 0.2.0
     - 0.5.2
   * - 0.3.0
     - 0.5.3
   * - 0.4.0
     - 0.6.0
   * - 0.4.1
     - 0.7.0

Sequential models
-----------------

``Sequential`` is an architecture-specific composition of complete child NAM
models. It uses the existing top-level file envelope and does not introduce a
new file version::

  {
    "version": "0.7.0",
    "architecture": "Sequential",
    "config": {
      "models": [
        {"version": "0.7.0", "architecture": "WaveNet", "config": {}, "weights": [], "sample_rate": 48000},
        {"version": "0.7.0", "architecture": "Linear", "config": {}, "weights": [], "sample_rate": 48000}
      ]
    },
    "weights": [],
    "sample_rate": 48000
  }

The top-level ``weights`` array is empty because the wrapper has no parameters
of its own. Each entry in ``config.models`` is a complete NAM model carrying
its own architecture, configuration, and weights. The top-level and child
sample rates must be compatible.

Sequential files emitted by the trainer before Core support was completed used
bare child configs and concatenated top-level weights. Those files omitted each
child's architecture and are not supported by this canonical format.
