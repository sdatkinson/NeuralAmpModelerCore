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

Format changes
--------------

0.5.5
~~~~~

Adds support for ``"sequential"`` models. A sequential model is a serial
composition of other models.

Sequential model configs may include ``weights_version``:

* Missing ``weights_version`` means version 1: weights are concatenated at the
  top level of the sequential model. This form is deprecated.
* ``weights_version: 2`` means each child in ``config.models`` carries its own
  weights as part of a complete ``.nam`` model object. The top-level sequential
  object does not contain ``"weights"`` or ``"sample_rate"`` fields; each child
  model carries its own weights and sample rate, and child sample rates must be
  compatible.

For ``weights_version: 2``, sequential models use either ``config.models`` or
``config.layers`` to list child models in processing order. Each entry may be a
child model object directly or an object with a ``model`` field containing the
child model.
