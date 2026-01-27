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
