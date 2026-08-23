Release guarantees
==================

.. warning::

   This policy is a draft. It does not apply retroactively to version 0.5.5 or
   any earlier release. The first release covered by this policy will be named
   here before the policy takes effect.

This page describes what users can rely on when choosing a tagged
NeuralAmpModelerCore release. Guarantees apply only to stable releases listed
as covered in the `Version coverage`_ table. They do not apply to development
branches, arbitrary commits, or release candidates unless explicitly stated.

Guarantees for covered releases
-------------------------------

Documentation
^^^^^^^^^^^^^

* Every public API exposed by the library's headers is included in the API
  reference and documented.
* Documentation identifies parameters, return values, errors, ownership and
  lifetime requirements, and thread-safety or real-time-safety constraints
  where they affect correct use.
* The Doxygen and Sphinx documentation builds complete without errors, broken
  references, or undocumented-public-API warnings.
* User documentation is updated when a feature or compatibility change changes
  how consumers use the library.

Build and test quality
^^^^^^^^^^^^^^^^^^^^^^

* The exact tagged source passes the project's formatting check.
* Clean Debug and Release builds succeed with the toolchains and platforms
  listed for that release.
* The complete unit and integration test suites pass on the exact tagged
  source.
* Supported build variants, including optional optimized implementations, are
  tested in both their enabled and disabled configurations where applicable.
* At least one supported downstream consumer is built against the release
  candidate before the stable release is tagged.

Model compatibility and correctness
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* The release loads and processes every ``.nam`` file version identified as
  supported in :doc:`nam_file_version`.
* Representative models for every supported architecture are loaded and
  rendered as part of release qualification.
* Optimized and generic processing paths are checked for equivalent output
  within documented numerical tolerances.
* Reference audio comparisons protect against unintended changes to rendered
  output.
* Intentional compatibility or output changes that require consumer action are
  identified in the relevant user or API documentation.

Real-time behavior
^^^^^^^^^^^^^^^^^^

* Processing APIs documented as real-time safe perform no dynamic allocation
  after required initialization and prewarming have completed.
* Real-time-safety tests cover every supported model architecture for which the
  guarantee is made.
* Any operation that can allocate, lock, perform file access, or otherwise be
  unsuitable for an audio thread is identified in its API documentation.

Performance
^^^^^^^^^^^

* Release-candidate benchmarks are compared with the preceding stable release
  using a recorded test environment.
* Material regressions in processing speed, memory use, or initialization time
  are either resolved before release or explicitly accepted during release
  qualification.
* A performance optimization does not relax correctness or real-time-safety
  requirements unless the exception is explicitly documented.

Dependencies and source integrity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Dependency and submodule revisions are pinned, publicly retrievable, and
  sufficient for a recursive clean checkout to build without untracked local
  files.
* The release tag identifies the exact source revision that passed release
  qualification.
* Applicable dependency licenses and attributions are retained in the source
  distribution.

Limits of the guarantees
------------------------

Unless a particular release says otherwise, these guarantees do not promise:

* ABI compatibility between releases;
* bit-identical floating-point output across different processors or
  toolchains;
* support for platforms, toolchains, architectures, or build configurations
  not listed for that release; or
* stability of undocumented implementation details.

Version coverage
----------------

.. list-table:: Release guarantee coverage
   :header-rows: 1
   :widths: 20 25 55

   * - Core release
     - Policy status
     - Notes
   * - 0.5.5 and earlier
     - Not covered
     - These releases predate this policy and receive no retroactive guarantee
       under it.
   * - To be determined
     - Draft
     - The first covered release will be recorded here when the policy is
       adopted.

For each covered release, this table will also identify the tested platforms,
toolchains, processor architectures, build variants, and any explicitly
documented exceptions.
