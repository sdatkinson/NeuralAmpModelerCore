# Ticket Notes: Sequential `.nam` Models

Context for issue #142 and branch `142-sequential`.

## Current State

This branch currently has two commits:

- `96526b3 Add sequential NAM model wrapper`
- `50cd1c3 Add sequential weights version handling`

The implementation supports a `sequential` / `Sequential` model wrapper in Core.
The first implementation chose a nested child-model format where each child in
`config.models` is a complete `.nam` model object carrying its own
`architecture`, `config`, `weights`, and `sample_rate`.

The follow-up commit added `config.weights_version`:

- Missing `weights_version` means legacy version 1: top-level concatenated
  weights. This currently errors as deprecated / unsupported.
- `weights_version: 2` means nested child model weights inside `config.models`.

## Versioning Concern

The main unresolved concern is whether the `.nam` file `version` is being asked
to do too much.

A global file version currently seems to mix several separate ideas:

- Top-level envelope schema, e.g. `version`, `architecture`, `config`, `weights`,
  `metadata`.
- Reader capability, e.g. whether Core knows how to instantiate
  `architecture: "sequential"`.
- Architecture-specific config schemas, e.g. WaveNet layer fields versus Linear
  fields.
- Producer / trainer release version.

Those axes can change independently. Sequential support is a new high-level
architecture capability, but it does not change the existing WaveNet config or
weight layout.

## Working Recommendation

Treat `.nam` file `version` as the minimum reader capability required by that
specific file, not as the current trainer/Core release version.

Implications:

- Do not bump every exported WaveNet merely because Sequential exists.
- Existing WaveNet files should keep the same file version if their serialized
  shape did not change.
- Sequential files should use a version/capability that indicates they require
  a reader that understands the `sequential` architecture.
- A reader that sees an unknown architecture should fail because the
  architecture is unsupported, not because unrelated model types use newer Core
  capabilities.

## Possible Future Shape

A more explicit schema could separate these concepts, for example:

```json
{
  "version": "0.5.4",
  "architecture": "WaveNet",
  "config_version": "wavenet-2",
  "config": {}
}
```

or:

```json
{
  "requires": ["architecture:sequential", "weights:nested-models"]
}
```

This would be more precise than one global semver describing every part of every
model.

## Open Decision

If we decide that no legacy Sequential models exist in the wild, consider
removing `config.weights_version` and defining only one Sequential format:

- `architecture: "sequential"`
- `config.models` contains complete child `.nam` model objects.
- No top-level `weights`.
- No top-level `sample_rate`; child sample rates must be compatible.

In that case, document that Sequential files require the file version that
introduces Sequential, but do not force unrelated WaveNet exports to use that
newer version.
