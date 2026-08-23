---
name: release-check
description: "Assess a repository against a release checklist or public release-guarantees document, and when fixes are explicitly requested, remediate deficiencies using two coordinated flights of subagents: evidence gathering, then isolated-worktree fixes merged into a user-named development integration branch. Use when asked to qualify, prepare, harden, audit, or make a codebase ready for a release without publishing the release."
---

# Release Check

Qualify a release with evidence, then repair reasonable deficiencies when the
request authorizes fixes, without weakening the release contract. Keep all work
on the user-named development integration branch. This workflow must never
merge to a branch literally named `main`.

## Establish scope and branch roles

1. Read repository instructions and inspect status, remotes, branches,
   worktrees, version sources, tags, and CI configuration. Do not alter the
   user's original checkout.
2. Find the governing criteria in this order:
   - a path or checklist supplied by the user;
   - `docs/release_guarantees.rst`;
   - `RELEASE_GUARANTEES.md`, `RELEASING.md`, or equivalent project docs;
   - a checklist explicitly derived with the user when no contract exists.
3. Determine whether the request is assessment-only or explicitly authorizes
   remediation. Assessment-only requests must not create branches or commits.
4. Determine and record `target_version`, the base ref, immutable `base_sha`,
   and, for remediation, the requested integration ref. Distinguish these
   branch roles:
   - **base branch**: the branch from which qualification starts;
   - **integration branch**: the user-named development branch that receives
     all qualification fixes, such as `release-check`;
   - **work branches**: short-lived branches in private worktrees used by
     remediation agents.
5. Resolve identity from the request and repository evidence. Ask if the target
   version, base, or integration branch remains ambiguous. Reject `main` as the
   integration target rather than silently substituting another branch.
6. If the intended release includes uncommitted changes, stop and ask for an
   immutable committed revision or explicit exclusion of those changes. Do not
   silently omit or capture them.
7. Define `qualification_start_sha`, the immutable commit Flight 1 assesses.
   For a new integration branch it equals `base_sha`; for an existing
   integration branch it is that branch's current `HEAD`; for assessment-only
   work it is the commit selected by the user or current repository context.
8. Preserve all pre-existing user changes. Do not stash, discard, commit, move,
   reset, or clean them.
9. Treat a draft guarantee policy as candidate criteria only. If it excludes
   the target version, do not change its coverage or claim the version is
   guaranteed without explicit user authorization.

## Plan the qualification

Create a plan with separate assessment, remediation, integration, and final
verification stages. Convert the governing document into atomic criteria with
stable identifiers. Include applicable criteria for:

- versioning, release scope, and deprecations;
- formatting, compiler warnings, and source hygiene;
- public API and user documentation;
- clean Debug and Release builds, supported build variants, and tests;
- supported platforms, toolchains, architectures, and downstream consumers;
- model/file compatibility and DSP output correctness;
- real-time safety and thread behavior;
- performance and memory regressions;
- dependencies, licenses, and source reproducibility; and
- tag, artifact, and publication readiness.

Do not create or fail a criterion because a changelog or release-notes file is
absent from the repository. This project uses GitHub-generated release notes;
their generation and publication happen outside the pre-release repository
qualification unless the user explicitly requests a hosted-release check.

Do not push, tag, create a pull request or hosted release, dispatch remote
workflows, upload packages or artifacts, deploy, sign or notarize artifacts, or
update downstream repositories unless the user separately authorizes those
external changes.

## Flight 1: independent assessment

Use subagents for a repository-read-only first flight. Before dispatch, create
one disposable detached worktree per assessor at `qualification_start_sha`.
Place build and test outputs in separately recorded temporary directories
outside the worktree wherever the tools allow it. Forbid tracked changes,
commits, branch/ref changes, and remote mutations. Partition criteria into
bounded, non-overlapping groups and adapt the number of groups to the available
concurrency. A useful partition is:

1. API documentation, user docs, versioning, and release metadata.
2. builds, CI, unit/integration tests, sanitizers, and consumer compatibility.
3. model compatibility, DSP correctness, real-time safety, and performance.
4. dependencies, licensing, reproducibility, and publishing readiness.

Give each agent its worktree path, target version, frozen qualification identity,
criteria IDs, and integration branch. Require this result for every criterion:

- status: `pass`, `fail`, `unknown`, or `not-applicable`;
- concise evidence with file paths, line numbers, commands, or test output;
- the exact deficiency when status is not `pass`;
- a proposed remediation and its validation command;
- dependencies, risk, and whether the environment can verify it.

Require each assessor to finish with `git status --porcelain` evidence and no
tracked or untracked changes. Before cleanup, inventory ignored files and
delete only explicitly recorded outputs created by the skill after resolving
their exact paths. Never use a broad clean command. Remove only the recorded
disposable worktrees, without force, after findings and outputs are captured.
If unexplained files remain, preserve the worktree and report it.

Require execution evidence where practical. The existence of a test, workflow,
or documentation setting is not proof that it passes. An unavailable platform
or service is `unknown`, not `pass`.

## Consolidate and decide

Wait for all assessment agents, resolve contradictory findings against primary
evidence, and create one qualification ledger containing every criterion.
Summarize the ledger for the user before remediation. For audit, assess, review,
or qualify-only requests, stop here. Continue into remediation only when the
user explicitly asked to fix, prepare, harden, or make the release ready.

Before remediation, inspect the worktree list. Create a dedicated integration
worktree at an explicit recorded path: create a new integration branch at
`base_sha`, or attach the existing integration branch at
`qualification_start_sha` without overwriting it. If that ref is already
checked out in a user-owned worktree, do not modify or detach it; ask the user
to release it or explicitly authorize use of that exact worktree. Perform every
coordinator branch, commit, and merge operation inside the recorded integration
worktree, never the original checkout.

Classify failed and unknown criteria as:

- **fix now**: bounded repository work directly needed to meet the criterion;
- **verify later**: requires an unavailable platform, credential, service, or
  long-running qualification environment;
- **needs decision**: requires a compatibility, support, versioning, or policy
  choice that would materially change the release contract;
- **not reasonable for this pass**: disproportionate or outside the requested
  release scope.

Never edit the contract merely to convert a failure into a pass. Never suppress
a warning, weaken a test, widen a numerical tolerance, remove platform support,
or relabel an unknown as passing without a technically justified decision.

Order remediation by dependency. Prefer:

1. version/scope decisions and build blockers;
2. correctness, compatibility, and real-time defects;
3. build/test/CI coverage needed to prove those fixes;
4. documentation and release metadata;
5. formatting and final hygiene.

## Flight 2: isolated remediation

Dispatch independent `fix now` groups to subagents. Use follow-up flights for
work that depends on earlier changes rather than forcing dependent tasks into
parallel execution.

For every agent that edits files:

1. Create a uniquely named work branch from the current integration-branch
   head and attach it to a private worktree in an explicit temporary path.
   Avoid branch names nested beneath the exact integration-branch ref.
2. Tell the agent to work only inside that worktree and only on its assigned
   deficiencies. Remind it that other agents are working concurrently.
3. Require focused verification, a coherent commit, the commit SHA, changed
   files, commands run, results, and remaining limitations.
4. Instruct the agent not to merge, push, rebase the integration branch, or
   modify another worktree.

The coordinator owns integration. Review each diff and verification result.
Immediately before every merge, assert that the command is running in the
recorded integration worktree, that its symbolic branch is the exact recorded
integration ref, and that the branch is not `main`. Then merge the work branch
in dependency order. Resolve conflicts deliberately; do not discard concurrent
work. After each merge, run the smallest useful cross-check before integrating
dependent work. Remove only skill-created worktrees, without force. Delete a
work branch only after its commit is verified as an ancestor of integration
`HEAD` and its worktree is clean.

## Final qualification

After all reasonable fixes are integrated:

1. Record integration `HEAD`, create a fresh detached final-verification
   worktree at that commit, and re-run the complete locally available
   qualification suite there. Put build and test outputs in separately recorded
   temporary directories outside the worktree wherever possible. Use clean
   builds rather than relying only on incremental artifacts. Do not reset or
   clean an existing checkout.
2. Re-check formatting, documentation coverage, version consistency, repository
   cleanliness, submodule state, and the diff from the frozen `base_sha`.
3. Reconcile every criterion in the ledger with final evidence.
4. Leave unavailable checks as `unknown` and policy choices as unresolved; do
   not claim full qualification while either remains material.
5. Confirm no merge was made to `main`, no remote mutation occurred, and the
   final-verification worktree has no tracked or untracked changes. Inventory
   ignored files and remove only exact, recorded skill-created outputs. Remove
   only recorded skill-created worktrees and never force their removal; preserve
   and report any worktree containing unexplained files.

Report:

- target version, base branch, and integration branch;
- criteria passed, fixed, unknown, not applicable, and still failed;
- commits integrated and the verification performed;
- guarantees the release can substantiate and guarantees it cannot yet make;
- user decisions or external checks still required; and
- the exact next action, while leaving the integration branch ready for review.
