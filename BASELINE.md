# Baseline before Codex refactor

This repository preserves the current research source bytes before refactoring. Commit message: `baseline before codex refactor`; tag: `pre-codex-refactor`. No scientific algorithms, processing logic, parameters, or calculated results were changed to prepare this baseline.

## Scope

- `mineral_prediction/`: current PPP and ACAWLR models, experimental comparisons, source-review and audit scripts, plotting/report generators, tests, launch script and method notes.
- `scripts/`: existing ACAWLR implementations and legacy variants, retained without consolidation or deduplication.
- `baseline/source_manifest.json`: SHA-256 of each preserved original source file. `baseline/run_config.json` is an exact copy of the completed run configuration, not a newly chosen experiment.
- `baseline/environment_snapshot.json`: installed package versions observed in the existing research runtime; this is a snapshot, not a tested portable dependency lock.

## Local-only material

Raw datasets, geographic files, point tables, model/feature caches, intermediate and final experiment outputs, downloaded references, rendered reports, virtual environments, editor state and databases are excluded. Personal agent instructions, migration logs/manifests, disk-organization utilities, and unrelated course notes are also excluded. Existing local files are not deleted.

The GitHub repository is a SOURCE baseline, not a complete data backup or immediately self-contained reproduction package. Current PPP execution relies on local raw data and existing feature/support tables under ignored folders. Keep those local data and result directories separately. Do not fabricate or silently regenerate missing data during refactoring.

Existing generic absolute project/runtime paths remain unchanged to preserve the baseline. A subsequent portability refactor can address them on a separate branch. No personal user-profile paths or detected credentials are included in the selected files. Automated pattern checks are not a guarantee that all possible sensitive information has been detected.

## Rollback and subsequent work

Use the `pre-codex-refactor` tag to recover this source state. Create `codex-refactor` only after the baseline commit and tag have been verified on GitHub. Preserve scientific behavior and use the existing tests when later changing code. This preparation does not assert that a new model run or a portable clean-environment reproduction was performed.

The `.gitignore` intentionally uses an exact file allowlist. New source files must be explicitly reviewed and allowed. `.gitattributes` disables automatic line-ending conversion to preserve existing source bytes.
