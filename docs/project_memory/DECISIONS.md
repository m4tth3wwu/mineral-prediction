# Decisions (append-only)

## D-20260917-001
2026-09-17T11:30:07.968028+00:00. Continue P1-E with frozen prior stages, no empirical/real-data training. Preserve the original decision record at mineral_prediction/docs/project_memory/DECISIONS.md. Evidence: user request and archived attached task; numerical protocol: mineral_prediction/p1e_identifiability/20260917T112555Z_p1e_v1/PROTOCOL.json. Does not authorize future searches.

## D-20260917-002
2026-09-17T11:30:07.968028+00:00. Correct repository-root identification. Earlier baseline described mineral_prediction as repository root. git rev-parse --show-toplevel instead returns D:/code/ResearchPractice, with prefix mineral_prediction/. This does not affect loaded states or diagnostics. Canonical memory now at docs/project_memory; original history retained. All paths here are Git-root-relative. Source/protocol files from the numerical execution are preserved, not silently rewritten.

## D-20260917-003
2026-09-17T11:30:07.968028+00:00. Population BLOCKED under the predeclared conservative gate: old .001/71 endpoint has |ΔlogZ96→192|=.0005061547841709313 > .0001. Source: mineral_prediction/p1e_identifiability/20260917T112555Z_p1e_v1/GATES.json and STRUCTURE.json#/historical/0.001_71/fidelity. All other recorded gates pass. This failure does not prove that fresh population integration would fail; the conservative gate is a protocol choice. Do not change it after seeing results. A subsequent explicit protocol review can distinguish legacy-only failures from teacher/control failures; no new experiment authorized by this note.

## D-20260917-004
2026-09-17T11:30:07.968028+00:00. Keep existing Git allowlist and AGENTS unchanged. .gitignore lines 2/6 ignore new root docs and mineral_prediction additions. New files are local, ignored, not staged/committed/pushed; an empty tracked diff does not mean new work is committed. Add no exceptions automatically. Root AGENTS already exists; suggested handoff text is in current session.

## D-20260917-005
2026-09-17T11:38:38.577506+00:00. User explicitly requested GitHub push. Commit and push this reviewed P1-E delivery to existing origin/codex-refactor; include complete raw evidence and both canonical and historical memory. No force push, visibility change, new training or edits to frozen prior stages. Earlier local-only/no-push statements describe the preceding session, not the current authorization.
