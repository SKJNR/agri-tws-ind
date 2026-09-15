# program/data/ — data persistence policy

**Adopted 2026-09-15 (founder ruling; DECISION_LOG addendum). This file
is tracked in git on purpose — it survived the 10th reset only because
its text lived in DECISION_LOG. Never leave policy files untracked.**

## ARRIVAL=COMMIT (binding)

Any data file landing in the sandbox is committed — or Release-asseted
with its manifest committed in-tree — as the FIRST action after arrival,
BEFORE any pipeline work on it. No exceptions, no "I'll commit it after
the analysis."

## Size classes

| Class | Rule |
|---|---|
| ≤ 95 MB (gzipped) | Commit directly to `program/data/` |
| > 95 MB | GitHub **Release asset** on the private repo; manifest (filename, bytes, SHA256, provenance) committed in-tree under `program/data/manifests/` |

## Visibility

The repository stays **PRIVATE** permanently:
- IndiaAI/AIKosh participant licensing applies to SoulVision-era assets.
- LGD (GODL-India) data shares the same roof for simplicity.
- Competition (Zindi, CC-BY-SA 4.0) material carries attribution +
  share-alike obligations; public exposure is a separate, deliberate
  decision (see `github-release` branch) — never an accident.

**Founder's one rule: never make this repo public without asking first.**

## gwl_data.csv arrival flow (pre-staged)

upload → SHA256 receipt (gwl_handback_check.py) → **immediate commit**
(this policy) → D#21 physical split (open ≤2022 / sealed 2023+) →
Task-55 provenance spot-check vs India-WRIS raw → D1.1 work on open
rows only. Sealed remainder: locked dir, separate hash, logged access,
written unseal procedure.
