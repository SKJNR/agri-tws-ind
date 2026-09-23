# REPO SWEEP CHECKLIST — standing periodic health sweep

Created by amendment H-1 (Qwen review of the Weather Backbone,
DECISION_LOG Addendum 9, 2026-09-23). Run this sweep after every
sandbox reset, before every push-heavy work session, and whenever a
courier asks for a repo health check (last full run: 2026-09-23).

## S1 — Secret scan (H-1, first item by design)

```
python3 program/scripts/secret_scan.py --all
```

MUST print `CLEAN`. On any hit: stop, do NOT push, remove the
credential, rotate it if it was ever live, and log the incident in
DECISION_LOG before continuing. The pre-commit hook
(`.git/hooks/pre-commit`) enforces `--staged` on every commit.

## S2 — Pre-commit hook present (wiped by every sandbox reset)

The hook lives inside `.git/` which resets destroy. Re-install after
any reset / recovery (playbook #4):

```
printf '#!/bin/sh\nexec python3 "$(git rev-parse --show-toplevel)/program/scripts/secret_scan.py" --staged\n' > .git/hooks/pre-commit && chmod +x .git/hooks/pre-commit
```

## S3 — Remote sync + branch state

- `git fetch origin`; `git status -sb` — main must be in sync with
  origin/main (ahead/behind = investigate before working).
- No untracked anomalies beyond KNOWN regenerables
  (`program/data/{raw,split,parquet,imd}/`, `my-project/` scaffold,
  `.git_seed_backup/` leftovers).

## S4 — Sealed custody (D#21)

- `program/data/manifests/sealed_contract/{ACCESS_LOG.md,
  README_UNSEAL_PROCEDURE.md}` present and committed.
- `git ls-files | grep -i gwl` must show ONLY manifests / contract
  docs — never a gwl data parquet or csv (open or sealed).
- Any sandbox reset WIPES the untracked sealed extract: on next need,
  re-supply from the founder's original arrival (receipt:
  `program/data/manifests/gwl_data_csv.receipt.json`) and re-verify
  the sealed hash against `gwl_split_manifest.json` BEFORE use.

## S5 — Visibility (founder's one rule: never public without asking)

API check: `private` must be `true`. If public: flag the founder
immediately (click-path: Settings → Danger Zone → Change visibility →
Private) and record the discrepancy in DECISION_LOG until flipped.
Standing record: Addendum 6; still public as of 2026-09-23.

## S6 — Regenerable data integrity

`program/data/parquet/*.parquet` (gitignored by policy): if present,
verify SHA256 against `program/data/manifests/*.json`; if absent
(post-reset), they are regenerable — `build_weather_backbone.py --run`
for the weather backbone (re-downloads IMD 1-deg via imdlib).

## S7 — Logs pushed

`download/DECISION_LOG.md` + `program/worklog.md` committed and
pushed; last worklog Task ID matches the session that just ended.
