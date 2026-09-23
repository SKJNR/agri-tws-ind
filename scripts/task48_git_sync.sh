#!/bin/bash
# Task 48: commit + push the Sep-14 closure state + AGRI-TWS-IND-v1 debate
# blackboard to the PRIVATE GitHub repo.
#
# USAGE:  GH_PAT=<token> bash scripts/task48_git_sync.sh
#   PAT passed ONLY via env var — never written to any tracked file, never
#   stored in .git/config, never pasted in chat. Fine-grained token with
#   Contents:write on this one repo, revoked after use, is the preferred form.
#
# Method (same as task45b): fresh clone from remote (authoritative) ->
# ADDITIVE overlay of the current workspace -> single commit -> fast-forward
# push. The stale local .git (pre-filter, no remote) is NOT used.
set -euo pipefail

REPO_URL_PATH="SKJNR/A-Step-Ahead-of-Drought-Forecasting-Global-Water-Storage-Challenge-by-ITU"
WS=/home/z/my-project/my-project
FRESH=/tmp/git_fresh_task48

if [ -z "${GH_PAT:-}" ]; then
  echo "GH_PAT not set. Provide it via env (never in chat), then run:"
  echo "  GH_PAT=<token> bash scripts/task48_git_sync.sh"
  exit 2
fi

echo "=== [1] fresh clone (PAT in URL only, not stored) ==="
rm -rf "$FRESH"
git -c credential.helper= clone --quiet \
  "https://${GH_PAT}@github.com/${REPO_URL_PATH}.git" "$FRESH"
cd "$FRESH"
git config user.name "Jisoo" && git config user.email "jisoo@local"
echo "remote main @ $(git rev-parse --short origin/main)"

echo "=== [2] ADDITIVE overlay (no --delete: never blindly remove repo content) ==="
rsync -a \
  --exclude '.git' --exclude 'data/' --exclude '*.nc' --exclude '*.npz' \
  --exclude 'download/*.bundle' \
  --exclude 'download/TWS_competition_package_2026-08-31.zip' \
  --exclude 'download/72h_code_package.zip' \
  --exclude 'tool-results' --exclude '__pycache__' \
  --exclude 'scripts/gdo_*' --exclude 'scripts/15a_era5_extract' \
  "$WS/" "$FRESH/"

# evidence that lives outside the nested workspace
mkdir -p "$FRESH/upload"
cp -f /home/z/my-project/upload/pasted_image_1789344207866.png \
      "$FRESH/upload/final_lb_screenshot_2026-09-14.png" 2>/dev/null || true

git add -A

echo "=== [3] commit ==="
git commit -q -m "Task 48 (2026-09-14): post-close closure + AGRI-TWS-IND-v1 debate blackboard

- FINAL_LB_READING.md: private-reveal read (v20c 0.631662555->0.689129487,
  v20a 0.633113636->0.694554086 private; leak transfers weakly; manual
  selection killed the auto-default v20 trap) + final-LB screenshot + OCR
- CLOSURE_README.md + TWS_CLOSURE_PACKAGE_2026-09-14.zip: full evidence
  export (final two md5-verified 0ebede07/6b6e3e41; P4 provenance all-clean;
  13th split-brain repair)
- worklog.md: full 47-task narrative (repo copy was stale at Task 40)
- DEBATE.md (R1-QWEN verbatim + R1-GLM reply) + DECISION_LOG.md (binding
  pre-registrations PR-1..PR-5, tombstones T1-T5) for the India retraining
  program AGRI-TWS-IND-v1" || {
  echo "NOTHING TO COMMIT — workspace already matches remote"; exit 0; }
git log --oneline -2

echo "=== [4] push (fast-forward) ==="
git push origin main

echo "=== [5] verify ==="
git ls-remote origin main
echo ""
echo "DONE. Optional release assets (web UI, zero credentials in chat):"
echo "  attach download/TWS_CLOSURE_PACKAGE_2026-09-14.zip and"
echo "  download/72h_code_package.zip to the v28-endgame release."
