#!/bin/bash
# Task 45b: commit + push the Task-45 state (11th-split-brain recovery,
# FINAL2 checklist tick, worklog Task 45) to the PRIVATE GitHub repo.
#
# USAGE:  GH_PAT=<token> bash scripts/task45b_git_commit.sh
#   (PAT passed ONLY via env var — never written to any tracked file,
#    never stored in .git/config. Token stays in the invoking shell.)
#
# Method: fresh clone from remote (authoritative, filtered history, main
# @ 87f0918+ per Task-44's 72H doc) -> overlay the current workspace files
# -> single commit -> push (fast-forward). The stale local .git (pre-filter,
# no remote) is NOT used for the push.
set -euo pipefail

REPO_URL_PATH="SKJNR/A-Step-Ahead-of-Drought-Forecasting-Global-Water-Storage-Challenge-by-ITU"
WS=/home/z/my-project/my-project
FRESH=/tmp/git_fresh_task45

if [ -z "${GH_PAT:-}" ]; then
  echo "GH_PAT not set. Re-paste the PAT in chat, then run:"
  echo "  GH_PAT=<token> bash scripts/task45b_git_commit.sh"
  exit 2
fi

echo "=== [1] fresh clone (PAT in URL, not stored) ==="
rm -rf "$FRESH"
git -c credential.helper= clone --quiet \
  "https://${GH_PAT}@github.com/${REPO_URL_PATH}.git" "$FRESH"
cd "$FRESH"
git config user.name "Jisoo" && git config user.email "jisoo@local"
echo "remote main @ $(git rev-parse --short origin/main)"

echo "=== [2] overlay workspace (tracked-file surface only) ==="
rsync -a --delete \
  --exclude '.git' --exclude 'data/' --exclude '*.nc' --exclude '*.npz' \
  --exclude 'download/*.bundle' --exclude 'download/*.zip' --exclude 'tool-results' \
  --exclude 'scripts/gdo_*' --exclude 'scripts/15a_era5_extract' \
  --exclude 'emissions.csv' \
  "$WS/" "$FRESH/"
git add -A

echo "=== [3] commit ==="
git commit -q -m "docs: worklog Task 45 (11th split-brain recovery + external-AI gap reconciliation verdict), FINAL2 checklist report-box ticked; task45a/45b scripts" || {
  echo "NOTHING TO COMMIT — workspace already matches remote"; exit 0; }
git log --oneline -2

echo "=== [4] push (fast-forward) ==="
git push origin main
echo "=== [5] verify ==="
git ls-remote origin main
echo "DONE. New main tip above; /tmp mirror authoritative for files."
