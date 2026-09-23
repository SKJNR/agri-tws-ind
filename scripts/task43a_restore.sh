#!/bin/bash
# Task 43a — 10TH SPLIT-BRAIN RECOVERY (reset ~08:25 Sep 10, detected 10:43).
# /home/z/my-project was wiped to a scaffold (fresh "Initial commit" .git, no
# scripts/download). Old pre-Task-42 workspace found NESTED at my-project/.
# Recovery sources:
#   files  <- /tmp/my-project (freshest: worklog w/ Task 42, task42 scripts)
#   .git   <- GitHub remote 2d639e3 (final filtered history, 797 files, verified last night)
#   nested pre-Task-42 workspace archived to /home/sync (not used, kept as backup)
# PAT via env $GHPAT — never written to any tracked file.
set -uo pipefail
W=/home/z/my-project
T=/tmp/my-project
PAT="${GHPAT:?GHPAT env var required}"
REPO_URL_CLEAN="https://github.com/SKJNR/A-Step-Ahead-of-Drought-Forecasting-Global-Water-Storage-Challenge-by-ITU.git"

echo "=== [1/6] Archive nested pre-Task-42 workspace ==="
if [ -d "$W/my-project" ]; then
  mv "$W/my-project" /home/sync/nested_workspace_pre_task42 && echo "archived -> /home/sync/nested_workspace_pre_task42"
else
  echo "no nested dir (already handled)"
fi

echo "=== [2/6] Clear scaffold artifacts ==="
rm -rf "$W/.git" "$W/.gitignore" "$W/.env" "$W/tool-results"

echo "=== [3/6] Restore files from /tmp mirror (skip big externals + zip) ==="
rsync -a --no-owner --no-group \
  --exclude 'data/external' \
  --exclude 'data/era5_extract' \
  --exclude 'data/era5_full.zip' \
  --exclude 'data/era5_grid.parquet' \
  --exclude 'data/gpcp_precip.mon.mean.nc' \
  --exclude 'download/*.zip' \
  --exclude 'tool-results' \
  --exclude '.initial_snapshot.json' \
  "$T/" "$W/"
echo "restored. download files: $(ls "$W/download" | wc -l), scripts: $(ls "$W/scripts" | wc -l)"

echo "=== [4/6] Restore .git from GitHub (final history 2d639e3) ==="
rm -rf /tmp/ghclone
git clone --quiet "https://SKJNR:${PAT}@github.com/SKJNR/A-Step-Ahead-of-Drought-Forecasting-Global-Water-Storage-Challenge-by-ITU.git" /tmp/ghclone 2>&1 | sed -E "s#${PAT}#***#g" | tail -2
git -C /tmp/ghclone remote set-url origin "$REPO_URL_CLEAN"
rm -rf "$W/.git"
mv /tmp/ghclone/.git "$W/.git"
rm -rf /tmp/ghclone

echo "=== [5/6] Git config ==="
cd "$W"
git config user.name "Jisoo"
git config user.email "jisoo@local"
git config core.fileMode false
git config http.postBuffer 524288000
git config http.version HTTP/1.1
git config http.lowSpeedLimit 0
git config http.lowSpeedTime 999999
git config branch.main.remote origin
git config branch.main.merge refs/heads/main

echo "=== [6/6] Verify ==="
echo "HEAD: $(git log --oneline -1)"
echo "tracked files: $(git ls-files | wc -l)  (expect 797)"
echo "refs: main=$(git rev-parse --short main) release=$(git rev-parse --short github-release) tag->$(git rev-parse --short v28-endgame^{commit})"
echo "--- status (expect only untracked junk, NO modifications) ---"
git status --short | head -10
echo "status line count: $(git status --short | wc -l)"
echo "--- canaries ---"
for f in download/FINAL2_SELECTION.md download/submission_v28_deccorr.csv download/submissions_manifest.md scripts/task42c_push_filtered.sh scripts/task40_zero_month_201812.py worklog.md data/"Train (1).csv"; do
  [ -f "$f" ] && echo "OK   $f" || echo "MISS $f"
done
echo "--- worklog sync vs /tmp ---"
md5sum "$W/worklog.md" "$T/worklog.md" | awk '{print $1}' | uniq -c
df -h / | tail -1
