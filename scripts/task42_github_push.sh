#!/bin/bash
# Task 42 — push the full repo to GitHub (user-provided fine-grained PAT via env $GHPAT).
# PAT is NEVER written to any file (script is tracked in git).
# Target: SKJNR/A-Step-Ahead-of-Drought-Forecasting-Global-Water-Storage-Challenge-by-ITU
#   (user's placeholder repo for this competition; remote main holds only a 75-byte stub
#    README with unrelated history -> force-push our full history over it).
set -uo pipefail
cd /home/z/my-project

REPO_PATH="SKJNR/A-Step-Ahead-of-Drought-Forecasting-Global-Water-Storage-Challenge-by-ITU"
CLEAN_URL="https://github.com/${REPO_PATH}.git"
AUTH_URL="https://SKJNR:${GHPAT}@github.com/${REPO_PATH}.git"

redact() { sed -E "s#${GHPAT}#***REDACTED***#g"; }

echo "=== [1/5] Local refs ==="
git rev-parse main github-release v28-endgame

echo "=== [2/5] Force-push main (24 commits, ~240MB) ==="
git push --force "$AUTH_URL" main 2>&1 | redact | tail -5
echo "main push exit: $?"

echo "=== [3/5] Push github-release branch + v28-endgame tag ==="
git push "$AUTH_URL" github-release v28-endgame 2>&1 | redact | tail -5
echo "release/tag push exit: $?"

echo "=== [4/5] Set clean remote origin (no token in config) ==="
git remote remove origin 2>/dev/null
git remote add origin "$CLEAN_URL"
git config branch.main.remote origin
git config branch.main.merge refs/heads/main
git remote -v

echo "=== [5/5] Verify remote refs (repo is public: anonymous read) ==="
git ls-remote "$CLEAN_URL" 2>&1 | redact
