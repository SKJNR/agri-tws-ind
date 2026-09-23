#!/bin/bash
# Task 42c — push the FILTERED history (data/ stripped, max blob 39.8MB) to GitHub.
# Remote main is already at 056b2dd (commit 3, SHA-preserved by the rewrite);
# all remaining pushes fast-forward. Commit-by-commit for request-size safety.
# PAT via env $GHPAT — never written to any tracked file.
set -uo pipefail
cd /home/z/my-project

REPO_PATH="SKJNR/A-Step-Ahead-of-Drought-Forecasting-Global-Water-Storage-Challenge-by-ITU"
AUTH_URL="https://SKJNR:${GHPAT}@github.com/${REPO_PATH}.git"
CLEAN_URL="https://github.com/${REPO_PATH}.git"

redact() { sed -E "s#${GHPAT}#***REDACTED***#g"; }

echo "=== [1/4] Re-add clean remote origin (filter-repo removed it) ==="
git remote remove origin 2>/dev/null
git remote add origin "$CLEAN_URL"
git config branch.main.remote origin
git config branch.main.merge refs/heads/main
git config http.postBuffer 524288000
git config http.version HTTP/1.1
git config http.lowSpeedLimit 0
git config http.lowSpeedTime 999999
git remote -v | head -2

echo "=== [2/4] Incremental push commits 4..24 ==="
FAILED=0; N=0; TOTAL=$(git rev-list --count main)
for c in $(git rev-list --reverse main); do
  N=$((N+1))
  if [ $N -le 3 ]; then continue; fi   # already on remote, SHA-identical
  TRIES=0; RC=1
  while [ $RC -ne 0 ] && [ $TRIES -lt 3 ]; do
    TRIES=$((TRIES+1))
    OUT=$(git push "$AUTH_URL" "$c:refs/heads/main" 2>&1 | redact)
    RC=$?
    if [ $RC -ne 0 ]; then
      echo "[$N/$TOTAL] attempt $TRIES failed at $c:"; echo "$OUT" | grep -E 'error|rejected' | head -2
      sleep $((TRIES*5))
    fi
  done
  if [ $RC -ne 0 ]; then echo "[$N/$TOTAL] PERMANENT FAIL at $c"; FAILED=1; break; fi
  echo "[$N/$TOTAL] OK $c"
done

if [ $FAILED -eq 0 ]; then
  echo "=== [3/4] Push github-release + tag v28-endgame ==="
  RC=1; TRIES=0
  while [ $RC -ne 0 ] && [ $TRIES -lt 3 ]; do
    TRIES=$((TRIES+1))
    OUT=$(git push "$AUTH_URL" github-release v28-endgame 2>&1 | redact)
    RC=$?
    [ $RC -ne 0 ] && { echo "attempt $TRIES failed:"; echo "$OUT" | grep -E 'error|rejected' | head -3; sleep $((TRIES*10)); }
  done
  echo "release+tag push rc=$RC"
  echo "=== [4/4] Verify remote refs ==="
  git ls-remote "$CLEAN_URL" 2>&1 | redact
else
  echo "PUSH INCOMPLETE — resume from the failed commit"
fi
