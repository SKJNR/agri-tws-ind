#!/bin/bash
# Task 42b — retry push after HTTP 408 on the monolithic 240MB request.
# Fixes: postBuffer 500M + HTTP/1.1 + no low-speed abort + commit-by-commit
# incremental pushes (each HTTP request carries only one commit's delta).
# PAT via env $GHPAT — never written to any tracked file.
set -uo pipefail
cd /home/z/my-project

REPO_PATH="SKJNR/A-Step-Ahead-of-Drought-Forecasting-Global-Water-Storage-Challenge-by-ITU"
AUTH_URL="https://SKJNR:${GHPAT}@github.com/${REPO_PATH}.git"
CLEAN_URL="https://github.com/${REPO_PATH}.git"

redact() { sed -E "s#${GHPAT}#***REDACTED***#g"; }

echo "=== Robustness config ==="
git config http.postBuffer 524288000
git config http.version HTTP/1.1
git config http.lowSpeedLimit 0
git config http.lowSpeedTime 999999
git config core.fileMode false

echo "=== Payload estimate (pack size of full main history) ==="
git rev-list --objects main | git pack-objects --revs --stdout -q > /dev/null 2>&1 || true
echo "objects in main history: $(git rev-list --objects main | wc -l)"

echo "=== Incremental push: commit-by-commit ==="
FAILED=0
N=0
for c in $(git rev-list --reverse main); do
  N=$((N+1))
  if [ $N -eq 1 ]; then
    # first push replaces the remote stub-README main (unrelated history) -> force ONCE
    OUT=$(git push --force "$AUTH_URL" "$c:refs/heads/main" 2>&1 | redact)
  else
    OUT=$(git push "$AUTH_URL" "$c:refs/heads/main" 2>&1 | redact)
  fi
  RC=$?
  if [ $RC -ne 0 ]; then
    echo "[$N/$?] FAIL at $c:"; echo "$OUT" | tail -3
    FAILED=1
    break
  else
    echo "[$N/24] OK $c"
  fi
done

if [ $FAILED -eq 0 ]; then
  echo "=== Push github-release + tag v28-endgame ==="
  git push "$AUTH_URL" github-release v28-endgame 2>&1 | redact | tail -4
  echo "=== Verify remote refs ==="
  git ls-remote "$CLEAN_URL" 2>&1 | redact
else
  echo "INCREMENTAL PUSH FAILED — see output above"
fi
