#!/bin/bash
# Task 41a (2026-09-10): 9th split-brain recovery — restore /home from /tmp archive
set -e
TMP=/tmp/my-project
HP=/home/z/my-project

echo "== [1/4] restore scripts/ (code + logs only; NO .nc/.npz binaries) =="
rsync -a --exclude='*.nc' --exclude='*.npz' "$TMP/scripts/" "$HP/scripts/"

echo "== [2/4] restore download/ (full ledger, minus competition zip) =="
rsync -a --exclude='TWS_competition_package_2026-08-31.zip' "$TMP/download/" "$HP/download/"

echo "== [3/4] restore root meta + upload =="
cp "$TMP/worklog.md" "$HP/worklog.md"
for f in leaderboard.json competition_main.json competition_data.json lb_now.html README.md; do
  [ -f "$TMP/$f" ] && cp "$TMP/$f" "$HP/$f" && echo "  restored $f"
done
cp -r "$TMP/upload/." "$HP/upload/" 2>/dev/null || true

echo "== [4/4] verify canonical set (md5, /home vs /tmp) =="
fail=0
for f in download/submissions_manifest.md download/FINAL2_SELECTION.md \
         download/HAT_PROTOCOL.md download/ENDGAME_TOP20_MEMO.md \
         download/submission_v28_deccorr.csv download/submission_v21a.csv \
         download/submission_probe_m201812_plus.csv \
         download/ADVERSARIAL_REVIEW_ROUND4.md worklog.md \
         scripts/task38_m201612_pair_v28.py scripts/task40_zero_month_201812.py; do
  h1=$(md5sum "$HP/$f" 2>/dev/null | cut -d' ' -f1)
  h2=$(md5sum "$TMP/$f" 2>/dev/null | cut -d' ' -f1)
  if [ -n "$h1" ] && [ "$h1" = "$h2" ]; then echo "  SYNC_OK  $f"; else echo "  FAIL     $f (home=$h1 tmp=$h2)"; fail=1; fi
done
echo "  csv files: $(ls "$HP"/download/*.csv | wc -l) | Task-40 in worklog: $(grep -c '^Task ID: 40' "$HP/worklog.md") | Task-41 entries: $(grep -c '^Task ID: 41' "$HP/worklog.md")"
if [ $fail -eq 0 ]; then echo "RESTORE VERIFIED — /home is current"; else echo "RESTORE INCOMPLETE"; exit 1; fi
