#!/bin/bash
# Task 47 — Closure export (2026-09-14)
# 1) 13th split-brain repair: additive sync /tmp/my-project -> /home nested workspace
# 2) Append Task 47 to the worklog (guarded against double-append)
# 3) Build TWS_CLOSURE_PACKAGE_2026-09-14.zip (curated, md5-recorded) and mirror it to /tmp
set -uo pipefail

SRC=/tmp/my-project
DST=/home/z/my-project/my-project
ROOT=/home/z/my-project
STAGE=/tmp/closure_stage_20260914
ZIPBASE=$DST/download/TWS_CLOSURE_PACKAGE_2026-09-14

echo "== 1) 13th split-brain repair: additive sync $SRC -> $DST =="
rsync -a "$SRC/download/" "$DST/download/"
rsync -a "$SRC/scripts/" "$DST/scripts/"
cp -f "$SRC/worklog.md" "$DST/worklog.md"
echo "   /home worklog lines now: $(wc -l < "$DST/worklog.md")"

# mirror the two new closure docs into /tmp (durability protocol)
cp -f "$DST/download/FINAL_LB_READING.md" "$SRC/download/" 2>/dev/null || true
cp -f "$DST/download/CLOSURE_README.md"   "$SRC/download/" 2>/dev/null || true

echo "== 2) append Task 47 to worklog (guarded) =="
if ! grep -q "Task ID: 47" "$DST/worklog.md"; then
cat >> "$DST/worklog.md" <<'EOF'

---
Task ID: 47
Agent: Super Z (main agent)
Task: Post-close session. User pasted the final-LB screenshot (own
submissions page showing the UNSELECTED v20c/v20a PRIVATE scores) with "we
failed just because i have not selected v20". Read the screenshot, verify
final-file integrity, resolve the pending P4 provenance items, export the
closure evidence package.

Work Log:
- SCREENSHOT READ (VLM OCR x2, /home/z/my-project/scripts/final_lb_ocr.json
  + final_lb_ocr2.json; source upload/pasted_image_1789344207866.png,
  1502x187, exactly 2 rows): v20c public 0.631662555 -> PRIVATE
  0.689129487; v20a public 0.633113636 -> PRIVATE 0.694554086. Leak
  public->private degradation +0.0575/+0.0615: the ~0.034 public advantage
  over the clean floor 0.666 collapses on the private 70%. Our final score
  (best private of the SELECTED v28+v21a) is NOT in the crop; paste-back
  requested.
- FINAL FILE INTEGRITY: v28 md5 0ebede07887d1efed869dfd7c55e2f0d and v21a
  md5 6b6e3e41c25317a68089e6b9ca707c05 — both match FINAL2_SELECTION.md;
  both 280,962 lines. Selected two intact and byte-clean.
- P4 PROVENANCE (2nd external review pending items — RESOLVED):
  SPEI_01(t+1) = native competition columns (Test (2).csv SPEI_01_t etc.;
  "(t+1)" = internal naming for the organizers' target-month covariates) —
  LEGAL; fast_resid(t) = the model's own fast-state residual from observed
  TWS anchor history (no LB feedback / masked inference / future obs) —
  LEGAL; nino34 = ENSO prior for pre-registered gates only, never a
  feature, corrections disclosed as measured public-LB calibration —
  HONEST; ERA5 absent from build_v24_compliant_k0.py, v22 splice never
  selected — EXCLUDED; coordinates removed per the 19-Aug ruling
  (17-column compliant LGBM, no raw lat/lon; cell codes = bookkeeping
  only) — CLEAN.
- 13TH SPLIT-BRAIN: /home was stale again (worklog 1001 lines / Task-40
  era, task42-46 scripts absent); /tmp freshest (Task 46, 1378 lines).
  Re-synced additive /tmp -> /home in this script before packaging.
- CLOSURE EXPORT: download/TWS_CLOSURE_PACKAGE_2026-09-14.zip — the final
  two CSVs (md5-verified) + rollback v27 + v20c as labeled AUDIT EVIDENCE
  (never selected), canonical docs (FINAL2, manifest, REPORT v2,
  72H_PACKAGE, MASTER_HANDOFF, HAT_PROTOCOL, reviews, FINAL_LB_READING),
  SHAP/CodeCarbon evidence, the screenshot + OCR, key scripts, worklog.
- Reply delivered: leak private-collapse read; two-branch v20
  counterfactual (top-20 -> announced review -> DQ+ban+2000pts per the
  organizer Sep-9 close-out email; outside top-20 -> unbankable mid-board
  number); the user's manual selection = the act that killed the
  auto-default v20c+v20a trap; cascade live until 4 Oct. Requested
  paste-backs: v28/v21a private rows, final position, top-20 board,
  report-upload confirmation.

Stage Summary:
- Final two verified intact; all five P4 items resolved CLEAN; closure
  package exported outside the session. Key numbers: v20c private
  0.689129487, v20a private 0.694554086 (the leak transfers weakly to
  private). Our final score pending paste. Winners + final LB by 4 Oct;
  top-20 code review running.
EOF
  cp -f "$DST/worklog.md" "$SRC/worklog.md"
  echo "   appended; /tmp worklog mirrored."
else
  echo "   Task 47 already present, skip append."
fi

echo "== 3) stage + build ZIP =="
rm -rf "$STAGE"
mkdir -p "$STAGE"/submissions "$STAGE"/docs "$STAGE"/evidence "$STAGE"/scripts

cp "$DST/download/submission_v28_deccorr.csv" "$STAGE/submissions/"
cp "$DST/download/submission_v21a.csv"        "$STAGE/submissions/"
cp "$DST/download/submission_v27_2corr.csv"   "$STAGE/submissions/"
cp "$DST/download/submission_v20c.csv"        "$STAGE/submissions/"

for f in FINAL2_SELECTION.md submissions_manifest.md REPORT_DRAFT.md \
         72H_PACKAGE.md MASTER_HANDOFF.md HAT_PROTOCOL.md \
         ENDGAME_TOP20_MEMO.md METHODOLOGY_AUDIT.md \
         ADVERSARIAL_REVIEW_ROUND1.md ADVERSARIAL_REVIEW_ROUND2.md \
         ADVERSARIAL_REVIEW_ROUND3.md ADVERSARIAL_REVIEW_ROUND4.md \
         TECHNICAL_HANDOFF.md; do
  [ -f "$DST/download/$f" ] && cp "$DST/download/$f" "$STAGE/docs/"
done

for f in report_shap_carbon.txt report_shap_carbon.json emissions.csv; do
  for cand in "$DST/download/$f" "$DST/$f" "$SRC/download/$f" "$SRC/$f"; do
    [ -f "$cand" ] && cp "$cand" "$STAGE/evidence/" && break
  done
done
cp "$ROOT/upload/pasted_image_1789344207866.png" "$STAGE/evidence/final_lb_screenshot_2026-09-14.png"
[ -f "$ROOT/scripts/final_lb_ocr.json" ]  && cp "$ROOT/scripts/final_lb_ocr.json"  "$STAGE/evidence/"
[ -f "$ROOT/scripts/final_lb_ocr2.json" ] && cp "$ROOT/scripts/final_lb_ocr2.json" "$STAGE/evidence/"

for f in build_v24_compliant_k0.py build_v21_phaseA.py build_v21_phaseB.py \
         build_v21_phaseC.py task38_m201612_pair_v28.py task39_v28_close.py \
         task44_shap_codecarbon.py gdo_match.py gdo_quantify.py \
         leak_check.py spei_shift_verify.py task47_closure_export.sh; do
  [ -f "$DST/scripts/$f" ] && cp "$DST/scripts/$f" "$STAGE/scripts/"
done

cp "$DST/worklog.md" "$STAGE/worklog.md"
cp "$DST/download/CLOSURE_README.md" "$STAGE/CLOSURE_README.md"
cp "$DST/download/FINAL_LB_READING.md" "$STAGE/FINAL_LB_READING.md"
( cd "$STAGE" && md5sum submissions/*.csv > MD5SUMS.txt )

python3 - "$STAGE" "$ZIPBASE" <<'PYEOF'
import sys, shutil, os
stage, zipbase = sys.argv[1], sys.argv[2]
out = shutil.make_archive(zipbase, 'zip', root_dir=stage)
print("ZIP built:", out)
print("ZIP size: %.1f MB" % (os.path.getsize(out) / 1e6))
PYEOF

cp -f "${ZIPBASE}.zip" "$SRC/download/TWS_CLOSURE_PACKAGE_2026-09-14.zip"

echo "== verify =="
ls -la "${ZIPBASE}.zip"
python3 -m zipfile -l "${ZIPBASE}.zip" | head -45
echo "DONE."
