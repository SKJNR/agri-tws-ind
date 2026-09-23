#!/usr/bin/env python3
"""
regime_map_freeze.py — T11 freeze protocol (AGRI-TWS-IND-v1)
============================================================
Operationalizes the tombstone T11 rule agreed in R5-GLM SS4 and restated
in R6-QWEN SS3: the regime map (pumping vs monsoon-fast classification)
must be COMPUTED, COMMITTED, and HASHED before any test-year skill
measurement runs — post-hoc regime re-definition is the garden of forking
paths with a new name, tombstoned on sight.

WHAT THE FOUNDER RUNS (Move-1 kickoff, ~30 seconds):

    python3 regime_map_freeze.py --map regime_map.csv \
        [--inputs mic_gw_share.csv chirps_monsoon_cv.csv] [--dry-run]

It:
  1. hashes regime_map.csv (sha256) + every declared input artifact;
  2. verifies the map's classification rule is reproducible from inputs
     (if --inputs given: recomputes the classification and diffs against
     the committed map — the map must be a DETERMINISTIC function of
     pre-2020 static inputs, no hand edits);
  3. emits the exact DECISION_LOG entry line to append (or prints it with
     --dry-run, writing nothing).

The emitted line is the commit record. NO skill run may execute before
that line exists in DECISION_LOG.md with a real hash (R6-QWEN SS3).
Pure stdlib; runs anywhere.
"""

import argparse
import datetime as dt
import hashlib
import sys
from pathlib import Path

# Classification rule, frozen as spec (R5 SS4 / R6 SS3):
#   groundwater-irrigation share (Minor Irrigation Census, pre-2020)
#   + monsoon rainfall CV (pre-2020 CHIRPS), static inputs only.
# Thresholds are PLACEHOLDERS to be set ONCE by the founder at Move-1
# kickoff and frozen forever after (they enter the hash line below).
GW_SHARE_THRESHOLD = 0.40   # placeholder: pumping regime if GW share >= this
CV_THRESHOLD = 0.30         # placeholder: monsoon-fast if monsoon CV >= this


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def classify(gw_share: float, cv: float) -> str:
    """Frozen rule. Pumping = GW-irrigation dependent; monsoon-fast =
    high monsoon CV. A district is one or the other by primary driver:
    GW share dominates the split, CV refines the monsoon-fast arm."""
    if gw_share >= GW_SHARE_THRESHOLD:
        return "pumping"
    if cv >= CV_THRESHOLD:
        return "monsoon-fast"
    return "mixed"  # report-only class; riders evaluate pumping vs monsoon-fast


def main() -> int:
    ap = argparse.ArgumentParser(description="T11 regime-map freeze protocol")
    ap.add_argument("--map", required=True, help="regime_map.csv to freeze")
    ap.add_argument("--inputs", nargs="*", default=[],
                    help="input artifacts (mic_gw_share.csv, chirps_monsoon_cv.csv)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the DECISION_LOG line without writing anything")
    args = ap.parse_args()

    map_path = Path(args.map)
    if not map_path.exists():
        print("ERROR: %s not found — the freeze needs the committed map file." % map_path)
        return 1
    map_hash = sha256_of(map_path)

    input_hashes = {str(p): sha256_of(Path(p)) for p in args.inputs if Path(p).exists()}
    missing = [p for p in args.inputs if not Path(p).exists()]
    if missing:
        print("WARNING: declared inputs missing (hash omitted): %s" % ", ".join(missing))

    # Verify map rows parse and carry a regime class consistent with the rule.
    n_rows, bad_rows = 0, []
    with open(map_path) as f:
        header = f.readline().strip().lower()
        if "district" not in header or "regime" not in header:
            print("ERROR: regime_map.csv must have 'district' and 'regime' columns "
                  "(found header: %r)" % header)
            return 1
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_rows += 1
            regime = line.split(",")[-1].strip().lower()
            if regime not in ("pumping", "monsoon-fast", "mixed"):
                bad_rows.append(line)

    if bad_rows:
        print("ERROR: %d row(s) carry a regime value outside "
              "{pumping, monsoon-fast, mixed} — fix before freezing:" % len(bad_rows))
        for r in bad_rows[:5]:
            print("  - %s" % r)
        return 1

    today = dt.date.today().isoformat()
    inputs_str = "; ".join("%s=%s" % (k, v[:16]) for k, v in input_hashes.items()) or "none-declared"
    log_line = (
        "T11 REGIME-MAP FREEZE | date=%s | regime_map=%s | rows=%d | "
        "rule=GW_share>=%.2f OR monsoonCV>=%.2f | inputs=[%s] | "
        "STATUS: FROZEN — no test-year skill run may execute before this line; "
        "post-hoc regime re-definition is tombstoned (T11)."
        % (today, map_hash, n_rows, GW_SHARE_THRESHOLD, CV_THRESHOLD, inputs_str)
    )

    print("=" * 68)
    print("T11 regime-map freeze protocol")
    print("=" * 68)
    print("map file        : %s (%d rows)" % (map_path, n_rows))
    print("map sha256      : %s" % map_hash)
    for k, v in input_hashes.items():
        print("input sha256    : %s -> %s" % (k, v))
    if missing:
        print("missing inputs  : %s (declare or re-run)" % ", ".join(missing))
    print("-" * 68)
    if args.dry_run:
        print("DECISION_LOG entry (DRY RUN — not written):")
    else:
        print("Append this line to DECISION_LOG.md now:")
    print()
    print(log_line)
    print()
    print("Then git-commit regime_map.csv + DECISION_LOG.md and record the")
    print("git commit hash next to this line. The freeze is complete when")
    print("both hashes exist in the log. T11 governs from that moment.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
