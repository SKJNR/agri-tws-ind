# FOUNDER UNBLOCK CARD — 2026-09-24 (what only YOU can do)

**Everything else is verified healthy** — repo at `dac49a0`, secret scan CLEAN
(1,980 files), backbone + riders + PR-8 all landed and API-verified, work
continues autonomously. This card is the short list of items that need YOUR
hands. Quickest first. Nothing here blocks me from working; all of it
accelerates or secures the path.

---

## 1. Flip the repo to PRIVATE (2 min) — oldest standing item

Your own rule ("never public without asking") — the repo has been public
since creation (standing record since Addendum 6, re-flagged on every sweep,
most recently TODAY).

- `github.com/SKJNR/agri-tws-ind` → **Settings** → scroll to **Danger Zone**
  → **Change visibility** → **Private**.
- Nothing breaks: my token is repo-scoped and keeps working.

## 2. Flip BOTH HuggingFace datasets to PRIVATE (2 min)

AIKosh participant licensing — your own advisory from Sep 16, not yet executed.

- `huggingface.co/datasets/han-jisso/ground-water-level-all` → Settings → Private
- `huggingface.co/datasets/han-jisso/ndvi-forecasting-model` → Settings → Private

Safe to do right now: I already fetched + SHA256-verified the zip I need for
the current analysis (bit-perfect re-supply, `1e9d0cf6…`).

## 3. Pilot-12 selection (D1.1 gate) — a decision, not clicks

Per PR-10: 12 districts, regime-mixed, may be a subset of the 59. When you
name them (a chat message is enough), the D1.1 empirical step unblocks
(together with the Qwen loaders L1–L12). Suggested format:
`pilot12 = [district names]`.

## 4. LGD / Bhuvan-grade district polygons (regime-map freeze + D#20 promotion)

Still absent from sandbox. Any open-licensed district polygon source
(Bhuvan download, LGD-boundary shapefile) uploaded to the HF dataset (or
pasted as a download link) unblocks: regime-map freeze, polygon binding for
the 2,471 PARENT_CONTINUE wells, and the D#20 promotion trigger.

## 5. OPTIONAL — 3 manual India-WRIS lookups (Track B confirmation, ~5 min)

India-WRIS is unreachable from my sandbox (timeouts — government network
blocks the cloud IP range). This is the ONLY route that can confirm the
AIKosh file's values match WRIS live records. My Track A analysis (running
now, pre-registered in Addendum 11) resolves the unit/datum question from
the file itself; these lookups would additionally confirm raw-value
provenance:

1. Open `indiawris.gov.in` → Water Resources → Ground Water Level.
2. Search station code: **pick any 3 from the sample list I will attach to
   the Track A report** (one from each verdict class, if classes emerge).
3. For each: note the displayed water-level value + units (mbgl/masl) for
   the same date shown in my sample table.
4. Paste the 3 results back in chat — I reconcile against the file values
   and file the confirmation in DECISION_LOG.

## 6. Sealed-extract custody note (informational, no action)

The 18th sandbox reset wiped the local sealed extract, as designed. This
session I re-supplied the founder original via the public HF zip
(SHA256-verified bit-perfect) and will re-verify the sealed split hash
(`71f492b2…`) before the D#21 split re-run. The sealed rows stay untouched
for all current work — no 2023+ data is read by anything on the registered
path before D1.1 → regime freeze.

---

**One-line status**: backbone done; next technical step (Task-55 datum
analysis) pre-registered and running; everything after that waits on items
3 and 4 above (and Qwen's L1–L12 loaders).
