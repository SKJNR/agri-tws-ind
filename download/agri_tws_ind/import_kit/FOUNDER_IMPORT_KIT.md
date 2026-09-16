# FOUNDER IMPORT KIT — AGRI-TWS-IND
**Task 57-KIT | built 2026-09-15 (GLM) | unblocks the 3 critical-path items
from Q57-A: gwl_data.csv handback, LGD district codes, polygons**

Total founder time: ~15 minutes. Each move is independent — do them in any
order, in different sittings if convenient. Nothing here consumes a decision
number; AM-6 implementation order for steps 3–8 is unchanged.

Live capability context (probed 2026-09-15): this sandbox CAN reach
CDS/GitHub/PyPI/Zindi-API directly (ERA5 batch already proven). It CANNOT
reach the Indian govt portals (WRIS DNS-dead, IMD refused, data.gov.in 500,
IndiaAI 403) — that is why these three moves are yours, not mine. Provenance
design (first-landing manifest, open/sealed split, blindfold) is the second
reason and applies regardless of network.

---

## MOVE 1 — LGD district codes (5 min, browser)

**Unblocks:** backfill of `lgd_district_code` in `d20/district_basis.csv`
(59 rows = PENDING_FOUNDER_EXPORT) + polygon binding.

1. Open **https://lgdirectory.gov.in** → Reports → **"District wise Detail
   Report"** page (State/District dropdowns + captcha). Let it load fully.
2. Press **F12 → Console**, paste the whole of `lgd_district_fetcher.js`,
   press Enter.
3. It calls the portal's own dropdown service — read-only, no form
   submission, no captcha typing needed (if it reports a session error,
   solve the captcha on the page once and re-paste).
4. `lgd_district_export_<date>.json` downloads automatically.
5. Send me that file.

**Expected:** AP (state code 28) → 26 districts, TG (36) → 33. A count
mismatch is saved anyway and is itself evidence — tell me either way.

**Plan B (optional, more durable):** NAPIX — https://dev.napix.gov.in/nic/lgd/
— offers LGD APIs with a subscription key. If you subscribe, say so and I
will write the pinned API fetcher; skip it for now, Plan A is faster.

---

## MOVE 2 — gwl_data.csv handback (5–10 min, local machine)

**Unblocks:** D#21 physical split → D1.1 (open-period stats) → everything
downstream (steps 3–8 of the frozen order).

1. Put `gwl_handback_check.py` next to your local `gwl_data.csv` copy.
2. Run: `python3 gwl_handback_check.py gwl_data.csv`
   (Windows: `py gwl_handback_check.py gwl_data.csv` — pure stdlib, no pip).
   Optional first: `python3 gwl_handback_check.py --selftest` (10-row
   synthetic check, prints PASS).
3. It writes `gwl_data.receipt.json` — a few KB, aggregate stats only.
   **Value statistics are computed on open-period (≤2022) rows only**; the
   2023–25 label stats stay sealed per D#24 even inside the receipt.
4. If the file is big for chat upload, re-run with `--gzip` and upload the
   `.gz` (typical CSV compression ~70–80%). If even that is too big, tell
   me the size and I will give you a chunked-upload plan.
5. Upload the file (or .gz) here in chat **and** paste/attach the receipt.

**What happens on arrival (my side, in order):**
- re-run the same census sandbox-side; SHA256 + row counts must match your
  receipt exactly (byte-level transfer proof), else FATAL and we stop;
- run pre-staged `d21_physical_split.py` (mocktest already PASS): open/
  ≤2022, sealed/ ≥2023, manifests, sealed-dir lock, access log;
- **Task-55 condition:** provenance spot-check on sample wells vs
  India-WRIS raw — values / dates / units / round-months. Keep your local
  copy until this passes;
- **AM-3:** the file becomes an ACTIVE pipeline input (open extract feeds
  D1.1), never a view-only archive — that is the whole point of the split.

---

## MOVE 3 — district polygons (5–10 min, browser)

**Unblocks:** `d20_assign_wells.py` (well→district assignment on real data).

Full instructions in **`POLYGON_HANDOFF_NOTES.md`** (Bhuvan Bhoonidhi path,
acceptance checks: AP 26 / TG 33 features, AP vintage ≥ 2022-04-04,
GADM banned, OSM fallback needs a logged amendment first).

---

## DO-NOT (founder edge, standing rules)

- **No credentials in chat** — never paste portal passwords/API keys; the
  scripts are designed so you never need to.
- **No IndiaAI covariate/stacker outputs** into anything touching val/test
  (AM-2) — if unsure what counts, ask before uploading.
- **Don't rename or hand-edit** any file after the receipt hashes it
  (including "fixing" one cell — report it instead; AM-5: no silent fixes).
- **Don't email or otherwise copy sealed extracts** once they exist — the
  sealed/ directory has a written unseal procedure; use it or ask.

## After the moves land

| Move | I run next | Consensus note |
|---|---|---|
| LGD JSON | district_basis.csv backfill + name/code cross-check vs D#20 | D#20 amended in place, no new number |
| gwl + receipt | census match → D#21 split → Task-55 spot-check plan | unseal procedure applies from minute one |
| polygons | d20_assign_wells.py on real wells + assignment report | T-D20-2 execution |

Courier line for Qwen (paste verbatim):
> GLM built founder import kit (Task 57-KIT): LGD browser fetcher,
> gwl handback checker w/ D#24-safe receipt, polygon handoff notes.
> No decision consumed; steps 3–8 ordering unchanged; Q57-B still
> awaited. Live egress probe archived in worklog (CDS/Zindi reachable;
> Indian govt portals refuse sandbox — founder moves remain by design).
