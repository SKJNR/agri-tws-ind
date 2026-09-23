# Sealed extract — unseal procedure (D#21)

This directory holds 2286174 groundwater-level readings dated 2023-01-01 or later
(the model-evaluation SEAL window: label statistics 2023-25 are SEALED per
D#24 terminology; the seal protects label stats, not the calendar).

Rules:
1. Nothing in this directory may be read, hashed into a report, plotted, or
   used as a feature/target/statistic by ANY task before its registered
   position in the critical path (AM-6), and never during D1.1 (<=2022 rows
   only), regime-map freeze (T11), or any baseline-ladder run (D#23).
2. Unsealing requires: (a) a DECISION_LOG entry stating the gate that
   authorizes evaluation on 2023-25, (b) founder notification, (c) a line in
   ACCESS_LOG.md BEFORE the read.
3. Physical re-seal after an authorized read is NOT automatic — each
   authorized evaluation read is logged; the file itself stays sealed-by-
   convention for all other lanes.
4. Any accidental read = protocol surprise = DECISION_LOG entry + founder
   notification (no silent fixes).
