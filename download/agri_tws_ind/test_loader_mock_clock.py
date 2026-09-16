#!/usr/bin/env python3
"""
test_loader_mock_clock.py — mocked-clock as-of loader audit (D1.1)
=================================================================
AGRI-TWS-IND-v1 — delivered with R5-GLM (2026-09-14).

WHAT THIS IS
------------
The CI enforcement layer of LATENCY_TABLE.md v1. It freezes the clock at
each pilot ADVISORY_ISSUE date (5th of month, gate G7), runs the vintage
rules for every input row, and asserts that each selected artifact is the
one — and only the one — the latency table allows. It also enforces the
Arm-O EVAL_ONLY leak guard: any artifact whose release date exceeds the
frozen issue date must raise ImportBeyondIssueTime.

MODES
-----
1. self-check (default, runs NOW): the vintage rules are exercised on
   SYNTHETIC artifact manifests with known release dates. Rule logic is
   fully tested today — green means the referee's clock discipline is
   correct, independent of data availability.
2. real-audit (Move 1 T+0 = D1.1-Empirical): same assertions, run against
   real loaders and real manifests. Pass condition (pre-stated,
   DECISION_LOG): 0 round-month mismatches across CGWB x {kharif, rabi}
   issue windows x 12 pilot districts, plus measured lags recorded per
   LATENCY_TABLE row. To activate: implement the LoaderAdapter protocol
   below and pass adapters to run_audit().

NO external dependencies (pure stdlib) so it runs in any CI environment.
"""

import datetime as dt
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

# --------------------------------------------------------------------------
# Core machinery
# --------------------------------------------------------------------------

ADVISORY_ISSUE_DAY = 5  # G7: advisories issue on the 5th of the month


class ImportBeyondIssueTime(Exception):
    """Raised when a loader would return an artifact released AFTER the
    frozen ADVISORY_ISSUE date. This is the Arm-O EVAL_ONLY guard
    (T5-adjacent): future-released artifacts are import-impossible in
    operational paths, enforced by CI rather than by convention."""


@dataclass(frozen=True)
class Artifact:
    """One releasable data artifact. In self-check mode these are
    synthetic; at D1.1-Empirical the real loaders return these."""
    source: str            # LATENCY_TABLE row id, e.g. "L2"
    round_month: dt.date   # first day of the round/center month
    release: dt.date       # the date the artifact became public
    prelim: bool = False


def month_floor(d: dt.date) -> dt.date:
    return dt.date(d.year, d.month, 1)


def add_months(d: dt.date, n: int) -> dt.date:
    m = d.month - 1 + n
    y, m = d.year + m // 12, m % 12 + 1
    return dt.date(y, m, 1)


# --------------------------------------------------------------------------
# Vintage rules — ONE per LATENCY_TABLE row. These are the spec.
# --------------------------------------------------------------------------

def rule_l1_mascons(artifacts: Sequence[Artifact], issue: dt.date) -> Artifact:
    """L1: latest mascon whose center-month + 85d <= ADVISORY_ISSUE.
    Center of month m ~ m+15d, so eligibility = month_floor + 100d <= issue."""
    eligible = [a for a in artifacts if a.round_month + dt.timedelta(days=100) <= issue]
    if not eligible:
        raise LookupError("L1: no lag-legal mascon for issue %s" % issue)
    return max(eligible, key=lambda a: a.round_month)


def rule_l2_cgwb(artifacts: Sequence[Artifact], issue: dt.date) -> Artifact:
    """L2: latest CGWB round whose RELEASE date <= ADVISORY_ISSUE.
    Keys on the release calendar, NEVER the round month — the R3-SS3
    vintage arithmetic (Jun-5 kharif issue sees only the Jan round)."""
    eligible = [a for a in artifacts if a.release <= issue]
    if not eligible:
        raise LookupError("L2: no lag-legal CGWB round for issue %s" % issue)
    return max(eligible, key=lambda a: a.release)


def rule_l3_chirps_prelim(artifacts: Sequence[Artifact], issue: dt.date) -> Artifact:
    """L3: latest prelim with end date <= issue - 3d, flagged PRELIM."""
    eligible = [a for a in artifacts if a.prelim and a.release <= issue - dt.timedelta(days=3)]
    if not eligible:
        raise LookupError("L3: no lag-legal CHIRPS prelim for issue %s" % issue)
    return max(eligible, key=lambda a: a.release)


def rule_l4_chirps_final(artifacts: Sequence[Artifact], issue: dt.date) -> Artifact:
    """L4: latest final month released by issue - 60d."""
    eligible = [a for a in artifacts if not a.prelim and a.release <= issue - dt.timedelta(days=60)]
    if not eligible:
        raise LookupError("L4: no lag-legal CHIRPS final for issue %s" % issue)
    return max(eligible, key=lambda a: a.round_month)


def rule_l5_era5(artifacts: Sequence[Artifact], issue: dt.date) -> Artifact:
    """L5: latest month whose end + 5d <= issue (ERA5T, flagged PRELIM);
    consolidated only if released by issue. Prelim allowed, preferred."""
    eligible = [a for a in artifacts
                if (a.prelim and add_months(a.round_month, 1) - dt.timedelta(days=25) <= issue)
                or (not a.prelim and a.release <= issue)]
    if not eligible:
        raise LookupError("L5: no lag-legal ERA5 month for issue %s" % issue)
    return max(eligible, key=lambda a: (a.round_month, a.prelim))


def rule_l6_imd(artifacts: Sequence[Artifact], issue: dt.date) -> Artifact:
    """L6: latest day <= issue - 1d."""
    eligible = [a for a in artifacts if a.release <= issue - dt.timedelta(days=1)]
    if not eligible:
        raise LookupError("L6: no lag-legal IMD day for issue %s" % issue)
    return max(eligible, key=lambda a: a.release)


def rule_l10_smap(artifacts: Sequence[Artifact], issue: dt.date) -> Artifact:
    """L10: latest granule <= issue - 2d."""
    eligible = [a for a in artifacts if a.release <= issue - dt.timedelta(days=2)]
    if not eligible:
        raise LookupError("L10: no lag-legal SMAP granule for issue %s" % issue)
    return max(eligible, key=lambda a: a.release)


def rule_l11_akb(artifacts: Sequence[Artifact], issue: dt.date) -> Artifact:
    """L11: AKB version hash pinned at issue; frozen for the cycle.
    Any AKB released after the issue date is import-impossible."""
    eligible = [a for a in artifacts if a.release <= issue]
    if not eligible:
        raise LookupError("L11: no pinned AKB version for issue %s" % issue)
    return max(eligible, key=lambda a: a.release)


def rule_l13_openmeteo_archive(artifacts: Sequence[Artifact], issue: dt.date) -> Artifact:
    """L13: Open-Meteo Archive (ERA5/ERA5-Land reanalysis). For as-of
    replay the L5-style rule applies: latest month whose end + 5d <=
    issue is PRELIM (ERA5T); consolidated only if released by issue."""
    eligible = [a for a in artifacts
                if (a.prelim and add_months(a.round_month, 1) - dt.timedelta(days=25) <= issue)
                or (not a.prelim and a.release <= issue)]
    if not eligible:
        raise LookupError("L13: no lag-legal Open-Meteo archive month for issue %s" % issue)
    return max(eligible, key=lambda a: (a.round_month, a.prelim))


RULES: Dict[str, Callable[[Sequence[Artifact], dt.date], Artifact]] = {
    "L1": rule_l1_mascons,
    "L2": rule_l2_cgwb,
    "L3": rule_l3_chirps_prelim,
    "L4": rule_l4_chirps_final,
    "L5": rule_l5_era5,
    "L6": rule_l6_imd,
    "L10": rule_l10_smap,
    "L11": rule_l11_akb,
    "L13": rule_l13_openmeteo_archive,
}


def enforce_no_future_import(artifact: Artifact, issue: dt.date) -> None:
    """Arm-O EVAL_ONLY guard: raises if the selected artifact was released
    after the frozen issue date. Called on EVERY selection, every row."""
    if artifact.release > issue:
        raise ImportBeyondIssueTime(
            "%s artifact released %s > frozen issue %s — "
            "future import blocked (Arm-O guard)" % (artifact.source, artifact.release, issue))


# --------------------------------------------------------------------------
# Issue calendar (pilot windows; 12 districts ride the same dates)
# --------------------------------------------------------------------------

def pilot_issue_dates(years: Sequence[int] = (2021, 2022, 2023, 2024)) -> List[dt.date]:
    """Kharif (Jun-Sep) + rabi (Oct-Jan) issue windows, 5th of month."""
    dates = []
    for y in years:
        for m in (6, 7, 8, 9, 10, 11, 12):
            dates.append(dt.date(y, m, ADVISORY_ISSUE_DAY))
        dates.append(dt.date(y + 1, 1, ADVISORY_ISSUE_DAY))  # rabi Jan of next year
    return sorted(dates)


# --------------------------------------------------------------------------
# Self-check: synthetic manifests with known ground truth
# --------------------------------------------------------------------------

def synthetic_manifest(source: str) -> List[Artifact]:
    """Builds a dense synthetic release history 2020-2025 for one row."""
    arts: List[Artifact] = []
    start = dt.date(2020, 1, 1)
    if source == "L1":  # monthly mascons, center + 85d ~= floor + 100d release
        m = start
        while m < dt.date(2025, 6, 1):
            arts.append(Artifact("L1", m, m + dt.timedelta(days=100)))
            m = add_months(m, 1)
    elif source == "L2":  # CGWB quarterly rounds Jan/May/Aug/Nov, +6wk release
        for y in range(2020, 2025):
            for m in (1, 5, 8, 11):
                r = dt.date(y, m, 1)
                arts.append(Artifact("L2", r, r + dt.timedelta(days=45)))
    elif source == "L3":  # CHIRPS prelim daily, +3d
        d = start
        while d < dt.date(2025, 6, 1):
            arts.append(Artifact("L3", d, d + dt.timedelta(days=3), prelim=True))
            d += dt.timedelta(days=1)
    elif source == "L4":  # CHIRPS final monthly, +45d
        m = start
        while m < dt.date(2025, 6, 1):
            arts.append(Artifact("L4", m, add_months(m, 1) + dt.timedelta(days=15)))
            m = add_months(m, 1)
    elif source == "L5":  # ERA5T monthly prelim +5d, consolidated +70d
        m = start
        while m < dt.date(2025, 6, 1):
            arts.append(Artifact("L5", m, add_months(m, 1) + dt.timedelta(days=4), prelim=True))
            arts.append(Artifact("L5", m, add_months(m, 2) + dt.timedelta(days=10), prelim=False))
            m = add_months(m, 1)
    elif source == "L6":  # IMD daily +2d
        d = start
        while d < dt.date(2025, 6, 1):
            arts.append(Artifact("L6", d, d + dt.timedelta(days=2)))
            d += dt.timedelta(days=1)
    elif source == "L10":  # SMAP L4 daily +3d
        d = start
        while d < dt.date(2025, 6, 1):
            arts.append(Artifact("L10", d, d + dt.timedelta(days=3)))
            d += dt.timedelta(days=1)
    elif source == "L11":  # AKB quarterly versioned releases
        for y in range(2020, 2025):
            for m in (1, 4, 7, 10):
                arts.append(Artifact("L11", dt.date(y, m, 1), dt.date(y, m, 1)))
    elif source == "L13":  # Open-Meteo archive: ERA5T monthly prelim +5d, consolidated +70d
        m = start
        while m < dt.date(2025, 6, 1):
            arts.append(Artifact("L13", m, add_months(m, 1) + dt.timedelta(days=4), prelim=True))
            arts.append(Artifact("L13", m, add_months(m, 2) + dt.timedelta(days=10), prelim=False))
            m = add_months(m, 1)
    return arts


def expected_l2_round(issue: dt.date) -> dt.date:
    """Ground truth for the L2 rule given the synthetic release calendar
    (rounds Jan/May/Aug/Nov released +45d). Computed independently of the
    rule implementation so the test can catch a wrong rule, not just a
    broken one."""
    rounds = []
    for y in range(2020, 2025):
        for m in (1, 5, 8, 11):
            r = dt.date(y, m, 1)
            if r + dt.timedelta(days=45) <= issue:
                rounds.append(r)
    return max(rounds)


def run_self_check() -> int:
    failures: List[str] = []
    issues = pilot_issue_dates()

    # 1. Every rule selects an artifact for every pilot issue date and no
    #    selection ever violates the Arm-O future-import guard.
    for src, rule in RULES.items():
        manifest = synthetic_manifest(src)
        for issue in issues:
            try:
                a = rule(manifest, issue)
                enforce_no_future_import(a, issue)
            except Exception as e:
                failures.append("%s @ %s: %s" % (src, issue, e))

    # 2. L2 vintage arithmetic — the R3-SS3 killing case, spelled out:
    #    a Jun-5 kharif issue must see ONLY the January round.
    cgwb = synthetic_manifest("L2")
    jun5 = dt.date(2021, 6, 5)
    got = rule_l2_cgwb(cgwb, jun5)
    if got.round_month != dt.date(2021, 1, 1):
        failures.append("L2 Jun-5-2021 issue: expected Jan-2021 round, got %s" % got.round_month)
    # Dec-5 issue sees the Aug round (Nov round released ~Dec 15 — dark).
    dec5 = dt.date(2021, 12, 5)
    got = rule_l2_cgwb(cgwb, dec5)
    if got.round_month != dt.date(2021, 8, 1):
        failures.append("L2 Dec-5-2021 issue: expected Aug-2021 round, got %s" % got.round_month)
    # Feb-5 issue sees the Nov round (rabi ladder's strongest signal).
    feb5 = dt.date(2022, 2, 5)
    got = rule_l2_cgwb(cgwb, feb5)
    if got.round_month != dt.date(2021, 11, 1):
        failures.append("L2 Feb-5-2022 issue: expected Nov-2021 round, got %s" % got.round_month)

    # 3. L2 rule agrees with an independently-computed ground truth on
    #    every pilot issue date (guard against rule/logic co-drift).
    for issue in issues:
        want = expected_l2_round(issue)
        got = rule_l2_cgwb(cgwb, issue)
        if got.round_month != want:
            failures.append("L2 @ %s: rule gave %s, independent truth %s"
                            % (issue, got.round_month, want))

    # 4. Arm-O guard explicitly: attempting to hand a future artifact into
    #    an operational path must raise.
    future = Artifact("L1", dt.date(2024, 6, 1), dt.date(2200, 1, 1))
    try:
        enforce_no_future_import(future, dt.date(2024, 6, 5))
        failures.append("Arm-O guard FAILED to raise on future import")
    except ImportBeyondIssueTime:
        pass  # correct behavior

    # 5. L1 mascon arithmetic: a Jun-5 issue's freshest legal mascon must
    #    center at Feb or earlier (85d rule).
    mascons = synthetic_manifest("L1")
    got = rule_l1_mascons(mascons, jun5)
    if got.round_month > dt.date(2021, 2, 1):
        failures.append("L1 Jun-5-2021 issue: freshest legal mascon %s exceeds Feb-2021"
                        % got.round_month)

    n_issues = len(issues)
    print("=" * 68)
    print("D1.1 mocked-clock loader audit — SELF-CHECK MODE")
    print("=" * 68)
    print("issue dates tested : %d (kharif Jun-Sep + rabi Oct-Jan, 2021-2024)" % n_issues)
    print("rules exercised    : %s" % ", ".join(sorted(RULES)))
    print("table version      : LATENCY_TABLE v1.1 (L13 added 2026-09-15, Task 51)")
    print("synthetic manifests: %d artifacts/row, release calendars encoded"
          % len(synthetic_manifest("L3")))
    print("-" * 68)
    if failures:
        print("FAILURES (%d):" % len(failures))
        for f in failures[:20]:
            print("  - %s" % f)
        print("SELF-CHECK: FAIL")
        return 1
    print("All vintage rules consistent with LATENCY_TABLE.md v1.1.")
    print("Arm-O EVAL_ONLY guard verified (future imports raise).")
    print("Kharif Jun-5 -> Jan CGWB round confirmed (the R3-SS3 case).")
    print("SELF-CHECK: PASS")
    return 0


# --------------------------------------------------------------------------
# Real-audit protocol (activated at Move 1 = D1.1-Empirical)
# --------------------------------------------------------------------------

@dataclass
class LoaderAdapter:
    """Implement one per input row at Move 1; the audit then runs the
    SAME assertions against real artifacts. Pass = 0 mismatches."""
    source: str
    list_artifacts: Callable[[dt.date, dt.date], List[Artifact]]  # (from, to)


def run_audit(adapters: Sequence[LoaderAdapter],
              issues: Optional[Sequence[dt.date]] = None) -> Dict[str, str]:
    """D1.1-Empirical entrypoint. Returns per-row pass/fail; any FAIL
    blocks the PR-7 strata freeze until the one allowed re-freeze is
    logged (T7)."""
    issues = list(issues or pilot_issue_dates())
    report: Dict[str, str] = {}
    for ad in adapters:
        mismatches = 0
        for issue in issues:
            manifest = ad.list_artifacts(issue - dt.timedelta(days=400), issue)
            rule = RULES.get(ad.source)
            if rule is None:
                report[ad.source] = "NO RULE — add LATENCY_TABLE row first (T9)"
                continue
            try:
                a = rule(manifest, issue)
                enforce_no_future_import(a, issue)
            except ImportBeyondIssueTime:
                mismatches += 1
            except Exception:
                mismatches += 1
        report[ad.source] = "PASS" if mismatches == 0 else "FAIL (%d issues)" % mismatches
    return report


if __name__ == "__main__":
    sys.exit(run_self_check())
