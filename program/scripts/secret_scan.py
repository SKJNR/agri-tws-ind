#!/usr/bin/env python3
"""
secret_scan.py — H-1 hygiene amendment (Qwen, Addendum 9, 2026-09-23).

Scans repo content for credential-shaped strings. FAIL-LOUD: exit 1 on
any hit (commit blocked), 0 clean, 2 internal error.

NOTE (ghost-execution incident, 2026-09-23): this file was originally
excluded from its own commit by the H-1 `*secret*` ignore glob — the
scanner failed to land while the scan RESULTS were reported clean
(they were: the local run was real, the file was simply never pushed).
Fixed with an explicit `!program/scripts/secret_scan.py` exception in
.gitignore. Founder audit caught it. Lesson: verify the FILE LIST of
every commit, not just the exit status.

Modes
  --staged   scan files staged for the next commit (pre-commit hook)
  --all      scan every git-tracked file (sweep-review mode)

Usage
  python3 program/scripts/secret_scan.py --staged
  python3 program/scripts/secret_scan.py --all

Install as pre-commit hook (re-run after every sandbox reset — the
hook lives in .git/ which resets wipe; sweep checklist item S2):
  printf '#!/bin/sh\nexec python3 "$(git rev-parse --show-toplevel)\
/program/scripts/secret_scan.py" --staged\n' > .git/hooks/pre-commit \
  && chmod +x .git/hooks/pre-commit
"""

import argparse
import re
import subprocess
import sys

# Service-specific token formats (high precision, no false-positive risk)
NAMED_PATTERNS = [
    (re.compile(r"github_pat_[A-Za-z0-9_]{20,}"), "GitHub fine-grained PAT"),
    (re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}"), "GitHub classic token"),
    (re.compile(r"sk-[A-Za-z0-9_-]{20,}"), "OpenAI-style API key"),
    (re.compile(r"hf_[A-Za-z0-9]{20,}"), "HuggingFace token"),
    (re.compile(r"AKIA[0-9A-Z]{16}"), "AWS access key id"),
    (re.compile(r"aws_secret_access_key\s*[=:]\s*\S+"), "AWS secret key"),
    (re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"), "Slack token"),
    (re.compile(r"AIza[0-9A-Za-z_-]{30,}"), "Google API key"),
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"), "private key block"),
    (re.compile(r"JSESSIONID[=:][A-Za-z0-9._!-]{16,}"), "live session cookie"),
    (re.compile(r"(?i)authorization:\s*bearer\s+[A-Za-z0-9._-]{20,}"),
     "bearer credential"),
]

# Generic assignment pattern, only for credential-LIKE FILENAMES
GENERIC_FILE_RE = re.compile(
    r"(?i)(token|secret|credential|password|passwd|api[_-]?key)")
GENERIC_LINE_RE = re.compile(
    r"(?i)\b(password|passwd|secret|token|api[_-]?key)\b\s*[=:]\s*"
    r"['\"]?([A-Za-z0-9+/_-]{16,})['\"]?")
PLACEHOLDERS = re.compile(
    r"(?i)^(x+|changeme|change_me|placeholder|your[_-]|example|dummy|"
    r"test|none|null|redacted|<[^>]*>|\$\{[^}]*\}).*")

MAX_FILE_BYTES = 20 * 1024 * 1024  # skip gigantic blobs (none tracked)


def sh(*args):
    return subprocess.run(args, capture_output=True, text=True,
                          check=False).stdout


def staged_files():
    out = sh("git", "diff", "--cached", "--name-only",
             "--diff-filter=ACMRT")
    return [f for f in out.splitlines() if f.strip()]


def all_files():
    out = sh("git", "ls-files")
    return [f for f in out.splitlines() if f.strip()]


def redact(s):
    return s[:6] + "...REDACTED..." if len(s) > 6 else "<REDACTED>"


def scan_file(path, is_generic_target):
    """Return list of (line_no, why, redacted_snippet)."""
    hits = []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            body = f.read(MAX_FILE_BYTES)
    except OSError:
        return hits  # unreadable (deleted in index etc.) — not our problem
    lines = body.splitlines()
    for i, line in enumerate(lines, 1):
        for pat, why in NAMED_PATTERNS:
            m = pat.search(line)
            if m:
                hits.append((i, why, redact(m.group(0))))
        if is_generic_target:
            m = GENERIC_LINE_RE.search(line)
            if m and not PLACEHOLDERS.match(m.group(2)):
                hits.append((i, "credential-like assignment",
                             redact(m.group(2))))
    return hits


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--staged", action="store_true")
    g.add_argument("--all", action="store_true")
    a = ap.parse_args()

    files = staged_files() if a.staged else all_files()
    if not files:
        print("secret_scan: nothing to scan")
        return 0

    bad = 0
    for path in files:
        generic = bool(GENERIC_FILE_RE.search(path))
        for line_no, why, snippet in scan_file(path, generic):
            bad += 1
            print(f"SECRET-SCAN HIT {path}:{line_no} [{why}] {snippet}",
                  file=sys.stderr)

    if bad:
        print(f"secret_scan: FAIL — {bad} hit(s); commit blocked. "
              "Remove the credential, rotate if it was ever live, "
              "then re-stage.", file=sys.stderr)
        return 1
    print(f"secret_scan: CLEAN ({len(files)} file(s))")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:  # noqa: BLE001
        print(f"secret_scan: ERROR {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(2)
