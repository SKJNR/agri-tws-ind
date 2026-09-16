# GITHUB PUSH INSTRUCTIONS — repo packaged 2026-09-10 (Task 41)

**Question answered:** "did all our solutions including analysis, successes,
failures and all methods we tried — everything — get into the GitHub repo?"
**Answer: they are ALL committed locally now** (796 files, 2 branches, 1 tag,
clean status). What was missing is only the PUSH — no remote/credentials
exist on this box. Two ways to finish it below.

---

## 1. What IS in the repo (everything we ever made)

| Category | Contents |
|---|---|
| **All solutions** | 80 submission CSVs: v1 → v28 (incl. lost-era recoveries, splice/rescue variants, v20 prohibited-lane files kept for the audit trail) + every probe pair (201509/201601/201606/201609/201612/201704/201807/201811 + 201812 zero-month pre-build) |
| **All methods** | 246+ task scripts — every lane tried: kept (Kalman lineage, correction chain) AND tombstoned (stacking, isotonic, E1, ERA5, GPCP, per-cell Kalman, blends, GRACE v20-lane …) |
| **All analysis** | submissions_manifest (pre-registered gates, 6 display-level exact predictions), FINAL2_SELECTION Round-13, HAT_PROTOCOL, ENDGAME_TOP20_MEMO, ADVERSARIAL_REVIEWS R1–R4, audits A/B/C, METHODOLOGY_AUDIT, gate reviews, build logs, CV↔LB correlation study |
| **The narrative** | worklog.md — Tasks 1–41, every decision with numbers, successes AND failures |
| **Evidence** | LB snapshots (html/json), competition metadata, upload screenshots |

Branches: `main` (full history, 8+4 commits) · `github-release` (single
snapshot commit, same tree — a clean release view) · tag `v28-endgame`.

## 2. What is deliberately NOT in the repo (and why)

- `data/` — organizer-provided competition files (rules + size; the repo
  regenerates everything from Zindi's own package).
- `*.nc` — external GRACE/GDO/ERA5 binaries (**prohibited data never enters
  the repo — the repo itself is audit-clean**; the v20-lane SCRIPTS are
  included because they document the tried-and-refused lane).
- caches (`.npz`), repo bundles, skills/tooling internals.

## 3. How to push — Path A (from this box; GitHub reachable, verified)

1. On github.com create a **PRIVATE** repo, e.g. `tws-drought-endgame`.
   Do NOT add README/.gitignore/license (content already exists).
2. Either paste here: the repo URL + a fine-grained PAT (Contents: Read/Write
   on that repo only) — and the push runs from this box — or run yourself:
   ```
   cd /home/z/my-project
   git remote add origin https://github.com/<YOU>/tws-drought-endgame.git
   git push -u origin main
   git push origin github-release v28-endgame
   ```
   First push ≈ 240 MB (one-time; after that increments are tiny).
3. Delete the PAT afterwards if shared. Commits are authored
   `Jisoo <jisoo@local>` — to link them to your GitHub account:
   `git config user.email "<real>" && git commit --amend --reset-author --no-edit`
   (HEAD) or `git rebase -i --exec 'git commit --amend --reset-author --no-edit' main~5`.

## 4. How to push — Path B (offline, from any machine with the bundle)

1. Download `download/git_repo_2026-09-10.bundle` (240 MB, md5-verified,
   contains all 4 refs).
2. On your machine:
   ```
   git clone git_repo_2026-09-10.bundle tws-drought-endgame
   cd tws-drought-endgame
   git remote set-url origin https://github.com/<YOU>/tws-drought-endgame.git
   git push -u origin main github-release v28-endgame
   ```

## 5. Durable copies (belt + suspenders)

| Artifact | Location | Survives workspace resets? |
|---|---|---|
| Full git repo (`.git`, 331M) | `/home/z/my-project/.git` | survived 9 so far, not guaranteed |
| Bundle (240M, all refs) | `download/git_repo_2026-09-10.bundle` | mirrored /tmp + /home/sync |
| Full workspace tar (774M) | `/home/sync/repo.tar` | YES — /home/sync outlived all 9 resets |
| Bundle copy | `/home/sync/git_repo_2026-09-10.bundle` | YES |

**Keep the repo PRIVATE until winners are announced (4 Oct).** The top-20
code review runs through Zindi, not GitHub — a public repo before results
only gives our methods away.
