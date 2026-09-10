#!/bin/bash
# Task 41b (2026-09-10): commit EVERYTHING to git; rebuild github-release;
# bundle + durable /home/sync refresh. Run AFTER task41a_restore.sh verified.
set -e
cd /home/z/my-project

echo "== [1/7] git config: kill mode noise, local identity =="
git config core.fileMode false
git config user.name "Jisoo"
git config user.email "jisoo@local"
git config --get user.name; git config --get user.email

echo "== [2/7] .gitignore hardening + untrack organizer data =="
grep -q '^data/$' .gitignore 2>/dev/null || printf 'data/\n*.nc\n*.npz\n' >> .gitignore
cat .gitignore
git rm -r -q --cached data/ 2>/dev/null && echo "  data/ untracked (content kept on disk)" || echo "  data/ was not tracked"

echo "== [3/7] commit 1/3 — code: every method, kept or tombstoned =="
git add -A scripts/
git diff --cached --quiet || git commit -q -m \
"code: correction-lane program, Tasks 18-40 — probe tomography (pair/single-plus/zero-month),
chained builds v25->v28 (round_trip float discipline, bit-audits), gate scripts, split-brain
recovery, Round-4 researcher-hat escalation. Every lane we tried, kept OR tombstoned."
git log --oneline -1

echo "== [4/7] commit 2/3 — artifacts: every solution shipped + full ledger =="
git add -A download/
git diff --cached --quiet || git commit -q -m \
"artifacts: full endgame ledger — submission chain v22->v28 + ALL probe pairs
(201601/201509/201606/201609/201612/201704/201807/201811/201812 zero-month pre-build),
splice/rescue variants, submissions_manifest (6 display-level exact LB predictions,
chain v24->v28 banked 0.008932), FINAL2_SELECTION Round-13 (picks v28+v21a),
HAT_PROTOCOL, ENDGAME_TOP20_MEMO, adversarial reviews R1-R4, audits A/B/C,
methodology audit, report draft. Successes AND failures, all of them."
git log --oneline -1

echo "== [5/7] commit 3/3 — worklog narrative + meta + evidence =="
git add -A
git diff --cached --quiet || git commit -q -m \
"docs: worklog Tasks 18-40 (full decision narrative incl. failures), LB snapshots,
competition metadata, upload evidence screenshots; .gitignore: data/ + *.nc + *.npz
excluded (organizer data + external climate binaries never enter the repo)."
git log --oneline -1
git tag -a v28-endgame -m "v28 public 0.674859467 — 6th display-level exact prediction;
clean team best; endgame snapshot 2026-09-10 (standing selection: v28 + v21a)"
echo "  tagged: $(git describe)"

echo "== [6/7] rebuild github-release orphan (push-ready single snapshot) =="
git checkout -q --orphan release-tmp
git commit -q -m "release: TWS drought endgame 2026-09-10 — complete clean-lane package
(code + ledgers + all submissions v1->v28 + probes; NO data/, NO external GRACE/GDO
binaries; audit-clean). Full history on main."
git branch -q -D github-release
git branch -q -m github-release
git checkout -q main
git log --oneline | head -9

echo "== [7/7] durable artifacts: bundle + /home/sync refresh =="
git bundle create download/git_repo_2026-09-10.bundle --all HEAD 2>/dev/null || git bundle create download/git_repo_2026-09-10.bundle --all
git bundle verify download/git_repo_2026-09-10.bundle | head -3
tar czf /home/sync/repo.tar -C /home \
  --exclude='my-project/data' --exclude='my-project/skills' \
  --exclude='my-project/tool-results' --exclude='my-project/db' \
  --exclude='my-project/node_modules' my-project
cp download/git_repo_2026-09-10.bundle /home/sync/git_repo_2026-09-10.bundle

echo "== REPORT =="
echo "  branches: $(git branch | tr '\n' ' ')"
echo "  tags:     $(git tag | tr '\n' ' ')"
echo "  tracked files: $(git ls-files | wc -l)"
echo "  uncommitted:   $(git status --porcelain | wc -l)"
du -sh .git download/git_repo_2026-09-10.bundle /home/sync/repo.tar 2>/dev/null
echo "  DONE — repo is complete and push-ready"
