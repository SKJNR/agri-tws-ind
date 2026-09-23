# FOUNDER PUSH CARD v2 — make the repo permanent + AI debate arena (2026-09-16)
**✅ STATUS 2026-09-17: DONE — Path T executed. Repo live at github.com/SKJNR/agri-tws-ind
(private; main + tags + github-release + Issue #1). GLM pushes autonomously each round.
Token: single-repo fine-grained PAT, 90-day expiry — delete it any time in Developer
settings if you ever want to revoke sandbox access. This card is now historical.**

**Goal: everything on GitHub (private), then GLM/Qwen/Claude debate through the repo itself.**
Everything is staged and waiting. One action from you activates it.

## Path T — RECOMMENDED (token, ~3 min; I do everything from then on)
1. On github.com create a **private** repo named `agri-tws-ind`
   (top-right **+** → New repository → Private → Create; NO README/license).
2. Profile photo (top-right) → **Settings** → left-bottom **Developer settings**
   → **Personal access tokens** → **Fine-grained tokens** → **Generate new token**.
3. Fill: name `agri-tws-ind-sandbox` · expiration 90 days ·
   Repository access → **Only select repositories** → `agri-tws-ind`.
4. Permissions → Repository permissions: **Contents = Read and write**,
   **Issues = Read and write**. Give it nothing else.
5. Generate → copy the token → **paste it here in chat**.
6. I immediately push the full repo (all history, logs, evidence,
   manifests) and confirm here. You verify by refreshing the repo page.

Why this is safe: the token touches ONLY this one repo — it cannot
see or change anything else in your account, cannot create repos,
expires by itself in 90 days, and you can delete it anytime on the
same page. If you'd rather not paste a token at all → Path B.

## Path B — proper git yourself (~5 min, keeps full history)
1. Create the same private repo on github.com (no README).
2. Download `agri_tws_ind_repo_full_2026-09-16.bundle` from this chat.
3. In a terminal, in the folder where you saved it:
   ```
   git clone agri_tws_ind_repo_full_2026-09-16.bundle agri-tws-ind
   cd agri-tws-ind
   git remote set-url origin https://github.com/<your-username>/agri-tws-ind.git
   git push -u origin main --tags
   ```
✅ Done.

## Path A — fallback (60 s, partial state only)
Upload `program_2026-09-16.zip` contents via the repo page's
"uploading an existing file" link. Gets program/ + the decision log
onto GitHub but NOT the git history. Only use if T and B are both
impossible today.

## The AI debate arena (activates on push, no extra setup)
- Every courier round becomes a **GitHub Issue** on the repo. The
  first one is already drafted: `program/GITHUB_ISSUE_DRAFT.md`
  (the 4 pending asks: 59-vs-61 district ruling, provisional
  assignment countersign, data findings, PRP v1.1 review).
- You keep working exactly as now: paste the issue text to
  Qwen/Claude, paste their reply back here. I transcribe the reply
  into the issue as a comment, record the ruling in DECISION_LOG.md,
  and commit — every debate round gets a permanent URL and a commit
  hash. Nothing lives only in chat scrollback anymore.
- If Qwen/Claude accounts ever get direct GitHub access, they can
  comment in the issues themselves — the loop is already shaped
  for that. DECISION_LOG.md stays the single source of truth;
  issues are the debate venue, not a second ledger.
- Rules carried over: repo stays **private** (licensing); never
  make it public without asking first; the big gwl zip goes up as
  a **Release asset** (I'll hand exact click-steps, or do it myself
  under Path T).

## After today
- Path T: nothing — I push after every work round automatically.
- Path B: whenever I hand you a `..._incr_....bundle`:
  `git fetch <bundle> main` then `git push`.
