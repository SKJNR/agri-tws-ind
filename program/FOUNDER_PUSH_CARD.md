# FOUNDER PUSH CARD — make the repo permanent (2026-09-15)
**One-time setup, ~60 seconds (Path A) or ~5 min (Path B). After this,
session resets can never eat our work again.**

## Path A — no git needed (60 seconds)
1. Go to **github.com** → sign in → top-right **+** → **New repository**
2. Name: `agri-tws-ind` · Visibility: **Private** (required — competition
   data licensing) → **Create** (do NOT add README/license)
3. On the empty-repo page click **"uploading an existing file"**
4. Open `program_2026-09-15.zip` (from this chat's downloads), drag its
   `program` folder contents into the page → **Commit changes**
✅ Done. All program state is now on GitHub.

## Path B — proper git (5 minutes, keeps full history since Task 18)
1. Create the same **private** repo on github.com (no README)
2. Download `agri_tws_ind_repo_full_2026-09-15.bundle`
3. In a terminal, inside the folder where you saved it:
   ```
   git clone agri_tws_ind_repo_full_2026-09-15.bundle agri-tws-ind
   cd agri-tws-ind
   git remote set-url origin https://github.com/<your-username>/agri-tws-ind.git
   git push -u origin main --tags
   ```
   (If you already pushed the Sep-10 bundle earlier, instead fetch the
   small `agri_tws_ind_repo_incr_2026-09-15.bundle`:
   `git fetch agri_tws_ind_repo_incr_2026-09-15.bundle program-2026-09-15` then push.)
✅ Done.

## After today
- Whenever I hand you a `..._incr_....bundle`: `git fetch <bundle> <tag>` + push.
- When gwl_data.csv arrives in chat, I commit its receipt + manifest
  immediately (ARRIVAL=COMMIT); the file itself goes up as a Release
  asset if it's over 95 MB — I'll hand you exact click-steps then.
- Everything stays PRIVATE. One rule for you: never make this repo public
  without asking me first (competition data licensing).
