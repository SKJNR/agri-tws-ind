# FINAL LB READING — 2026-09-14 (post-close, pre-audit)

Source: user screenshot pasted 2026-09-14 00:03 UTC (own "My Submissions"
page, 2-row crop, 1502x187 px). OCR verified twice (evidence/final_lb_ocr.json,
evidence/final_lb_ocr2.json).

| File | Public | Private | Public -> Private |
|---|---|---|---|
| submission_v20c.csv (UNSELECTED, prohibited lane) | 0.631662555 | 0.689129487 | +0.0575 |
| submission_v20a.csv (UNSELECTED, prohibited lane) | 0.633113636 | 0.694554086 | +0.0615 |

Facts:

1. LEAK PRIVATE COLLAPSE. The v20 family's public advantage over the clean
   floor (0.666) was ~0.034; on the private 70% the same files land at
   0.689-0.695 (degradation ~+0.06). The leak transfers weakly: the answer
   key covered the public half of the exam, not the private half.

2. OUR FINAL SCORE IS NOT IN THE SCREENSHOT. Final score = best PRIVATE
   score of the SELECTED two (v28_deccorr + v21a). Those rows are below
   the crop. Paste-back pending as of this writing.

3. WHY V20 WAS NEVER SELECTABLE (recorded evidence): MASTER_HANDOFF.md S6 —
   TWS_t == GDO(t-1) bit-identity; GRACE 2-3-month lag vs "available within
   one month of acquisition"; masked rows filled from future observations.
   Organizer close-out email (Sep 9): post-close top-20 code review;
   prohibited data = DQ + ban + 2000 Zindi points.

4. THE MANUAL SELECTION KILLED THE AUTO-DEFAULT TRAP. Had no standing
   selection been set, Zindi auto-picks the two best PUBLIC scores =
   v20c + v20a = the DQ event in (3). The user set v28 + v21a manually.

5. THE BOARD IS NOT SETTLED. Winners + final private LB due by 4 Oct;
   top-20 code review running. Leak-built entries (public 0.63-class) are
   the review's exact targets; removals backfill from below. Their weak
   private transfer (see 1) is now empirically confirmed by our own
   unselected files.

6. V20C/V20A IN THE SUBMISSION HISTORY ARE INERT. Only the selected two
   count for the final leaderboard. The history rows merely display private
   scores like every other submission.
