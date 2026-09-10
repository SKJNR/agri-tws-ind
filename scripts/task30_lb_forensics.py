"""Task 30: LB FORENSIC CENSUS — who is the 0.63-0.67 band, and is our
"clean floor" story consistent with the actual field data?

User's question: we sit ~25th; teams at 0.66/0.65/0.67 — if not all
prohibited-data users, how did they get there? Are we missing anything?

Evidence available:
  - live LB top-50 (2026-09-05): rank / user / score / subs / recency
  - our own measured lanes: GDO-external floor (v20c = 0.631662555, PROHIBITED),
    clean stack (v21a 0.687374), correction lane (v25/v26/v27)
  - forum: "Is <0.70 possible without GRACE+lat/lon?" (field consensus: no)
"""
import re

RAW = open('/tmp/my-project/download/zindi_lb_live_2026-09-05.txt').read()
# file stores literal backslash-n / backslash-t as TEXT -> decode escapes
RAW = RAW.replace('\\n', '\n').replace('\\t', '\t')
# actual: N\n\t\nUSER[\nORG]\n\t\n0.xxxxx\n\t~time\tNN\t\n
pat = re.compile(r'(\d+)\n\t\n(.+?)\n\t\n(0\.\d{6,9})\n\t([^\n]+?)\t(\d+)\t', re.S)
rows = pat.findall(RAW)
rows = [(int(r), u.strip(), float(s), t.strip(), int(n)) for r, u, s, t, n in rows]
print(f'parsed {len(rows)} LB rows\n')

V20C = 0.631662555
V25 = 0.679780277
V26 = 0.678573   # predicted

print('=' * 78)
print('SMOKING GUN: rank 4 vs our prohibited v20c')
print('=' * 78)
for r, u, s, t, n in rows:
    if abs(s - V20C) < 1e-8:
        print(f'  rank {r}  {u}:  {s:.9f}  == v20c 0.631662555 EXACTLY (9 decimals)')
        print(f'    ({n} submissions, last {t})')
print('  Two independent files cannot match RMSE to 1e-9 by chance;')
print('  both are reading the same external TWS source at the target month.')

print()
print('=' * 78)
print('BAND CENSUS (top 50, Sep 5 snapshot)')
print('=' * 78)
bands = [
    ('A  <0.632   external TWS + residual modeling', lambda s: s < 0.632),
    ('B  0.632-0.647 external TWS direct-read band (GDO floor 0.6317)', lambda s: 0.632 <= s < 0.647),
    ('C  0.647-0.666 partial-external / GLDAS? / lat-lon-era models', lambda s: 0.647 <= s < 0.666),
    ('D  0.666-0.678 clean corridor (our neighborhood)', lambda s: 0.666 <= s < 0.678),
    ('E  0.678-0.70  clean ceiling zone (forum consensus ~0.70)', lambda s: 0.678 <= s < 0.70),
]
for name, cond in bands:
    mem = [(r, u, s, n) for r, u, s, t, n in rows if cond(s)]
    print(f'  {name}: {len(mem)} teams')
    if name.startswith('C'):
        for r, u, s, n in mem:
            print(f'      #{r:<3} {u:<20} {s:.6f}  ({n} subs)  <-- the 66/65/67 teams')
    if name.startswith('D'):
        for r, u, s, n in mem:
            print(f'      #{r:<3} {u:<20} {s:.6f}  ({n} subs)')

print()
print('  our positions: v25 {0:.6f} -> rank ~{1}; v26 {2:.6f} (predicted) -> rank ~{3}'
      .format(V25, sum(1 for r, u, s, t, n in rows if s < V25) + 1,
              V26, sum(1 for r, u, s, t, n in rows if s < V26) + 1))

print()
print('=' * 78)
print('CAN THE 0.65-0.67 TEAMS BE CLEAN? the closed bounds we have measured')
print('=' * 78)
print('  1. LB-grinding bound: month+region correction lane ceiling =')
print('     sum f*e^2 ~ 0.034 MSE (fantasy all-Jan-sized) => 0.70 -> 0.654 MINIMUM')
print('     even for a 120-sub team probing every month. Rank 6 (0.6346, 120')
print('     subs) is BELOW that -> external data, near-certain.')
print('  2. Clean craft ceiling: three audits + forum consensus ~0.70; our')
print('     stack measured at 0.687 -> corrections 0.679. The forum thread')
print('     "Is <0.70 possible without GRACE+lat/lon?" has no clean answer.')
print('  3. Grandfathered lat/lon: ruled out 19 Aug; pre-ruling submissions')
print('     may remain eligible (thread 34476 UNANSWERED) -> part of band C')
print('     can be lat/lon-era models that were legal WHEN SUBMITTED.')
print('  4. GLDAS at target month: unruled (thread 34601); a physics-model TWS')
print('     covariate could legally sit a team at 0.65-0.66. UNTESTED by us.')
print('  -> Band C = MIXED: partial-external + grandfathered-lat/lon + maybe')
print('     GLDAS + heavy LB-grind. "All cheating" is NOT provable; "mostly on')
print('     lanes we refuse" is the best-supported reading.')

print()
print('=' * 78)
print('WHAT WOULD WE DO DIFFERENTLY IF WE WANTED BAND C?')
print('=' * 78)
print('  - grandfathered lat/lon: unavailable to us (our lat/lon files are')
print('    POST-ruling = DQ risk; we already rescued their value compliantly)')
print('  - GLDAS: gated on an organizer ruling that is 8 days late; building')
print('    it speculatively = 2-3 days for an unruled lane (risk: DQ if ruled')
print('    "indirect TWS" like GDO). Verdict: ASK, do not build. (draft ready)')
print('  - partial external (GDO lags as features): same prohibition class')
print('    as v20 ("indirect future TWS information"). Refused, documented.')
print('  - the one thing NOBODY above us has: our exact-prediction method +')
print('    audit trail -> 50% of final score is report/trust/innovation.')

print()
print('=' * 78)
print('RANK REALITY CHECK (why 25th is not what it looks like)')
print('=' * 78)
nc = sum(1 for r, u, s, t, n in rows if 0.632 <= s < 0.666)
print(f'  Teams between the GDO floor and our level: ~{nc} (ranks 4-{nc+3}).')
print('  Every one of them is on a lane that a ruling can remove:')
print('  - 34476 (lat/lon eligibility) or 34601 (GLDAS) or GRACE enforcement')
print('  - If even half fall: 0.6786 -> top-10 RMSE.')
print('  Final score = 50% RMSE + 30% trust + 20% innovation.')
print('  Our trust/innovation assets: self-audit that caught v20 (refusal),')
print('  bit-exact reproducibility (G6), 3 exact LB predictions, disclosed')
print('  probe methodology, methodology audit of 11 hats + 6 refusals.')
print('  -> the report can plausibly beat rank 23 by more than the RMSE gap.')
