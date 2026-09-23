"""
AUDIT A1 — TEST STRUCTURE CENSUS (Test (2).csv ONLY, no Train loaded)
Outputs: /home/z/my-project/download/auditA_test_census.txt (+ .json)
"""
import numpy as np, pandas as pd, json

DATA = '/home/z/my-project/data'
OUT = '/home/z/my-project/download/auditA_test_census.txt'
lines = []
def P(s=''):
    print(s, flush=True); lines.append(str(s))

test = pd.read_csv(f'{DATA}/Test (2).csv')
test['time'] = pd.to_datetime(test['time'])
test['masked'] = test['TWS_t_masked'].astype(bool)

# ---- basic identity checks ----
P("=== A1.0 BASIC ===")
P(f"rows: {len(test):,}   cols: {list(test.columns)}")
P(f"masked=True: {int(test['masked'].sum()):,} ({test['masked'].mean()*100:.2f}%)  masked=False: {int((~test['masked']).sum()):,}")
# is masked <=> TWS_t NaN?
tws_nan = test['TWS_t'].isna()
P(f"TWS_t NaN rows: {int(tws_nan.sum()):,} | masked==NaN agreement: {(test['masked']==tws_nan).all()}")
P(f"mask agreement matrix (masked vs TWS-NaN): {pd.crosstab(test['masked'], tws_nan).to_dict()}")
# covariate availability on masked rows
COVS = ['SPEI_01_t','SPEI_03_t','SPEI_06_t','SPEI_12_t','SOIL_MOISTURE_t']
for c in COVS:
    P(f"  {c}: NaN on masked={int(test.loc[test['masked'],c].isna().sum()):,}/{int(test['masked'].sum()):,}, "
      f"NaN on unmasked={int(test.loc[~test['masked'],c].isna().sum()):,}")

# ---- time encoding ----
test['ym'] = test['time'].dt.year*100 + test['time'].dt.month
test['t_abs'] = (test['ym']//100)*12 + (test['ym']%100) - 1   # matches builder
test['cal_m'] = test['time'].dt.month
# cell code
test['cc'] = (np.round(test['lat']*10).astype(np.int64)+1000)*100000 + (np.round(test['lon']*10).astype(np.int64)+5000)

P("\n=== A1.1 PER-MONTH CENSUS (t_abs) ===")
g = test.groupby('t_abs').agg(rows=('masked','size'), n_masked=('masked','sum'), cal=('cal_m','first'))
g['frac'] = g['n_masked']/g['rows']
g['date'] = [f"{int(t)//12}-{int(t)%12+1:02d}" for t in g.index]
P(f"distinct t_abs months in test: {len(g)}")
P(f"{'t_abs':>6} {'date':>8} {'cal':>3} {'rows':>7} {'masked':>7} {'frac':>6}  role")
for t, r in g.iterrows():
    role = 'ANCHOR(<0.01)' if r['frac'] < 0.01 else ('FULL-MASK(>=0.99)' if r['frac'] >= 0.99 else 'PARTIAL')
    P(f"{int(t):>6} {r['date']:>8} {int(r['cal']):>3} {int(r['rows']):>7} {int(r['n_masked']):>7} {r['frac']:>6.3f}  {role}")

# gaps between consecutive present months
present = np.array(sorted(g.index))
all_span = np.arange(present[0], present[-1]+1)
absent = sorted(set(all_span) - set(present.tolist()))
P(f"\ntest span: t_abs {present[0]}..{present[-1]}  = {g['date'].iloc[0]} .. {g['date'].iloc[-1]}")
P(f"train last month = 2015-08 = t_abs {2015*12+8-1}; test starts {int(present[0])} "
  f"({int(present[0])-(2015*12+8-1)} month(s) after train end)")
P(f"ABSENT t_abs months inside span ({len(absent)}): {[(int(a), f'{int(a)//12}-{int(a)%12+1:02d}') for a in absent]}")
P(f"absent calendar months (of present months' cal): {sorted(set(range(1,13)) - set(g['cal'].unique().tolist()))}")

P("\n=== A1.2 POOLED CALENDAR-MONTH MASK FRACTIONS ===")
gc = test.groupby('cal_m').agg(rows=('masked','size'), n_masked=('masked','sum'))
gc['frac'] = gc['n_masked']/gc['rows']
for m, r in gc.iterrows():
    P(f"cal month {int(m):>2}: rows={int(r['rows']):>7}  masked={int(r['n_masked']):>7}  frac={r['frac']:.4f}")

# anchors
anchors = sorted(int(t) for t in g.index[g['frac'] < 0.01])
P(f"\nANCHOR months (frac<0.01): {[(a, f'{a//12}-{a%12+1:02d}') for a in anchors]}")
P(f"anchor gaps (months): {[int(anchors[i+1]-anchors[i]) for i in range(len(anchors)-1)]}")
# partial months
partial = sorted(int(t) for t in g.index[(g['frac'] >= 0.01) & (g['frac'] < 0.99)])
P(f"PARTIAL months (0.01<=frac<0.99): {len(partial)}")
for t in partial:
    r = g.loc[t]
    P(f"  t_abs {int(t)} ({int(t)//12}-{int(t)%12+1:02d}) cal={int(r['cal'])}: rows={int(r['rows']):,} masked_frac={r['frac']:.4f} visible_cells={int(r['rows']-r['n_masked']):,}")
fullmask = sorted(int(t) for t in g.index[g['frac'] >= 0.99])
P(f"FULL-MASK months: {len(fullmask)}: {[(t, f'{t//12}-{t%12+1:02d}') for t in fullmask]}")

# ---- cells ----
P("\n=== A1.3 PER-CELL STRUCTURE ===")
ncell = test['cc'].nunique()
P(f"distinct cells in test: {ncell}")
rpc = test.groupby('cc').size()
P(f"rows/cell: mean={rpc.mean():.2f} p5={np.percentile(rpc,5):.0f} p25={np.percentile(rpc,25):.0f} "
  f"p50={np.percentile(rpc,50):.0f} p75={np.percentile(rpc,75):.0f} p95={np.percentile(rpc,95):.0f} min={rpc.min()} max={rpc.max()}")
upc = test[~test['masked']].groupby('cc').size().reindex(rpc.index, fill_value=0)
P(f"unmasked obs/cell: mean={upc.mean():.2f} p5={np.percentile(upc,5):.0f} p25={np.percentile(upc,25):.0f} "
  f"p50={np.percentile(upc,50):.0f} p75={np.percentile(upc,75):.0f} p95={np.percentile(upc,95):.0f} min={upc.min()} max={upc.max()}")
P(f"unmasked obs/cell value counts: {dict(sorted(upc.value_counts().items()))}")
# anchors per cell
apc = test[(~test['masked']) & test['t_abs'].isin(anchors)].groupby('cc').size().reindex(rpc.index, fill_value=0)
P(f"anchor-month obs/cell: mean={apc.mean():.2f} min={apc.min()} max={apc.max()}  (n anchors={len(anchors)})")

# per-cell gaps between consecutive unmasked obs
uv = test[~test['masked']].sort_values(['cc','t_abs'])
u_cc = uv['cc'].values; u_ta = uv['t_abs'].values
newc = np.empty(len(uv), dtype=bool); newc[0] = True; newc[1:] = u_cc[1:] != u_cc[:-1]
gaps_u = (u_ta[1:] - u_ta[:-1])[~newc[1:]]
P(f"\nconsecutive-unmasked-obs gaps (months, within test): n={len(gaps_u):,}")
if len(gaps_u):
    P(f"  p25={np.percentile(gaps_u,25):.0f} p50={np.percentile(gaps_u,50):.0f} p75={np.percentile(gaps_u,75):.0f} "
      f"p90={np.percentile(gaps_u,90):.0f} p95={np.percentile(gaps_u,95):.0f} max={gaps_u.max()}")
    P(f"  value counts: {dict(sorted(pd.Series(gaps_u).value_counts().items()))}")

# ---- horizons for masked rows ----
P("\n=== A1.4 MASKED-ROW HORIZONS (relative to unmasked obs of same cell, TEST-only) ===")
# per-cell sorted unmasked t_abs
cell_unmasked = {}
for cc, ta in zip(u_cc, u_ta):
    cell_unmasked.setdefault(cc, []).append(int(ta))
for k in cell_unmasked: cell_unmasked[k] = np.array(cell_unmasked[k], dtype=np.int64)

mv = test[test['masked']]
h_back = np.full(len(mv), -99, dtype=np.int64)   # months since last unmasked (strictly before t)
h_fwd = np.full(len(mv), -99, dtype=np.int64)    # months until next unmasked (strictly after t)
ccm = mv['cc'].values; tam = mv['t_abs'].values
for i in range(len(mv)):
    arr = cell_unmasked.get(ccm[i])
    if arr is None or len(arr) == 0: continue
    j = np.searchsorted(arr, tam[i])
    if j > 0: h_back[i] = tam[i] - arr[j-1]
    if j < len(arr): h_fwd[i] = arr[j] - tam[i]
P(f"masked rows with NO prior unmasked obs in test: {(h_back<0).sum():,}")
hb = h_back[h_back > 0]
P(f"horizon_back (t - last unmasked): n={len(hb):,}")
P(f"  value counts: {dict(sorted(pd.Series(hb).value_counts().items()))}")
P(f"  p50={np.percentile(hb,50):.0f} p90={np.percentile(hb,90):.0f} max={hb.max()}")
hf = h_fwd[h_fwd > 0]
P(f"months-until-next-unmasked (next unmasked - t): n={len(hf):,} (no next: {(h_fwd<0).sum():,})")
P(f"  value counts: {dict(sorted(pd.Series(hf).value_counts().items()))}")
P(f"  p50={np.percentile(hf,50):.0f} p90={np.percentile(hf,90):.0f} max={hf.max()}")

# horizon for scored rows in terms of target month: tm = t+1; forward gap = h_fwd-1, backward gap = h_back+1
P("\n-- in TARGET-month units (tm=t+1): dist from tm back to last unmasked = h_back+1; from tm fwd to next unmasked = h_fwd-1")
P(f"  fwd-from-target value counts: {dict(sorted(pd.Series(hf-1).value_counts().items()))}")
P(f"  back-from-target value counts: {dict(sorted(pd.Series(hb+1).value_counts().items()))}")

# ---- A1.5 determinism of masking ----
P("\n=== A1.5 MASK-ASSIGNMENT DETERMINISM ===")
# For each partial month: visible cell set. Compare across months.
vis_sets = {}
for t in partial:
    vis = set(test.loc[(test['t_abs']==t) & (~test['masked']), 'cc'].tolist())
    vis_sets[t] = vis
if len(partial) >= 2:
    names = [f"{t//12}-{t%12+1:02d}" for t in partial]
    P("pairwise Jaccard of visible-cell sets across PARTIAL months:")
    for i in range(len(partial)):
        for j in range(i+1, len(partial)):
            a, b = vis_sets[partial[i]], vis_sets[partial[j]]
            jac = len(a & b)/max(len(a | b), 1)
            P(f"  {names[i]} vs {names[j]}: |A|={len(a)} |B|={len(b)} inter={len(a&b)} J={jac:.3f}")
    allv = set.union(*vis_sets.values())
    P(f"union of visible cells across partial months: {len(allv)}")

# same calendar month across years: is the ROW SET (cells present) stable? is the masked set stable?
P("\nsame-calendar-month instances (rows present, frac) — checking cell-level stability:")
for cal in sorted(g['cal'].unique()):
    ts = [int(t) for t in g.index[g['cal']==cal]]
    if len(ts) >= 2:
        rowsets = [set(test.loc[test['t_abs']==t, 'cc'].tolist()) for t in ts]
        base = rowsets[0]
        for k, t in enumerate(ts):
            P(f"  cal {int(cal)} t_abs {t} ({t//12}-{t%12+1:02d}): rows={len(rowsets[k])} frac={g.loc[t,'frac']:.3f} "
              f"J_vs_first={len(rowsets[k]&base)/max(len(rowsets[k]|base),1):.3f}")

# parity / hash tests on a partial month (if any) and on anchor membership
if partial:
    t0 = partial[0]
    sub = test[test['t_abs']==t0]
    sub_v = sub[~sub['masked']]
    la = np.round(sub_v['lat'].values*2).astype(int); lo = np.round(sub_v['lon'].values*2).astype(int)
    P(f"\nparity test on visible cells of {t0}: lat*2%2 = {pd.Series(la%2).value_counts().to_dict()}, lon*2%2={pd.Series(lo%2).value_counts().to_dict()}")
    P(f"  lat range visible: {sub_v['lat'].min()}..{sub_v['lat'].max()}; all-cells lat range: {test['lat'].min()}..{test['lat'].max()}")
# anchors: are all cells present at every anchor?
for a in anchors:
    n_rows_a = int(g.loc[a,'rows'])
    P(f"anchor {a} ({a//12}-{a%12+1:02d}): rows={n_rows_a} (vs ncell={ncell})")

# ---- A1.6 consecutive-month structure: does (cell, t+1) exist & visible for masked rows? (direct-copy check) ----
P("\n=== A1.6 DIRECT-COPY SURFACE (masked row whose (cell,t+1) row exists & visible) ===")
vis_key = set(zip(test.loc[~test['masked'],'cc'].tolist(), (test.loc[~test['masked'],'t_abs']).tolist()))
keys = list(zip(ccm.tolist(), (tam+1).tolist()))
n_copy = sum(1 for k in keys if k in vis_key)
P(f"masked rows with (cell,t+1) visible in test: {n_copy:,}")
# also: rows present at t+1 at all (masked or not)
pres_key = set(zip(test['cc'].tolist(), test['t_abs'].tolist()))
n_pres = sum(1 for k in keys if k in pres_key)
P(f"masked rows with (cell,t+1) PRESENT in test (any mask state): {n_pres:,}")

# consecutive month cell-set overlap in general
P("\nconsecutive-present-month cell-set overlaps (all months):")
for i in range(len(present)-1):
    a, b = present[i], present[i+1]
    if b - a > 1: continue
    sa = set(test.loc[test['t_abs']==a,'cc'].tolist()); sb = set(test.loc[test['t_abs']==b,'cc'].tolist())
    P(f"  {a//12}-{a%12+1:02d} -> {b//12}-{b%12+1:02d}: |A|={len(sa)} |B|={len(sb)} inter={len(sa&sb)} J={len(sa&sb)/max(len(sa|sb),1):.3f}")

# save
with open(OUT, 'w') as f:
    f.write('\n'.join(lines))
print(f"\nsaved {OUT}")
