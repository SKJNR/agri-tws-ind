"""
GATE REVIEW v18 — reproduction, leakage probes, mirror validity, bootstrap,
final-phase end-to-end reproduction, spatial autocorrelation.

New file (gate review). Does NOT modify any existing file.
The pipeline functions (build_infra / get_smoother / run_masked_v18) are EXECUTED
VERBATIM by extracting their source from build_v18_final.py (authoritative script),
so the reproduction tests the actual shipped code, not a re-typed copy.
"""
import numpy as np, pandas as pd, time, warnings, re, sys
import lightgbm as lgb  # noqa (unused here, parity with source env)
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
warnings.filterwarnings('ignore', category=RuntimeWarning)

DATA = '/home/z/my-project/data'
DL = '/home/z/my-project/download'
SRC = '/home/z/my-project/scripts/build_v18_final.py'
COVS = ['SPEI_01_t','SPEI_03_t','SPEI_06_t','SPEI_12_t','SOIL_MOISTURE_t']
t0 = time.time()
OUT = []
def P(s=''):
    print(s, flush=True); OUT.append(str(s))
def log(msg): P(f"[{time.time()-t0:7.1f}s] {msg}")

# ============ 1. load data (verbatim semantics from build_v18_final.py) ============
log("loading train...")
train = pd.read_csv(f'{DATA}/Train (1).csv', usecols=['time','lat','lon','TWS_t','target']+COVS)
train['time'] = pd.to_datetime(train['time'])
for c in ['TWS_t','target']+COVS:
    train[c] = pd.to_numeric(train[c], errors='coerce').astype('float32')
train['cc'] = (train['lat'].round(2).astype(str)+'_'+train['lon'].round(2).astype(str)).astype('category').cat.codes.astype('int32')
train['ym'] = train['time'].dt.year*100 + train['time'].dt.month
train['t_abs'] = (train['ym']//100)*12 + (train['ym']%100) - 1
n_cells = int(train['cc'].max())+1
cell_xy = train[['cc','lat','lon']].drop_duplicates('cc').sort_values('cc')[['lat','lon']].values.astype(np.float64)

log("loading test...")
test = pd.read_csv(f'{DATA}/Test (2).csv')
test['time'] = pd.to_datetime(test['time'])
for c in ['TWS_t']+COVS:
    test[c] = pd.to_numeric(test[c], errors='coerce').astype('float32')
codes = train[['cc','lat','lon']].drop_duplicates('cc').sort_values('cc')
pos = {(int(round(float(la)*1000)), int(round(float(lo)*1000))): int(cc) for cc,la,lo in zip(codes['cc'],codes['lat'],codes['lon'])}
test['cc'] = [pos.get((int(round(float(a)*1000)), int(round(float(b)*1000))), -1) for a,b in zip(test['lat'], test['lon'])]
assert (test['cc'] >= 0).all()
test['ym'] = test['time'].dt.year*100 + test['time'].dt.month
test['t_abs'] = (test['ym']//100)*12 + (test['ym']%100) - 1
test['masked'] = test['TWS_t_masked'].astype(bool)

# ============ 2. exec VERBATIM function block from build_v18_final.py ============
src = open(SRC).read()
i0 = src.index('def build_infra')
i1 = src.index('# ---------- k0 machinery')
block = src[i0:i1]
ns = dict(np=np, pd=pd, gc=__import__('gc'), warnings=warnings, lgb=lgb,
          cKDTree=cKDTree, csr_matrix=csr_matrix, COVS=COVS, n_cells=n_cells,
          cell_xy=cell_xy, log=log, P=P, t0=t0, time=time)
exec(compile(block, SRC+'::<functions>', 'exec'), ns)
build_infra = ns['build_infra']; get_smoother = ns['get_smoother']; run_masked_v18 = ns['run_masked_v18']
P(f"[gate] executed verbatim function block from build_v18_final.py ({len(block.splitlines())} lines)")

# ============ 3. mirror month lists: verify vs SOURCE TEXT and auditA spec ============
m1a_src = re.search(r'M1_ANCHORS = \[([^\]]+)\]', src).group(1)
m1r_src = re.search(r'M1_RUNS = \{([^}]+)\}', src).group(1)
m2a_src = re.search(r'M2_ANCHORS = \[([^\]]+)\]', src).group(1)
m2r_src = re.search(r'M2_RUNS = \{([^}]+)\}', src).group(1)
M1_ANCHORS = [int(x) for x in m1a_src.split(',')]
M2_ANCHORS = [int(x) for x in m2a_src.split(',')]
M1_RUNS = {int(k): int(v) for k, v in re.findall(r'(\d+):(\d+)', m1r_src)}
M2_RUNS = {int(k): int(v) for k, v in re.findall(r'(\d+):(\d+)', m2r_src)}
P(f"\n[gate] month lists parsed from source: M1_ANCHORS={M1_ANCHORS}")
P(f"[gate]                              M2_ANCHORS={M2_ANCHORS}")
# auditA_report.md spec (hard expectation)
EXP_M1A = [24156, 24160, 24166, 24172, 24183, 24186]
EXP_M2A = [24156, 24160, 24166, 24172, 24186]
EXP_M1R = {24157:1, 24161:2, 24162:2, 24167:2, 24168:2, 24173:3, 24176:3, 24177:3, 24178:3, 24179:3, 24180:3, 24187:4}
EXP_M2R = dict(EXP_M1R); EXP_M2R.update({24181:3, 24182:3, 24183:3})
assert M1_ANCHORS == EXP_M1A and M2_ANCHORS == EXP_M2A, "anchor lists deviate from auditA spec!"
assert M1_RUNS == EXP_M1R and M2_RUNS == EXP_M2R, "run lists deviate from auditA spec!"
def t2date(t): y = t//12; m = t%12+1; return f"{y}-{m:02d}"
P("[gate] month-list match vs auditA_report.md §4: EXACT (anchors + runs)")
P("[gate] M1 anchors as dates: " + ", ".join(t2date(t) for t in M1_ANCHORS))
P("[gate] M2 anchors as dates: " + ", ".join(t2date(t) for t in M2_ANCHORS))
# auditA names: M1 = Jan13 May13 Nov13 May14 Apr15 Jul15 ; M2 drops Apr15
names_m1 = [t2date(t) for t in M1_ANCHORS]
assert names_m1 == ['2013-01','2013-05','2013-11','2014-05','2015-04','2015-07']

# ============ 4. infra_cv from fit<=2012 ONLY + leakage assertion ============
log("building infra_cv (fit<=2012)...")
fit = train[train['time'].dt.year <= 2012].copy()
assert fit['time'].dt.year.max() == 2012
infra_cv = build_infra(fit, 'cv')
mu12 = train[train['time'].dt.year <= 2012].groupby('cc')['TWS_t'].mean()
muall = train.groupby('cc')['TWS_t'].mean()
d12 = np.abs(infra_cv['mu_c'][:len(mu12)] - mu12.sort_index().values.astype(np.float32))
dall = np.abs(infra_cv['mu_c'][:len(muall)] - muall.sort_index().values.astype(np.float32))
P(f"[gate] infra_cv mu_c vs <=2012 per-cell mean: max|d|={d12.max():.2e} (float32 noise)")
P(f"[gate] infra_cv mu_c vs FULL-train per-cell mean: mean|d|={dall.mean():.4f} (must be >0 -> <=2012 filter active)")
assert dall.mean() > 0.005, "infra_cv looks like it saw full train!"

# ============ 5. mirrors + validity checks ============
log("building M1/M2 mirrors...")
val_all = train[(train['time'].dt.year >= 2013) & (train['time'].dt.year <= 2015)].copy()
val_tws_visible = val_all['TWS_t'].copy()
def make_mirror(anchors, runs):
    months = set(anchors) | set(runs)
    mir = val_all[val_all['t_abs'].isin(months)].copy()
    mir['masked'] = mir['t_abs'].isin(runs)
    mir['TWS_t'] = val_tws_visible.loc[mir.index].values
    mir.loc[mir['masked'], 'TWS_t'] = np.nan
    return mir
M1 = make_mirror(M1_ANCHORS, M1_RUNS); M2 = make_mirror(M2_ANCHORS, M2_RUNS)
for nm, mir, anchors, runs in [('M1', M1, M1_ANCHORS, M1_RUNS), ('M2', M2, M2_ANCHORS, M2_RUNS)]:
    mfrac = mir.groupby('t_abs')['masked'].mean()
    auto = sorted(int(v) for v in mfrac[mfrac < 0.01].index)
    assert auto == anchors, f"{nm} auto-anchors {auto} != hard-coded {anchors}"
    # TWS restore/null correctness vs original train values
    orig = train.loc[mir.index, 'TWS_t']
    an = mir['t_abs'].isin(anchors).values
    ok_anchor = np.allclose(mir['TWS_t'].values[an].astype(np.float64), orig.values[an].astype(np.float64), atol=1e-6, equal_nan=True)
    n_null = mir['TWS_t'].isna().sum(); n_msk = int(mir['masked'].sum())
    assert ok_anchor and n_null == n_msk
    P(f"[gate] {nm}: {len(mfrac)} months, auto-anchors==hard-coded, anchor TWS==train, "
      f"masked TWS nulled ({n_msk:,}); scored rows={n_msk:,}")
P(f"[gate] auditA expectation: M1 scored 187,302 / M2 scored 234,287")
# M1/M2 scored-row overlap (independence check)
k1 = set(zip(M1['cc'].values[M1['masked'].values].tolist(), M1['t_abs'].values[M1['masked'].values].tolist()))
k2 = set(zip(M2['cc'].values[M2['masked'].values].tolist(), M2['t_abs'].values[M2['masked'].values].tolist()))
inter = len(k1 & k2)
P(f"[gate] scored (cell,month) overlap: |M1|={len(k1):,} |M2|={len(k2):,} shared={inter:,} "
  f"-> {inter/len(k1)*100:.1f}% of M1 inside M2, {inter/len(k2)*100:.1f}% of M2 is M1")

def sc(mir, pred):
    mm = mir['masked'].values & np.isfinite(pred) & np.isfinite(mir['target'].values)
    return float(np.sqrt(np.mean((pred[mm]-mir['target'].values[mm])**2))), int(mm.sum())

# ============ 6. REPRODUCE headline numbers ============
P("\n================ REPRODUCTION (M1 / M2) ================")
BASE = dict(phi=0.80, use_bwd=True, dtil_w=(0.70,0.45,0.073), lam=0.84)
FINAL_CFGS = [
    dict(phi=0.80, lam=0.80, bwd_max_gap=8, dhat_tau=24, dhat_mode='full', smooth_sigma=2.0),
    dict(phi=0.85, lam=0.80, bwd_max_gap=8, dhat_tau=24, dhat_mode='full', smooth_sigma=2.0),
    dict(phi=0.80, lam=0.84, bwd_max_gap=8, dhat_tau=24, dhat_mode='full', smooth_sigma=2.0),
]
def ev(cfg, mir):
    kw = dict(cfg); phi = kw.pop('phi')
    return run_masked_v18(infra_cv, mir, mir['TWS_t'], phi, **kw)

pb1 = ev(BASE, M1); pb2 = ev(BASE, M2)
b1, n1 = sc(M1, pb1); b2, n2 = sc(M2, pb2)
P(f"BASE v17b : M1={b1:.4f} (n={n1:,}) | M2={b2:.4f} (n={n2:,})   [claimed 0.6319 / 0.6905]")
preds = []
for i, c in enumerate(FINAL_CFGS):
    p1 = ev(c, M1); p2 = ev(c, M2)
    s1, _ = sc(M1, p1); s2, _ = sc(M2, p2)
    preds.append((p1, p2))
    P(f"cfg{i}      : M1={s1:.4f} | M2={s2:.4f}   phi={c['phi']} lam={c['lam']}")
pe1 = np.mean([p[0] for p in preds], axis=0); pe2 = np.mean([p[1] for p in preds], axis=0)
e1, _ = sc(M1, pe1); e2, _ = sc(M2, pe2)
P(f"ENSEMBLE   : M1={e1:.4f} | M2={e2:.4f}   [claimed 0.6078 / 0.6694]")
EXP = [(0.6319, 0.6905), (0.6077, 0.6688), (0.6080, 0.6703), (0.6081, 0.6695), (0.6078, 0.6694)]
GOT = [(b1, b2)] + [tuple(sc(mir, p)[0] for mir, p in [(M1, pp[0]), (M2, pp[1])]) for pp in preds] + [(e1, e2)]
for (g1, g2), (x1, x2) in zip(GOT, EXP):
    P(f"[gate] check vs claim M1 {x1:.4f} (d={g1-x1:+.4f}) | M2 {x2:.4f} (d={g2-x2:+.4f})" +
      ("" if (abs(g1-x1) <= 0.0005 and abs(g2-x2) <= 0.0005) else "  <-- OUT OF TOLERANCE"))

# ============ 7. LEAKAGE PROBES ============
P("\n================ LEAKAGE PROBES ================")
# probe 1: corrupt targets of ALL rows -> predictions must be unchanged
M1c = M1.copy(); M1c['target'] = np.float32(917.0)
q1 = ev(BASE, M1c)
P(f"probe target-corruption (BASE):   max|dPred|={np.nanmax(np.abs(q1-pb1)):.2e}")
q1b = ev(FINAL_CFGS[0], M1c)
p0 = ev(FINAL_CFGS[0], M1)
P(f"probe target-corruption (cfg0):   max|dPred|={np.nanmax(np.abs(q1b-p0)):.2e}")
# probe 2: garbage TWS in masked months (should be invisible)
M1c2 = M1.copy(); M1c2.loc[M1c2['masked'], 'TWS_t'] = np.float32(917.0)
q2 = ev(BASE, M1c2); q2b = ev(FINAL_CFGS[0], M1c2)
P(f"probe masked-TWS-corruption (BASE): max|dPred|={np.nanmax(np.abs(q2-pb1)):.2e}")
P(f"probe masked-TWS-corruption (cfg0): max|dPred|={np.nanmax(np.abs(q2b-p0)):.2e}")
# probe 3: smoother kernel provenance — built from train cell coords only
K = get_smoother(2.0)
rs = np.asarray(K.sum(axis=1)).ravel()
P(f"probe smoother: shape={K.shape} (== n_cells train grid {n_cells}), row-sum min/max={rs.min():.6f}/{rs.max():.6f}")
d2self = np.abs(K.diagonal() - np.exp(0.0))
# weight of self ~ exp(0)/sum -> check self-weight in plausible range
P(f"probe smoother: self-weight mean={K.diagonal().mean():.4f} (sigma=2deg -> spread over many cells)")
assert K.shape == (n_cells, n_cells)
# grid spacing check
la = np.deg2rad(cell_xy[:,0]); lo = np.deg2rad(cell_xy[:,1])
pts = np.column_stack([np.cos(la)*np.cos(lo), np.cos(la)*np.sin(lo), np.sin(la)])
tree = cKDTree(pts)
dnn, _ = tree.query(pts, k=2)
ang = 2*np.arcsin(np.clip(dnn[:,1]/2, 0, 1))*180/np.pi
P(f"probe grid: NN great-circle deg: p5={np.percentile(ang,5):.3f} p50={np.percentile(ang,50):.3f} p95={np.percentile(ang,95):.3f} (1deg grid; E/W pairs shrink by cos(lat))")

# ============ 8. per-run-type + per-month decomposition ============
P("\n================ DECOMPOSITION (BASE vs cfg0) ================")
for nm, mir, runs, pb, pc in [('M1', M1, M1_RUNS, pb1, p0), ('M2', M2, M2_RUNS, pb2, ev(FINAL_CFGS[0], M2))]:
    msk = mir['masked'].values; y = mir['target'].values; ta = mir['t_abs'].values
    P(f"--- {nm} ---")
    for r in sorted(set(runs.values())):
        months = [m for m, rr in runs.items() if rr == r]
        sel = msk & np.isin(ta, months) & np.isfinite(y) & np.isfinite(pb) & np.isfinite(pc)
        rb = float(np.sqrt(np.mean((pb[sel]-y[sel])**2))); rc = float(np.sqrt(np.mean((pc[sel]-y[sel])**2)))
        P(f"  run{r} months={sorted(t2date(m) for m in months)}: BASE {rb:.4f} -> cfg0 {rc:.4f} ({rc-rb:+.4f})")
    for m in sorted(runs):
        sel = msk & (ta == m) & np.isfinite(y) & np.isfinite(pb) & np.isfinite(pc)
        rb = float(np.sqrt(np.mean((pb[sel]-y[sel])**2))); rc = float(np.sqrt(np.mean((pc[sel]-y[sel])**2)))
        P(f"    {t2date(m)}: BASE {rb:.4f} -> cfg0 {rc:.4f} ({rc-rb:+.4f})  n={sel.sum():,}")

# ============ 9. bootstrap SE (resample cells) + leave-one-month-out ============
P("\n================ BOOTSTRAP (cell clusters, 2000 draws) ================")
rng = np.random.default_rng(7)
def boot(mir, pa, pb_, B=2000):
    msk = mir['masked'].values & np.isfinite(pa) & np.isfinite(pb_) & np.isfinite(mir['target'].values)
    y = mir['target'].values[msk].astype(np.float64)
    cc = mir['cc'].values[msk]
    ea = (pa[msk]-y)**2; eb = (pb_[msk]-y)**2
    cells = np.unique(cc)
    sse_a = np.zeros(n_cells); sse_b = np.zeros(n_cells); cnt = np.zeros(n_cells)
    np.add.at(sse_a, cc, ea); np.add.at(sse_b, cc, eb); np.add.at(cnt, cc, 1)
    idx = cnt > 0
    sse_a, sse_b, cnt = sse_a[idx], sse_b[idx], cnt[idx]
    n_cl = len(cnt)
    ra = np.empty(B); rb_ = np.empty(B)
    for b in range(B):
        s = rng.integers(0, n_cl, n_cl)
        ra[b] = np.sqrt(sse_a[s].sum()/cnt[s].sum()); rb_[b] = np.sqrt(sse_b[s].sum()/cnt[s].sum())
    lvl_a = np.sqrt(sse_a.sum()/cnt.sum()); lvl_b = np.sqrt(sse_b.sum()/cnt.sum())
    return lvl_a, lvl_b, ra.std(), rb_.std(), (rb_-ra).std(), lvl_b-lvl_a, (rb_-ra)
for nm, mir, pa, pb_ in [('M1', M1, p0, pb1), ('M2', M2, ev(FINAL_CFGS[0], M2), pb2)]:
    la_, lb_, sa, sb, sd, d, dd = boot(mir, pa, pb_)
    lo, hi = np.percentile(dd, [2.5, 97.5])
    P(f"{nm}: cfg0={la_:.4f} (SE {sa:.4f})  BASE={lb_:.4f} (SE {sb:.4f})  "
      f"DIFF BASE-cfg0={d:+.4f} (SE {sd:.4f}, 95% CI [{lo:+.4f},{hi:+.4f}])")
P("[gate] leave-one-run-month-out jackknife of (BASE-cfg0) diff:")
for nm, mir, runs, pa, pb_ in [('M1', M1, M1_RUNS, p0, pb1), ('M2', M2, M2_RUNS, ev(FINAL_CFGS[0], M2), pb2)]:
    msk = mir['masked'].values & np.isfinite(pa) & np.isfinite(pb_) & np.isfinite(mir['target'].values)
    y = mir['target'].values; ta = mir['t_abs'].values
    full = float(np.sqrt(np.mean((pb_[msk]-y[msk])**2)) - np.sqrt(np.mean((pa[msk]-y[msk])**2)))
    vals = []
    for m in sorted(runs):
        s = msk & (ta != m)
        vals.append(float(np.sqrt(np.mean((pb_[s]-y[s])**2)) - np.sqrt(np.mean((pa[s]-y[s])**2))))
    vals = np.array(vals)
    P(f"  {nm}: full diff {full:+.4f}; LOO range [{vals.min():+.4f}, {vals.max():+.4f}], SD {vals.std(ddof=1):.4f}")

# ============ 10. FINAL PHASE end-to-end reproduction ============
P("\n================ FINAL-PHASE REPRODUCTION vs submission_v18a/v18b ================")
del fit, val_all, M1, M2, preds; import gc; gc.collect()
log("building infra_full (full train)...")
infra_full = build_infra(train, 'full')
test_for_pred = test.copy()
test_for_pred.loc[test_for_pred['masked'], 'TWS_t'] = np.nan
unm = ~test['masked'].values
final_masked = []
for c in FINAL_CFGS:
    kw = dict(c); phi = kw.pop('phi')
    p = run_masked_v18(infra_full, test_for_pred, test_for_pred['TWS_t'], phi, **kw)
    final_masked.append(p)
    log(f"  phi={phi} lam={kw['lam']}: masked pred std={np.nanstd(p):.4f}")
masked_ens = np.mean(final_masked, axis=0)
masked_single = final_masked[0]
sub = pd.read_csv(f'{DATA}/SampleSubmission (4).csv')
va = pd.read_csv(f'{DL}/submission_v18a.csv'); vb = pd.read_csv(f'{DL}/submission_v18b.csv')
v17b = pd.read_csv(f'{DL}/submission_v17b.csv')
for tag, mine, csvdf in [('v18a=ens', masked_ens, va), ('v18b=cfg0', masked_single, vb)]:
    p = mine.copy()
    mu_all = infra_full['mu_c'][test['cc'].values]
    p[~unm & np.isnan(p)] = mu_all[~unm & np.isnan(p)]   # replicate assemble() NaN->mu_c fallback
    d = np.abs(p[~unm].astype(np.float32).astype(np.float64) - csvdf['Target'].values[~unm].astype(np.float64))
    P(f"{tag}: masked rows reproduced (with mu_c fallback for NaN rows), max|d|={d.max():.2e}, mean|d|={d.mean():.2e}, n={d.size:,}")
d_ab = np.abs(va['Target'].values - vb['Target'].values)
P(f"v18a vs v18b: max|d| overall={d_ab.max():.2e}; unmasked max|d|={d_ab[unm].max():.2e} (k0 identical?)")
dk = np.abs(va['Target'].values[unm].astype(np.float64) - v17b['Target'].values[unm].astype(np.float64))
P(f"k0 CHECK v18a vs v17b on {unm.sum():,} unmasked rows: max|d|={dk.max():.2e}, mean|d|={dk.mean():.2e}, n>|1e-6|={(dk>1e-6).sum()}")
dk2 = np.abs(vb['Target'].values[unm].astype(np.float64) - v17b['Target'].values[unm].astype(np.float64))
P(f"k0 CHECK v18b vs v17b on unmasked rows: max|d|={dk2.max():.2e}, n>|1e-6|={(dk2>1e-6).sum()}")
# NaN coverage on masked rows in repro (fallback path exercised?)
P(f"repro masked NaN count: ens={int(np.isnan(masked_ens[~unm]).sum())}, cfg0={int(np.isnan(masked_single[~unm]).sum())}")

# ============ 11. spatial autocorrelation of prediction anomaly field ============
P("\n================ SPATIAL AUTOCORR (pred - mu_c, masked months) ================")
la_r = np.deg2rad(cell_xy[:,0]); lo_r = np.deg2rad(cell_xy[:,1])
pts = np.column_stack([np.cos(la_r)*np.cos(lo_r), np.cos(la_r)*np.sin(lo_r), np.sin(la_r)])
key = {(int(round(la*10)), int(round(lo*10))): i for i, (la, lo) in enumerate(cell_xy)}
pairs1, pairs2, pairs5 = [], [], []
for (lai, loi), i in key.items():
    for dla, dlo in [(0, 10), (10, 0)]:
        j = key.get((lai+dla, loi+dlo))
        if j is not None: pairs1.append((i, j))
    for dla, dlo in [(0, 20), (20, 0)]:
        j = key.get((lai+dla, loi+dlo))
        if j is not None: pairs2.append((i, j))
    for dla, dlo in [(0, 50), (50, 0)]:
        j = key.get((lai+dla, loi+dlo))
        if j is not None: pairs5.append((i, j))
pairs1 = np.array(pairs1); pairs2 = np.array(pairs2); pairs5 = np.array(pairs5)
P(f"pairs: 1deg={len(pairs1):,}  2deg={len(pairs2):,}  5deg={len(pairs5):,}")
mu = infra_full['mu_c']
def acf(pred_rows, cc_sel, pi):
    fld = np.full(n_cells, np.nan)
    fld[cc_sel] = pred_rows - mu[cc_sel]
    i, j = pi[:, 0], pi[:, 1]
    ok = np.isfinite(fld[i]) & np.isfinite(fld[j])
    return float(np.corrcoef(fld[i[ok]], fld[j[ok]])[0, 1]), int(ok.sum())
msk_months = sorted(set(test['t_abs'].values[test['masked'].values].tolist()))
ta_t = test['t_abs'].values; cc_t = test['cc'].values
for tag, field in [('v18a-ens', masked_ens), ('v18b-cfg0', masked_single)]:
    r1s, r2s, r5s = [], [], []
    for m in msk_months:
        sel = (ta_t == m) & (~unm)
        r1s.append(acf(field[sel], cc_t[sel], pairs1)[0])
        r2s.append(acf(field[sel], cc_t[sel], pairs2)[0])
        r5s.append(acf(field[sel], cc_t[sel], pairs5)[0])
    P(f"{tag}: 1deg autocorr per masked month min/mean={min(r1s):.4f}/{np.mean(r1s):.4f} | "
      f"2deg {np.mean(r2s):.4f} | 5deg {np.mean(r5s):.4f}")
# v17b comparison from CSV
v17b_masked = v17b['Target'].values[~unm].astype(np.float64)
ta_m = ta_t[~unm]; cc_m = cc_t[~unm]
r1s = []
for m in msk_months:
    sel = ta_m == m
    r1s.append(acf(v17b_masked[sel], cc_m[sel], pairs1)[0])
P(f"v17b (submitted): 1deg autocorr min/mean={min(r1s):.4f}/{np.mean(r1s):.4f}")

with open('/home/z/my-project/scripts/gate_repro_out.txt', 'w') as f:
    f.write('\n'.join(OUT))
P("\nGATE REPRO DONE")
