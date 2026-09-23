"""
DECISIVE MEASUREMENTS for the cross-AI debate.

Settles:
  D1: corr(TWS_t, target) — is it 0.9766 (old EDA claim) or 0.746 (field ACF)?
      And is target(t) == TWS_t(t+1) exactly?
  D2: Long-lag train ACF k=13..36 — does train also show a floor?
  D3: Spatial structure: var(mu_c) vs within-cell var; neighbor corr of anomaly
      field at 1/2/5/10 deg separations; spatial PCA of anomaly field.
  D4: How much of y(t) do same-month covariates explain (honest fit/eval split)?
      Field-level r and PC-score-level r (spatial estimation quality).
  D5: Target decomposition (honest): y(t) alone / covs(t) / y(t)+covs(t) /
      y(t)+covs(t)+covs(t+1)  <- ceiling if we use next-row covariates
  D6: Proper common-mode test: remove top-J field PCs from anchor fields,
      recompute pair correlations.
"""
import numpy as np, pandas as pd

DATA = '/home/z/my-project/data'
OUT = '/home/z/my-project/download/decisive_measurements.txt'
COVS = ['SPEI_01_t','SPEI_03_t','SPEI_06_t','SPEI_12_t','SOIL_MOISTURE_t']
log = []
def P(s=''):
    print(s, flush=True)
    log.append(str(s))

# ================= load =================
P("=== loading train (10 cols) ===")
train = pd.read_csv(f'{DATA}/Train (1).csv',
                    usecols=['time','lat','lon','TWS_t','target']+COVS)
train['time'] = pd.to_datetime(train['time'])
for c in ['TWS_t','target']+COVS:
    train[c] = pd.to_numeric(train[c], errors='coerce').astype('float32')
train['cc'] = (train['lat'].round(2).astype(str)+'_'+train['lon'].round(2).astype(str)).astype('category').cat.codes.astype('int32')
train['ym'] = train['time'].dt.year*100 + train['time'].dt.month
n_cells = int(train['cc'].max())+1
yms = np.sort(train['ym'].unique())
ym_to_i = {int(v):i for i,v in enumerate(yms)}
T = len(yms)
P(f"train: {T} months ({yms[0]}..{yms[-1]}), {n_cells} cells, {len(train)} rows")

# ================= D1: target identity =================
P("\n=== D1: what is the target? ===")
d = train[['cc','ym','TWS_t','target']].copy()
d = d.sort_values(['cc','ym'])
same = d['TWS_t'].values
tgt = d['target'].values
r_same = float(np.corrcoef(same[~np.isnan(tgt)], tgt[~np.isnan(tgt)])[0,1])
P(f"corr(TWS_t, target) same row = {r_same:.4f}   [old EDA claim: 0.9766]")
# next month's TWS_t within cell
nxt = d.groupby('cc')['TWS_t'].shift(-1)
ymv = d['ym'].values; ccv = d['cc'].values
nxt_ym = d.groupby('cc')['ym'].shift(-1)
consec = ((nxt_ym - ymv) == 1) & nxt.notna() & d['TWS_t'].notna() & d['target'].notna()
r_next = float(np.corrcoef(nxt.values[consec.values], tgt[consec.values])[0,1])
diff_next = nxt.values[consec.values] - tgt[consec.values]
P(f"corr(TWS_t(t+1), target(t)) = {r_next:.6f}   [if ~1.0, target IS next month's TWS_t]")
P(f"  std of (TWS_t(t+1) - target) = {np.nanstd(diff_next):.6f}, max|diff| = {np.nanmax(np.abs(diff_next)):.6f}")
prev = d.groupby('cc')['TWS_t'].shift(1)
r_prev = float(np.corrcoef(prev.values[consec.values], tgt[consec.values])[0,1])
P(f"corr(TWS_t(t-1), target(t)) = {r_prev:.4f}")

# ================= D2: long-lag ACF =================
P("\n=== D2: train field ACF, long lags ===")
F = np.full((T, n_cells), np.nan, dtype=np.float32)
F[train['ym'].map(ym_to_i).values, train['cc'].values] = train['TWS_t'].values
mu_c = np.nanmean(F, axis=0)
A = F - mu_c
for k in [12,18,24,30,36,48]:
    if k >= T: break
    x, y = A[:-k], A[k:]
    ok = np.isfinite(x) & np.isfinite(y)
    r = float(np.corrcoef(x[ok].ravel(), y[ok].ravel())[0,1])
    P(f"  r({k:2d}) = {r:.4f}   [AR(1) 0.839*0.89^{k} = {0.839*0.89**k:.4f}]")

# ================= D3: spatial structure =================
P("\n=== D3: spatial structure ===")
var_y = float(np.nanvar(A))
var_mu = float(np.var(mu_c))
P(f"within-cell anomaly var = {var_y:.4f}; spatial var of cell means = {var_mu:.4f} (ratio {var_mu/var_y:.3f})")
codes = train[['cc','lat','lon']].drop_duplicates('cc').sort_values('cc')
la = codes['lat'].values; lo = codes['lon'].values
pos = {(int(round(float(la[i])*1000)), int(round(float(lo[i])*1000))): i for i in range(len(la))}
for sep_deg in [1,2,5,10]:
    s = int(sep_deg*1000)
    ia, ib = [], []
    for i in range(len(la)):
        j = pos.get((int(round(float(la[i])*1000)), int(round(float(lo[i])*1000))+s))
        if j is not None: ia.append(i); ib.append(j)
    if len(ia) < 100:
        P(f"  {sep_deg} deg east: too few pairs")
        continue
    ia = np.array(ia); ib = np.array(ib)
    x = A[:, ia]; y = A[:, ib]
    ok = np.isfinite(x) & np.isfinite(y)
    r = float(np.corrcoef(x[ok].ravel(), y[ok].ravel())[0,1])
    P(f"  same-time neighbor corr at {sep_deg} deg east: {r:.4f} (n_pairs={len(ia)})")

# spatial PCA of anomaly field
full = ~np.isnan(A).any(axis=0)
Af = A[:, full] - np.nanmean(A[:, full], axis=0, keepdims=True)
U, S, Vt = np.linalg.svd(Af, full_matrices=False)
v = S**2/(S**2).sum()
P(f"spatial PCA of anomaly field: PC1={v[0]*100:.1f}%, PC1-5={v[:5].sum()*100:.1f}%, PC1-10={v[:10].sum()*100:.1f}%, PC1-20={v[:20].sum()*100:.1f}%, PC1-50={v[:50].sum()*100:.1f}%")

# ================= D4: same-month cov -> y regression =================
P("\n=== D4: same-month covariates -> TWS_t (honest: fit <2010, eval >=2010) ===")
Z = train[COVS].values.astype(np.float32)
yv = train['TWS_t'].values.astype(np.float32)
fitm = (train['time'] < '2010-01-01').values
eva = ~fitm
Zf = np.column_stack([Z[fitm], np.ones(fitm.sum())])
zf = yv[fitm]
okf = np.isfinite(Zf).all(axis=1) & np.isfinite(zf)
coef = np.linalg.solve(Zf[okf].T@Zf[okf] + 1e-2*np.eye(6), Zf[okf].T@zf[okf])
Ze = np.column_stack([Z[eva], np.ones(eva.sum())])
ze = yv[eva]
oke = np.isfinite(Ze).all(axis=1) & np.isfinite(ze)
pred = Ze[oke] @ coef
r_cell = float(np.corrcoef(pred, ze[oke])[0,1])
rmse_cell = float(np.sqrt(np.mean((pred-ze[oke])**2)))
P(f"per-cell: corr(cov-pred, TWS_t) = {r_cell:.4f}, RMSE = {rmse_cell:.4f}, R2 = {r_cell**2:.3f}")

# field-level: per eval month, corr across cells between cov-pred field and actual field
P("field-level quality of cov estimate, per month (eval period):")
ev_months = np.sort(train.loc[eva, 'ym'].unique())
# build pred field and actual field per month
fld_r = []
for m in ev_months[::6]:
    sel = (train['ym'].values == m)
    if sel.sum() < 5000: continue
    pm = np.full(n_cells, np.nan, dtype=np.float32)
    am = np.full(n_cells, np.nan, dtype=np.float32)
    cc_m = train['cc'].values[sel]
    Zm = np.column_stack([Z[sel], np.ones(sel.sum())])
    okm = np.isfinite(Zm).all(axis=1) & np.isfinite(yv[sel])
    pm[cc_m[okm]] = Zm[okm] @ coef
    am[cc_m[okm]] = yv[sel][okm]
    okk = np.isfinite(pm) & np.isfinite(am)
    if okk.sum() > 5000:
        fld_r.append(float(np.corrcoef(pm[okk]-np.nanmean(pm[okk]), am[okk]-np.nanmean(am[okk]))[0,1]))
fld_r = np.array(fld_r)
P(f"  field-level anomaly corr: mean={fld_r.mean():.4f}, min={fld_r.min():.4f}, max={fld_r.max():.4f} (n_months={len(fld_r)})")

# PC-score-level: how well are top field PCs estimated from covs?
P("PC-score estimation quality (top 5 PCs of anomaly field, eval months):")
topJ = 5
V = Vt[:topJ].T  # (cells_full, topJ) loadings
score_rs = [[] for _ in range(topJ)]
for m in ev_months:
    sel = (train['ym'].values == m)
    if sel.sum() < 5000: continue
    pm = np.full(n_cells, np.nan, dtype=np.float32)
    am = np.full(n_cells, np.nan, dtype=np.float32)
    cc_m = train['cc'].values[sel]
    Zm = np.column_stack([Z[sel], np.ones(sel.sum())])
    okm = np.isfinite(Zm).all(axis=1) & np.isfinite(yv[sel])
    pm[cc_m[okm]] = Zm[okm] @ coef
    am[cc_m[okm]] = yv[sel][okm]
    pf = pm[full]; af = am[full]          # restrict to full cells (14558)
    okk = np.isfinite(pf) & np.isfinite(af)
    if okk.sum() < 5000: continue
    sp = (pf[okk]-pf[okk].mean()) @ V[okk]   # (topJ,) predicted PC scores
    sa = (af[okk]-af[okk].mean()) @ V[okk]   # (topJ,) actual PC scores
    for j in range(topJ):
        score_rs[j].append((float(sp[j]), float(sa[j])))
for j in range(topJ):
    if len(score_rs[j]) > 5:
        xs = np.array([p[0] for p in score_rs[j]]); ys = np.array([p[1] for p in score_rs[j]])
        if np.std(xs) > 0 and np.std(ys) > 0:
            P(f"  PC{j+1} score corr across months: {np.corrcoef(xs, ys)[0,1]:.4f} (n={len(xs)} months)")

# ================= D6: test-era persistent component D(c) =================
P("\n=== D6: test-era persistent component (anchor fields) ===")
test = pd.read_csv(f'{DATA}/Test (2).csv', usecols=['time','lat','lon','TWS_t','TWS_t_masked'])
test['time'] = pd.to_datetime(test['time'])
test['TWS_t'] = pd.to_numeric(test['TWS_t'], errors='coerce').astype('float32')
test['cc'] = [pos.get((int(round(float(a)*1000)), int(round(float(b)*1000))), -1) for a,b in zip(test['lat'], test['lon'])]
test['ym'] = test['time'].dt.year*100 + test['time'].dt.month
test['masked'] = test['TWS_t_masked'].astype(bool)
mfrac = test.groupby('ym')['masked'].mean()
anchors = sorted(mfrac[mfrac < 0.01].index.tolist())
P(f"anchors: {anchors}")
AF = []
for a in anchors:
    sel = (test['ym']==a) & (~test['masked'])
    fa = np.full(n_cells, np.nan, dtype=np.float32)
    fa[test['cc'].values[sel]] = test['TWS_t'].values[sel]
    AF.append(fa - mu_c)
AF = np.array(AF)  # (6, n_cells) anchor anomaly fields
# D-hat: mean anchor field
Dhat = np.nanmean(AF, axis=0)
var_anchor = float(np.nanvar(AF))
var_D = float(np.nanvar(Dhat))
P(f"var(anchor anomaly) = {var_anchor:.4f}; var(D-hat=mean of 6 anchors) = {var_D:.4f}; ratio = {var_D/var_anchor:.3f}")
P(f"std(D-hat) = {np.sqrt(var_D):.4f}  <- test-era mean shift per cell (vs train mean)")
# PCA of the 6 anchor fields: PC1 = shared persistent pattern
AFc = AF - np.nanmean(AF, axis=1, keepdims=True)
Ua, Sa, Vta = np.linalg.svd(AFc[:, ~np.isnan(AFc).any(axis=0)], full_matrices=False)
va = Sa**2/(Sa**2).sum()
P(f"PCA of 6 anchor fields: PC1={va[0]*100:.1f}%, PC2={va[1]*100:.1f}%, PC3={va[2]*100:.1f}%")
# pair correlations after removing D-hat
P("pair corrs after removing D-hat (fast component only):")
for i in range(len(anchors)):
    for j in range(i+1, len(anchors)):
        k = (anchors[j]//100-anchors[i]//100)*12 + (anchors[j]%100-anchors[i]%100)
        xa, xb = AF[i]-Dhat, AF[j]-Dhat
        ok = np.isfinite(xa) & np.isfinite(xb)
        r0 = float(np.corrcoef(AF[i][ok], AF[j][ok])[0,1])
        r1 = float(np.corrcoef(xa[ok], xb[ok])[0,1])
        P(f"  {anchors[i]}->{anchors[j]} (k={k:2d}): r={r0:.4f} -> D-removed {r1:.4f}")
# how much of D-hat lives in the train top-50 PC subspace?
full_idx = np.where(full)[0]
Df = Dhat[full]
okD = np.isfinite(Df)
Vfull = Vt[:50].T  # (14558, 50)
Dc = Df[okD] - np.mean(Df[okD])
proj = Vfull[okD] @ (Vfull[okD].T @ Dc)
P(f"fraction of D-hat variance inside train top-50 PC subspace: {np.var(proj)/np.var(Dc):.3f}")

# ================= D5: target decomposition =================
P("\n=== D5: what predicts the target? (honest: fit <2010, eval >=2010) ===")
# construct next-month covs within cell
train_s = train.sort_values(['cc','ym'])
for c in COVS:
    train_s[c+'_nxt'] = train_s.groupby('cc')[c].shift(-1)
consec = (train_s.groupby('cc')['ym'].shift(-1) - train_s['ym']) == 1
train_s['ok'] = consec & train_s['TWS_t'].notna() & train_s['target'].notna()
fitm2 = (train_s['time'] < '2010-01-01').values & train_s['ok'].values
evam2 = (~((train_s['time'] < '2010-01-01').values)) & train_s['ok'].values

def eval_model(feats, name):
    Xf = np.column_stack([train_s[f].values[fitm2] for f in feats] + [np.ones(fitm2.sum())])
    yf = train_s['target'].values[fitm2]
    okf2 = np.isfinite(Xf).all(axis=1)
    cf = np.linalg.solve(Xf[okf2].T@Xf[okf2] + 1e-2*np.eye(Xf.shape[1]), Xf[okf2].T@yf[okf2])
    Xe = np.column_stack([train_s[f].values[evam2] for f in feats] + [np.ones(evam2.sum())])
    ye = train_s['target'].values[evam2]
    oke2 = np.isfinite(Xe).all(axis=1)
    pe = Xe[oke2] @ cf
    r = float(np.corrcoef(pe, ye[oke2])[0,1])
    rm = float(np.sqrt(np.mean((pe-ye[oke2])**2)))
    P(f"  {name:28s}: corr={r:.4f}  RMSE={rm:.4f}")

eval_model(['TWS_t'], 'TWS_t alone (persistence)')
eval_model(COVS, 'covs(t) alone')
eval_model(['TWS_t']+COVS, 'TWS_t + covs(t)')
eval_model(['TWS_t']+COVS+[c+'_nxt' for c in COVS], 'TWS_t + covs(t) + covs(t+1)!!')
eval_model([c+'_nxt' for c in COVS], 'covs(t+1) alone!!')

# baseline: predict 0 anomaly
ye = train_s['target'].values[evam2]
P(f"  (target std for reference: {np.nanstd(ye):.4f})")

with open(OUT, 'w') as f:
    f.write('\n'.join(log))
P(f"\nsaved -> {OUT}")
