"""
AUDIT 15-c part 4: PER-CELL HETEROGENEITY (item 5)
Per-cell corr between cov-regression output deviation (Wd) and TWS detrended anomaly (A),
exactly the quantities used for per-cell H/R in submission_v8.py. Then: for low-corr cells,
compare v8a masked-row prediction std / fast-component std vs typical cells.
"""
import numpy as np, pandas as pd

DATA = '/home/z/my-project/data'; DL = '/home/z/my-project/download'
COVS = ['SPEI_01_t','SPEI_03_t','SPEI_06_t','SPEI_12_t','SOIL_MOISTURE_t']
LAM_F = 0.84

tr = pd.read_csv(f'{DATA}/Train (1).csv', usecols=['time','lat','lon','TWS_t']+COVS)
tr['time'] = pd.to_datetime(tr['time'])
for c in ['TWS_t']+COVS: tr[c] = pd.to_numeric(tr[c], errors='coerce').astype('float32')
tr['cc'] = (tr['lat'].round(2).astype(str)+'_'+tr['lon'].round(2).astype(str)).astype('category').cat.codes.astype('int32')
ym = tr['time'].dt.year*100 + tr['time'].dt.month
tr['t_abs'] = (ym//100)*12 + (ym%100) - 1
n_cells = int(tr['cc'].max())+1

# field, mu_c, beta_c (as in v8)
yms = np.sort(np.unique(ym)); T = len(yms)
ym_to_i = {int(v):i for i,v in enumerate(yms)}
t_abs_tr = np.array([(int(v)//100)*12+(int(v)%100)-1 for v in yms], dtype=np.float64)
F = np.full((T, n_cells), np.nan, dtype=np.float32)
F[tr['ym'].map(ym_to_i).values if 'ym' in tr else [ym_to_i[int(v)] for v in ym], tr['cc'].values] = tr['TWS_t'].values
mu_c = np.nanmean(F, axis=0)
F64 = F.astype(np.float64); ok_t = np.isfinite(F64)
t_mat = np.where(ok_t, t_abs_tr[:,None], np.nan)
tbar_c = np.nanmean(t_mat, axis=0); td = t_mat - tbar_c[None,:]
sxx = np.nansum(td*td, axis=0)
beta_c = np.where(sxx>100, np.nansum(td*F64,axis=0)/np.where(sxx>0,sxx,1), 0.0).astype(np.float32)
def trendex(t): return ((np.float64(t)-tbar_c)*beta_c).astype(np.float32)

# global cov regression (as in v8)
Z = tr[COVS].values.astype('float32'); yv = tr['TWS_t'].values.astype('float32')
Z1 = np.column_stack([Z, np.ones(len(Z))]); okz = np.isfinite(Z1).all(axis=1)&np.isfinite(yv)
coef = np.linalg.solve(Z1[okz].T@Z1[okz] + 1e-2*np.eye(6), Z1[okz].T@yv[okz])

# per-cell corr: Wd = cov_est - per-cell mean; A = TWS - mu - trendex
cov_est = np.full(len(tr), np.nan, dtype=np.float32)
Zok = np.isfinite(Z).all(axis=1)
cov_est[Zok] = np.column_stack([Z[Zok], np.ones(Zok.sum())]) @ coef
ta_r = tr['t_abs'].values; cc_r = tr['cc'].values
txv = ((ta_r.astype(np.float64)-tbar_c[cc_r])*beta_c[cc_r])
A = yv - mu_c[cc_r] - txv
gmean = pd.Series(cov_est).groupby(cc_r).mean().reindex(range(n_cells)).fillna(0).values
Wd = cov_est - gmean[cc_r]
ok = np.isfinite(Wd)&np.isfinite(A)
ci = cc_r[ok]; Wv = Wd[ok].astype(np.float64); Av = A[ok].astype(np.float64)
n_c = np.bincount(ci, minlength=n_cells)
sW = np.bincount(ci, weights=Wv, minlength=n_cells); sA = np.bincount(ci, weights=Av, minlength=n_cells)
sWW = np.bincount(ci, weights=Wv*Wv, minlength=n_cells); sAA = np.bincount(ci, weights=Av*Av, minlength=n_cells)
sWA = np.bincount(ci, weights=Wv*Av, minlength=n_cells)
with np.errstate(invalid='ignore', divide='ignore'):
    varW = sWW/n_c-(sW/n_c)**2; varA = sAA/n_c-(sA/n_c)**2; covWA = sWA/n_c-(sW/n_c)*(sA/n_c)
    r_cell = covWA/np.sqrt(varW*varA)
    r_cell_raw = covWA/np.sqrt(varW*varA)  # same
valid = (n_c>=100)&(varW>1e-6)&(varA>1e-6)
r_cell = np.where(valid, r_cell, np.nan)
print("per-cell corr(Wd, A_detrended):")
pct = np.nanpercentile(r_cell, [10,25,50,75,90])
print(f"  10/25/50/75/90 pct: {pct[0]:.4f}/{pct[1]:.4f}/{pct[2]:.4f}/{pct[3]:.4f}/{pct[4]:.4f}")
print(f"  cells with |corr|<0.3: {int((np.abs(r_cell)<0.3).sum()):,} ({np.nanmean(np.abs(r_cell)<0.3)*100:.1f}%)")
print(f"  cells with  corr<0.3 (signed): {int((r_cell<0.3).sum()):,}")
print(f"  cells with corr>0.6: {int((r_cell>0.6).sum()):,} ({np.nanmean(r_cell>0.6)*100:.1f}%)")
print(f"  negative-corr cells: {int((r_cell<0).sum()):,}")

# alt definition (no detrend) for reference
A2 = yv - mu_c[cc_r]
ok2 = np.isfinite(Wd)&np.isfinite(A2)
ci2 = cc_r[ok2]; W2 = Wd[ok2].astype(np.float64); A2v = A2[ok2].astype(np.float64)
n2 = np.bincount(ci2, minlength=n_cells)
sW2 = np.bincount(ci2, weights=W2, minlength=n_cells); sA2 = np.bincount(ci2, weights=A2v, minlength=n_cells)
sWW2 = np.bincount(ci2, weights=W2*W2, minlength=n_cells); sAA2 = np.bincount(ci2, weights=A2v*A2v, minlength=n_cells)
sWA2 = np.bincount(ci2, weights=W2*A2v, minlength=n_cells)
with np.errstate(invalid='ignore', divide='ignore'):
    r2 = (sWA2/n2-(sW2/n2)*(sA2/n2))/np.sqrt((sWW2/n2-(sW2/n2)**2)*(sAA2/n2-(sA2/n2)**2))
r2 = np.where((n2>=100), r2, np.nan)
pct2 = np.nanpercentile(r2, [10,25,50,75,90])
print(f"\nalt (no detrend) 10/25/50/75/90: {pct2[0]:.4f}/{pct2[1]:.4f}/{pct2[2]:.4f}/{pct2[3]:.4f}/{pct2[4]:.4f}")
print(f"alt cells with |corr|<0.3: {int((np.abs(r2)<0.3).sum()):,} ({np.nanmean(np.abs(r2)<0.3)*100:.1f}%)")

# ================= prediction overconfidence by cov quality =================
print("\n=== v8a masked predictions by cell cov quality ===")
te = pd.read_csv(f'{DATA}/Test (2).csv', usecols=['time','lat','lon','TWS_t','TWS_t_masked']+COVS)
te['time'] = pd.to_datetime(te['time'])
ymt = te['time'].dt.year*100 + te['time'].dt.month
te['t_abs'] = (ymt//100)*12 + (ymt%100) - 1
msk = te['TWS_t_masked'].astype(bool).values
codes = tr[['cc','lat','lon']].drop_duplicates('cc').sort_values('cc')
pos = {(int(round(float(la)*1000)), int(round(float(lo)*1000))): int(cc) for cc,la,lo in zip(codes['cc'],codes['lat'],codes['lon'])}
cc_t = np.array([pos.get((int(round(float(a)*1000)), int(round(float(b)*1000))), -1) for a,b in zip(te['lat'],te['lon'])])

v8a = pd.read_csv(f'{DL}/submission_v8a.csv')['Target'].values.astype(np.float64)
v2b = pd.read_csv(f'{DL}/submission_v2b.csv')['Target'].values.astype(np.float64)

# rebuild Dtil exactly as v8 (cheap: no SVD needed for Dtil itself)
mfrac = pd.Series(msk).groupby(te['t_abs'].values).mean()
anchors = sorted(int(v) for v in mfrac[mfrac<0.01].index)
AF = {}
ta_t = te['t_abs'].values
for a_ in anchors:
    sel = (ta_t==a_)&(~msk)
    fa = np.full(n_cells, np.nan, dtype=np.float32)
    fa[cc_t[sel]] = te['TWS_t'].values[sel].astype(np.float32)
    AF[a_] = fa - mu_c
Zt = te[COVS].values.astype('float32'); okt = np.isfinite(Zt).all(axis=1)
cov_est_t = np.full(len(te), np.nan, dtype=np.float32)
cov_est_t[okt] = np.column_stack([Zt[okt], np.ones(okt.sum())]) @ coef
cov_field = {}
for m in np.sort(np.unique(ta_t)):
    selm = (ta_t==m)&okt
    fm = np.full(n_cells, np.nan, dtype=np.float32)
    fm[cc_t[selm]] = cov_est_t[selm]
    cov_field[m] = fm - mu_c
all_m = np.array(sorted(cov_field.keys()))
S = np.nanmean(np.array([cov_field[m] for m in all_m]), axis=0)
Xs, ys = [], []
for a_ in anchors:
    dloo = np.nanmean(np.array([AF[b] for b in anchors if b!=a_]), axis=0)
    tx = trendex(a_)
    ok = np.isfinite(dloo)&np.isfinite(S)&np.isfinite(AF[a_])&np.isfinite(tx)
    Xs.append(np.column_stack([dloo[ok],S[ok],tx[ok]])); ys.append(AF[a_][ok])
w_ = np.linalg.solve(np.vstack(Xs).T@np.vstack(Xs) + 1e-3*np.eye(3), np.vstack(Xs).T@np.concatenate(ys))
W1,W2_,W3 = map(float, w_)
print(f"D-tilde weights (recomputed): {W1:.3f}/{W2_:.3f}/{W3:.3f}")
Dhat = np.nanmean(np.array([AF[a_] for a_ in anchors]), axis=0)
print(f"cells with NaN Dhat (zero anchor obs): {int(np.isnan(Dhat).sum())}")
def Dtil(t): return (W1*Dhat + W2_*S + W3*trendex(t)).astype(np.float32)

tm = ta_t + 1  # target month
Dt_tm = Dtil(10**9)  # placeholder
# vectorized Dtil per row target-month
Drow = np.full(len(te), np.nan, dtype=np.float64)
for m in np.unique(tm):
    d = Dtil(int(m))
    sel = tm==m
    Drow[sel] = d[cc_t[sel]]

x_fast = v8a - mu_c[cc_t] - Drow     # v8a fast component actually predicted
x_fast_v2b = v2b - mu_c[cc_t] - Drow
mr = np.where(msk)[0]
q_low  = np.abs(r_cell[cc_t[mr]]) < 0.3
q_mid  = (np.abs(r_cell[cc_t[mr]]) >= 0.3) & (np.abs(r_cell[cc_t[mr]]) < 0.6)
q_high = np.abs(r_cell[cc_t[mr]]) >= 0.6
print(f"\nmasked rows by cell cov-quality bucket: low={q_low.sum():,} mid={q_mid.sum():,} high={q_high.sum():,}")
for lbl, q in [('|corr|<0.3', q_low), ('0.3-0.6', q_mid), ('>=0.6', q_high)]:
    print(f"  {lbl:10s}: pred_std={v8a[mr[q]].std():.4f}  |x_fast|_mean={np.abs(x_fast[mr[q]]).mean():.4f}  "
          f"x_fast_std={x_fast[mr[q]].std():.4f}  (v2b |x_fast|={np.abs(x_fast_v2b[mr[q]]).mean():.4f})")

# does v8a inflate |x| vs v2b in low-corr cells specifically?
for lbl, q in [('low', q_low), ('mid', q_mid), ('high', q_high)]:
    d = np.abs(x_fast[mr[q]]).mean() - np.abs(x_fast_v2b[mr[q]]).mean()
    print(f"  v8a-vs-v2b |x_fast| change, {lbl:5s}: {d:+.4f}")

# Hc clipping effect: how many low-corr cells get Hc floored by clip(0.3,3.0)?
Hc_raw = covWA/(LAM_F*varA)
Hg = float(np.nanmean(np.where(valid, Hc_raw, np.nan)))
rH = np.clip(Hc_raw/Hg, 0.3, 3.0)
print(f"\nglobal Hg={Hg:.4f}; cells where rH clipped at 0.3 floor: {int((valid&(Hc_raw/Hg<0.3)).sum()):,} "
      f"({100*np.mean(valid&(Hc_raw/Hg<0.3)):.1f}% of valid)")
print(f"cells with raw H ratio < 0.3 (would want near-zero gain): {int((valid&(Hc_raw/Hg<0.3)).sum()):,}")
print(f"Hc after 50/50 mix for those cells: H*{0.5+0.5*0.3:.2f} = {Hg*(0.5+0.5*0.3):.4f} vs global {Hg:.4f}")

print("\nDONE part 4")
