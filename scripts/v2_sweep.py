"""
Post-V1 LB analysis + anchor-LOO parameter sweep + PC1 investigation.

LB feedback: v3a=0.7984 (predicted 0.79-0.80 OK), v1b=0.7152 (new best), v1c=0.7168.

S1: implied error decomposition from LB
S2: anchor-LOO sweep of (phi_f, lam_f, obs-mode) — direct masked-row predictor eval
S3: PC1 investigation (the cov-invisible 26.8%-variance component)
"""
import numpy as np, pandas as pd

DATA = '/home/z/my-project/data'
OUT = '/home/z/my-project/download/v2_sweep.txt'
COVS = ['SPEI_01_t','SPEI_03_t','SPEI_06_t','SPEI_12_t','SOIL_MOISTURE_t']
log = []
def P(s=''):
    print(s, flush=True)
    log.append(str(s))

# ================= S1: implied decomposition =================
P("=== S1: implied decomposition from LB ===")
for u in [0.44, 0.50, 0.55, 0.60, 0.64]:
    m2 = (0.7152**2 - 0.3345*u**2)/0.6655
    P(f"  if unmasked RMSE u={u:.2f}: implied masked RMSE = {np.sqrt(m2):.4f}")
P("  leader 0.5596:")
for u in [0.45, 0.50, 0.55]:
    m2 = (0.5596**2 - 0.3345*u**2)/0.6655
    P(f"  if u={u:.2f}: implied masked RMSE = {np.sqrt(m2):.4f}")

# ================= load =================
train = pd.read_csv(f'{DATA}/Train (1).csv', usecols=['time','lat','lon','TWS_t']+COVS)
train['time'] = pd.to_datetime(train['time'])
for c in ['TWS_t']+COVS:
    train[c] = pd.to_numeric(train[c], errors='coerce').astype('float32')
train['cc'] = (train['lat'].round(2).astype(str)+'_'+train['lon'].round(2).astype(str)).astype('category').cat.codes.astype('int32')
train['ym'] = train['time'].dt.year*100 + train['time'].dt.month
train['t_abs'] = (train['ym']//100)*12 + (train['ym']%100) - 1
n_cells = int(train['cc'].max())+1
yms = np.sort(train['ym'].unique())
T = len(yms)
ym_to_i = {int(v):i for i,v in enumerate(yms)}

F = np.full((T, n_cells), np.nan, dtype=np.float32)
F[train['ym'].map(ym_to_i).values, train['cc'].values] = train['TWS_t'].values
mu_c = np.nanmean(F, axis=0)

test = pd.read_csv(f'{DATA}/Test (2).csv')
test['time'] = pd.to_datetime(test['time'])
for c in ['TWS_t']+COVS:
    test[c] = pd.to_numeric(test[c], errors='coerce').astype('float32')
codes = train[['cc','lat','lon']].drop_duplicates('cc').sort_values('cc')
pos = {(int(round(float(la)*1000)), int(round(float(lo)*1000))): int(cc) for cc,la,lo in zip(codes['cc'],codes['lat'],codes['lon'])}
test['cc'] = [pos.get((int(round(float(a)*1000)), int(round(float(b)*1000))), -1) for a,b in zip(test['lat'], test['lon'])]
test['ym'] = test['time'].dt.year*100 + test['time'].dt.month
test['t_abs'] = (test['ym']//100)*12 + (test['ym']%100) - 1
test['masked'] = test['TWS_t_masked'].astype(bool)
ta = test['t_abs'].values
cc_t = test['cc'].values
msk = test['masked'].values

# global train cov regression
Z = train[COVS].values.astype('float32')
yv = train['TWS_t'].values.astype('float32')
Z1 = np.column_stack([Z, np.ones(len(Z))])
okz = np.isfinite(Z1).all(axis=1) & np.isfinite(yv)
coef_tr = np.linalg.solve(Z1[okz].T@Z1[okz] + 1e-2*np.eye(6), Z1[okz].T@yv[okz])

# cov fields (train-reg)
Zt_raw = test[COVS].values.astype('float32')
okt = np.isfinite(Zt_raw).all(axis=1)
cov_est_tr = np.full(len(test), np.nan, dtype=np.float32)
cov_est_tr[okt] = np.column_stack([Zt_raw[okt], np.ones(okt.sum())]) @ coef_tr
cov_field_tr = {}
for m in np.sort(test['t_abs'].unique()):
    selm = (ta == m) & okt
    fm = np.full(n_cells, np.nan, dtype=np.float32)
    fm[cc_t[selm]] = cov_est_tr[selm]
    cov_field_tr[m] = fm - mu_c
all_m = np.array(sorted(cov_field_tr.keys()))
S_tr = np.nanmean(np.array([cov_field_tr[m] for m in all_m]), axis=0)
W_tr = {int(m): cov_field_tr[m] - S_tr for m in all_m}

# anchors
mfrac = test.groupby('t_abs')['masked'].mean()
anchors = sorted(int(v) for v in mfrac[mfrac < 0.01].index)
AF = {}
for a in anchors:
    sel = (ta == a) & (~msk)
    fa = np.full(n_cells, np.nan, dtype=np.float32)
    fa[cc_t[sel]] = test['TWS_t'].values[sel]
    AF[a] = fa - mu_c

# anchor-refit cov regression (per LOO target j)
def refit_coef(exclude):
    Xs, ys = [], []
    for a in anchors:
        if a == exclude: continue
        sel = (ta == a) & okt
        Xs.append(np.column_stack([Zt_raw[sel], np.ones(sel.sum())]))
        ys.append(AF[a][cc_t[sel]])
    X = np.vstack(Xs); y = np.concatenate(ys)
    okr = np.isfinite(X).all(axis=1) & np.isfinite(y)
    return np.linalg.solve(X[okr].T@X[okr] + 1e-2*np.eye(6), X[okr].T@y[okr])

# LOO D-tilde (global-combo form, excluding target anchor j)
wD, wS = 0.676, 0.490  # from V1 build (stable across LOO in R2f)
def Dtil_loo(j):
    dhat = np.nanmean(np.array([AF[a] for a in anchors if a != j]), axis=0)
    return wD*dhat + wS*S_tr

# anchor-refit cov fields per LOO j
cov_field_rf = {}
for j in anchors:
    c_ = refit_coef(j)
    est = np.full(len(test), np.nan, dtype=np.float32)
    est[okt] = np.column_stack([Zt_raw[okt], np.ones(okt.sum())]) @ c_
    fld = {}
    for m in all_m:
        selm = (ta == m) & okt
        fm = np.full(n_cells, np.nan, dtype=np.float32)
        fm[cc_t[selm]] = est[selm]
        fld[m] = fm - mu_c
    S_rf = np.nanmean(np.array([fld[m] for m in all_m]), axis=0)
    cov_field_rf[j] = {int(m): fld[m] - S_rf for m in all_m}

# ================= S2: anchor-LOO sweep =================
P("\n=== S2: anchor-LOO sweep (consecutive anchor pairs, gap<=8) ===")
pairs = []
for i_idx in range(len(anchors)-1):
    i, j = anchors[i_idx], anchors[i_idx+1]
    g = j - i
    if g <= 8:
        pairs.append((i, j, g))
P(f"pairs (anchor_i, target_j, horizon h): {[(p[0]%10000, p[1]%10000, p[2]) for p in pairs]}")
P("(h = t_j - t_i: the simulated row sits at t_j - 1, propagates h months from anchor i)")

RH = {1:0.874, 2:0.781, 3:0.712, 4:0.660, 5:0.623, 6:0.595, 7:0.574, 8:0.557}

def kalman_pred_loo(i, j, g, phi_f, lam_f, obs_mode, use_obs_tm=True):
    Dl = Dtil_loo(j)
    x = np.where(np.isfinite(AF[i]-Dl), lam_f*(AF[i]-Dl), 0.0).astype(np.float32)
    var_f = float(np.nanmean([np.nanvar(AF[a]-Dtil_loo(a)) for a in anchors if a != j]))
    P0 = lam_f*(1-lam_f)*var_f
    P = np.where(np.isfinite(AF[i]), P0, var_f).astype(np.float32)
    q = lam_f*var_f*(1-phi_f**2)
    # observation calibration on anchors != j
    if obs_mode == 'none':
        H = 0.0; R = 1e18
    else:
        src = W_tr if obs_mode=='train' else cov_field_rf[j]
        cs, zs = [], []
        for a in anchors:
            if a == j: continue
            wv = src[a]; dlv = Dtil_loo(a)
            ok = np.isfinite(wv) & np.isfinite(AF[a]) & np.isfinite(dlv)
            cs.append(np.cov(wv[ok], (AF[a]-dlv)[ok])[0,1]); zs.append(np.var(wv[ok]))
        c_ = float(np.mean(cs)); varz = float(np.mean(zs))
        var_x = lam_f*var_f
        H = c_/var_x; R = max(varz - c_*c_/var_x, 1e-4)
    for k in range(1, g):
        m = i + k
        x = phi_f*x; P = phi_f**2*P + q
        if obs_mode != 'none' and m in (W_tr if obs_mode=='train' else cov_field_rf[j]):
            wv = (W_tr if obs_mode=='train' else cov_field_rf[j])[m]
            okw = np.isfinite(wv)
            K = np.where(okw, P*H/(H*H*P+R), 0).astype(np.float32)
            x = np.where(okw, x + K*(wv - H*x), x)
            P = np.where(okw, (1-K*H)*P, P)
    # final step to target month j
    x = phi_f*x; P = phi_f**2*P + q
    if use_obs_tm and obs_mode != 'none' and j in (W_tr if obs_mode=='train' else cov_field_rf[j]):
        wv = (W_tr if obs_mode=='train' else cov_field_rf[j])[j]
        okw = np.isfinite(wv)
        K = np.where(okw, P*H/(H*H*P+R), 0).astype(np.float32)
        x = np.where(okw, x + K*(wv - H*x), x)
    return Dl + x

def decay_pred_loo(i, j, g):
    return RH.get(g, 0.55)*(AF[i])

results = []
for phi_f in [0.70, 0.75, 0.80, 0.85, 0.90]:
    for lam_f in [0.80, 0.84, 0.88, 0.92]:
        for obs_mode in ['none', 'train', 'refit']:
            errs, base_errs, dec_errs = [], [], []
            for (i, j, g) in pairs:
                pred = kalman_pred_loo(i, j, g, phi_f, lam_f, obs_mode)
                dec = decay_pred_loo(i, j, g)
                tgt = AF[j]
                ok = np.isfinite(pred) & np.isfinite(tgt)
                errs.append(float(np.sqrt(np.nanmean((pred[ok]-tgt[ok])**2))))
                dec_errs.append(float(np.sqrt(np.nanmean((dec[ok]-tgt[ok])**2))))
            results.append((np.mean(errs), phi_f, lam_f, obs_mode, np.mean(dec_errs), errs))
results.sort(key=lambda r: r[0])
P("\ntop 15 configs (mean RMSE over pairs, lower=better):")
for r in results[:15]:
    P(f"  RMSE={r[0]:.4f}  phi_f={r[1]:.2f} lam_f={r[2]:.2f} obs={r[3]:5s}  [pure-decay baseline: {r[4]:.4f}]  per-pair: {np.round(r[5],3)}")
best = results[0]
P(f"\nBEST: phi_f={best[1]}, lam_f={best[2]}, obs={best[3]} -> RMSE {best[0]:.4f} vs decay {best[4]:.4f} (gain {best[4]-best[0]:.4f})")
# per obs-mode best
P("\nbest per obs-mode:")
for mode in ['none','train','refit']:
    b = min([r for r in results if r[3]==mode])
    P(f"  {mode:5s}: RMSE={b[0]:.4f} (phi={b[1]}, lam={b[2]})")

# ================= S3: PC1 investigation =================
P("\n=== S3: PC1 of train anomaly field ===")
full = ~np.isnan(F).any(axis=0)
Af = F[:, full] - np.nanmean(F[:, full], axis=0, keepdims=True)
U, S_, Vt = np.linalg.svd(Af, full_matrices=False)
v_exp = S_**2/(S_**2).sum()
P(f"variance: PC1={v_exp[0]*100:.1f}%, PC2={v_exp[1]*100:.1f}%, PC3={v_exp[2]*100:.1f}%")
v1 = Vt[0]  # loading over full cells
mu_full = mu_c[full]
P(f"corr(PC1 loading, mu_c) = {np.corrcoef(v1, mu_full)[0,1]:.4f}")
# latitude structure of PC1 loading
la_full = codes['lat'].values[full]
for lo, hi in [(-90,-60),(-60,-30),(-30,0),(0,30),(30,60),(60,90)]:
    selm = (la_full>=lo)&(la_full<hi)
    if selm.sum()>50:
        P(f"  lat [{lo},{hi}): mean loading {v1[selm].mean():+.4f} (n={selm.sum()})")
s1 = U[:,0]*S_[0]
P(f"\nPC1 score (train, {T} months): std={s1.std():.3f}")
acf = [float(np.corrcoef(s1[:-k], s1[k:])[0,1]) for k in range(1,25)]
P("PC1 score ACF 1..24: " + ", ".join(f"{a:.2f}" for a in acf))
t_ax = np.arange(T)
P(f"corr(s1, time) = {np.corrcoef(s1, t_ax)[0,1]:.4f} (trend check)")
mo = yms % 100
P(f"PC1 score by calendar month (mean): " + ", ".join(f"{s1[mo==m].mean():+.2f}" for m in range(1,13)))
# global cov means vs PC1 score
gm = {}
for c in COVS:
    v = train[c].values.astype('float32')
    okv = np.isfinite(v) & np.isfinite(train['TWS_t'].values)
    # monthly mean of cov
    s = pd.Series(v[okv]).groupby(train['ym'].values[okv]).mean()
    gm[c] = s.reindex(yms).values
for c in COVS:
    okg = np.isfinite(gm[c])
    P(f"corr(PC1 score, monthly global mean {c}) = {np.corrcoef(s1[okg], gm[c][okg])[0,1]:+.4f}")
# test-era projections
P("\ntest-era PC1 projections (anchor fields onto train PC1 loading):")
for a in anchors:
    fa = AF[a][full]
    okf = np.isfinite(fa)
    proj = float((fa[okf]-fa[okf].mean()) @ v1[okf] / np.sum(v1[okf]**2))
    P(f"  {a%10000}: PC1 coord = {proj:+.3f}  (train z-score {proj/s1.std():+.2f})")
dh = np.nanmean(np.array([AF[a] for a in anchors]), axis=0)[full]
okd = np.isfinite(dh)
proj_d = float((dh[okd]-dh[okd].mean()) @ v1[okd] / np.sum(v1[okd]**2))
P(f"  D-hat: PC1 coord = {proj_d:+.3f} (train z {proj_d/s1.std():+.2f})")
# anchor-own PC1 vs train PC1
AFm = np.array([AF[a][full] for a in anchors])
AFc = AFm - np.nanmean(AFm, axis=1, keepdims=True)
ok_cols = ~np.isnan(AFc).any(axis=0)
Ua, Sa, Vta = np.linalg.svd(AFc[:, ok_cols], full_matrices=False)
P(f"anchor-field PC1 vs train-PC1 loading corr = {np.corrcoef(Vta[0], v1[ok_cols])[0,1]:+.4f}")
P(f"anchor-field PC1 vs -mu_c corr = {np.corrcoef(Vta[0], -mu_full[ok_cols])[0,1]:+.4f}")

with open(OUT, 'w') as f:
    f.write('\n'.join(log))
P(f"\nsaved -> {OUT}")
