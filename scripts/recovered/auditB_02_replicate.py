"""
AUDIT B - 02: replicate v17b honest-CV numbers + capture diagnostics (B1/B6 groundwork).
Verifies: 0.6919 (no-bwd,no-dn), 0.6535 (bwd phi.80 lam.84), 0.6804 (dn+bwd),
k0 0.6287/0.6273/0.6254. Saves predictions for bootstrap.
"""
import numpy as np, pandas as pd, time, sys
sys.path.insert(0, '/home/z/my-project/scripts')
from auditB_lib import *
t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)

tr, te = load_cached()
train = build_train_df(tr)
n_cells = int(train['cc'].max())+1
log(f"train df: {len(train):,} rows, {n_cells} cells")

# test calendar mask fractions
test = pd.read_csv(f'{DATA}/Test (2).csv', usecols=['time','TWS_t_masked'])
test['time'] = pd.to_datetime(test['time'])
cal = test.assign(m=test['TWS_t_masked'].astype(bool)).groupby(test['time'].dt.month)['m'].mean()

# ---- val protocol (exact v17b) ----
train['year'] = train['t_abs']//12
val = train[(train['year'] >= 2013) & (train['year'] <= 2015)].copy()
val['masked'] = val['t_abs'].apply(lambda t: (t%12+1)).map(cal).values > 0.5
val_tws_visible = val['TWS_t'].copy()
val.loc[val['masked'], 'TWS_t'] = np.nan
val_target = val['target'].values.astype(np.float64)
val_msk = val['masked'].values
val_ta = val['t_abs'].values
mfrac_v = val.groupby('t_abs')['masked'].mean()
val_anchor_tas = set(int(v) for v in mfrac_v[mfrac_v < 0.01].index)
shortcut = np.array([(t+1) in val_anchor_tas for t in val_ta])
honest = val_msk & ~shortcut
log(f"val masked={val_msk.sum():,} honest={honest.sum():,}")

fit = train[train['year'] <= 2012].copy()
log("building infra_cv (fit<=2012)...")
infra_cv = build_infra(fit, n_cells, 'cv')
log(f"  infra_cv: full cells={infra_cv['full'].sum()}, V={infra_cv['V'].shape}, T_fit={infra_cv['T_fit']}")

def cv(pred):
    mm = honest & np.isfinite(pred) & np.isfinite(val_target)
    return float(np.sqrt(np.mean((pred[mm]-val_target[mm])**2))), mm

# ---- replicate key configs ----
log("\n=== masked-row replication ===")
res = {}
p_nb_nd = run_masked2(infra_cv, val, val_tws_visible, 0.80, use_bwd=False, use_denoise=False, lam=0.84)
res['no-bwd,no-dn'] = cv(p_nb_nd)[0]; log(f"  no-bwd no-dn phi.80: {res['no-bwd,no-dn']:.4f} (v17 log: 0.6919 @ phi.74)")
p_bwd = run_masked2(infra_cv, val, val_tws_visible, 0.80, use_bwd=True, use_denoise=False, lam=0.84, diag=True)
res['bwd-only'] = cv(p_bwd[0])[0]; log(f"  bwd-only phi.80 lam.84: {res['bwd-only']:.4f} (claimed 0.6535)")
DG = p_bwd[1]
log(f"  DIAG: var_f={DG['var_f']:.4f} (std={np.sqrt(DG['var_f']):.4f})  c_={DG['c_']:.4f}  varz={DG['varz']:.4f}")
log(f"  DIAG: H={DG['H']:.4f}  R={DG['R']:.4f}  P0={DG['P0']:.4f}  q={DG['q']:.6f}")
log(f"  DIAG: implied noise std from (1-lam)*var_f = {np.sqrt((1-0.84)*DG['var_f']):.4f} (team claims 0.456)")
log(f"  DIAG: H*lam*var_f = {DG['H']*0.84*DG['var_f']:.4f} (should equal c_ = {DG['c_']:.4f})")
p_dn = run_masked2(infra_cv, val, val_tws_visible, 0.80, use_bwd=True, use_denoise=True, lam=0.84)
res['dn+bwd'] = cv(p_dn)[0]; log(f"  dn+bwd phi.80: {res['dn+bwd']:.4f} (claimed 0.6804)")
p_dn_only = run_masked2(infra_cv, val, val_tws_visible, 0.80, use_bwd=False, use_denoise=True, lam=0.84)
res['dn-only'] = cv(p_dn_only)[0]; log(f"  dn-only: {res['dn-only']:.4f} (v17 log: 0.7023)")

# ---- B6: fwd vs bwd per honest month ----
log("\n=== per-month honest decomposition (phi.80/lam.84) ===")
mm = honest & np.isfinite(p_bwd[0]) & np.isfinite(val_target)
rows_h = DG['rows']; meta = DG['meta']
tgt_h = val_target[rows_h]
mu_h = infra_cv['mu_c'][val['cc'].values[rows_h]]
pred_full = p_bwd[0][rows_h]
xf_h = DG['xf']; xb_h = DG['xb']; wgt_h = DG['wgt']; dtil_h = DG['Dtil']
for j in range(len(meta)):
    r0 = rows_h[j]
    sl = slice(0,0)
    # rows for month j are contiguous positions in the concatenated arrays
    start = sum(len(rows_h[k]) for k in range(j))
    end = start + len(rows_h[j])
    m = meta['m'].iloc[j]; gb = meta['gap_back'].iloc[j]; gf = meta['gap_fwd'].iloc[j]
    rws = rows_h[j]
    t_ = val_target[rws]
    pf = mu_h[start:end] + dtil_h[start:end] + xf_h[start:end]
    pb_full = mu_h[start:end] + dtil_h[start:end] + xb_h[start:end]
    r_full = np.sqrt(np.nanmean((pred_full[start:end]-t_)**2))
    r_fwd  = np.sqrt(np.nanmean((pf-t_)**2))
    r_bwd  = np.sqrt(np.nanmean((pb_full-t_)**2)) if np.isfinite(xb_h[start:end]).all() else np.nan
    log(f"  m={m//12}-{m%12+1:02d} n={len(rws):,} gap_back={gb} gap_fwd={gf}: blend={r_full:.4f} fwd={r_fwd:.4f} bwd={r_bwd if r_bwd==r_bwd else float('nan'):.4f} meanwgt_fwd={np.mean(wgt_h[start:end]):.3f}")

# overall fwd-only (B6 headline)
r_fwd_all = float(np.sqrt(np.mean((pred_full - wgt_h*(pred_full-(mu_h+dtil_h+xb_h)) - val_target[rows_h])**2))) # not used
# simpler: recompute pred with fwd only
p_fwd = run_masked2(infra_cv, val, val_tws_visible, 0.80, use_bwd=False, use_denoise=False, lam=0.84)
res['fwd-only'] = cv(p_fwd)[0]; log(f"\n  FWD-ONLY honest CV: {res['fwd-only']:.4f}  (bwd gain = {res['fwd-only']-res['bwd-only']:.4f})")

# bwd gain for rows bucketed by gap_fwd (test-weighting later)
log("\n=== bwd gain by gap_fwd bucket (val honest rows) ===")
gf_arr = np.concatenate([np.full(len(rows_h[j]), meta['gap_fwd'].iloc[j]) for j in range(len(meta))])
gb_arr = np.concatenate([np.full(len(rows_h[j]), meta['gap_back'].iloc[j]) for j in range(len(meta))])
err_blend = (pred_full - val_target[rows_h])
pf_all = mu_h + dtil_h + xf_h
err_fwd = pf_all - val_target[rows_h]
pb_all = mu_h + dtil_h + xb_h
err_bwd = pb_all - val_target[rows_h]
for gf in sorted(set(gf_arr.tolist())):
    s = gf_arr == gf
    log(f"  gap_fwd={gf}: n={s.sum():,} blend={np.sqrt(np.mean(err_blend[s]**2)):.4f} fwd={np.sqrt(np.mean(err_fwd[s]**2)):.4f} bwd={np.sqrt(np.nanmean(np.where(np.isfinite(err_bwd[s]),err_bwd[s],np.nan)**2)) if np.isfinite(err_bwd[s]).any() else float('nan'):.4f}")

# ---- k0 replication ----
log("\n=== k0 replication ===")
tmax_cv = 2012*12 + 10
fit_k0 = train[train['t_abs'] <= tmax_cv]
Xtr, ytr, yr, tws_ok, mg_tr = build_k0_features(fit_k0, infra_cv)
ok_tr = np.isfinite(ytr) & tws_ok
Xtr, ytr, yr = Xtr[ok_tr], ytr[ok_tr], yr[ok_tr]
n_nan_next_tr = int((~np.isfinite(Xtr[:,6:11]).all(axis=1)).sum())
log(f"  k0 train rows: {len(Xtr):,} (with NaN next-covs: {n_nan_next_tr:,} = {n_nan_next_tr/len(Xtr)*100:.1f}%)")
wtr = np.where(yr <= 2009, 1.0, np.where(yr <= 2012, 2.0, 3.0)).astype(np.float32)
val_k0 = val.copy()
Xev, yev, _, tws_ok_ev, mg_ev = build_k0_features(val_k0, infra_cv)
ok_ev = (~val_msk) & tws_ok_ev & np.isfinite(yev)
Xev_o, yev_o = Xev[ok_ev], yev[ok_ev]
has_nxt_ev = np.isfinite(Xev_o[:,6:11]).all(axis=1)
log(f"  k0 eval rows: {len(Xev_o):,} (with next covs: {has_nxt_ev.sum():,} = {has_nxt_ev.mean()*100:.1f}%; TEST split is 66.5%/33.5%)")
p_lin, coefF, coefR = k0_linear(Xtr, ytr, wtr, Xev_o)
r_lin = float(np.sqrt(np.mean((p_lin-yev_o)**2)))
p_lgb, bst = k0_lgb(Xtr, ytr, wtr, Xev_o, rounds=400)
r_lgb = float(np.sqrt(np.mean((p_lgb-yev_o)**2)))
r_bd = float(np.sqrt(np.mean((0.5*p_lin+0.5*p_lgb-yev_o)**2)))
log(f"  LINEAR: {r_lin:.4f} (claimed 0.6287)  LGB: {r_lgb:.4f} (claimed 0.6273)  50/50: {r_bd:.4f} (claimed 0.6254)")
# split by next-cov availability
for tag, s in [('with next covs', has_nxt_ev), ('WITHOUT next covs', ~has_nxt_ev)]:
    if s.sum():
        log(f"    {tag}: n={s.sum():,} lin={np.sqrt(np.mean((p_lin[s]-yev_o[s])**2)):.4f} lgb={np.sqrt(np.mean((p_lgb[s]-yev_o[s])**2)):.4f}")
# persistence baseline
p_persist = Xev_o[:,0]  # TWS anomaly at t
r_persist = float(np.sqrt(np.mean((p_persist-yev_o)**2)))
log(f"  PERSISTENCE (predict TWS_t anomaly): {r_persist:.4f}")
# covs(t) only linear (no next) for reference
# coefficients
log(f"  coefF (full): {np.round(coefF,4).tolist()}")
log(f"  coefR (reduced): {np.round(coefR,4).tolist()}")

# ---- save for bootstrap / later ----
np.savez_compressed('/home/z/my-project/scripts/auditB_preds.npz',
    honest=honest, val_target=val_target, val_msk=val_msk,
    p_bwd=p_bwd[0], p_fwd=p_fwd, p_dn=p_dn, p_dn_only=p_dn_only, p_nb_nd=p_nb_nd,
    rows_h=rows_h, gf_arr=gf_arr, gb_arr=gb_arr,
    err_blend=err_blend, err_fwd=err_fwd, err_bwd=err_bwd,
    pred_full_h=pred_full, mu_h=mu_h, dtil_h=dtil_h, xf_h=xf_h, xb_h=xb_h, wgt_h=wgt_h,
    val_cc=val['cc'].values, val_ta=val_ta,
    yev_o=yev_o, p_lin=p_lin, p_lgb=p_lgb, has_nxt_ev=has_nxt_ev,
    Xev_o_0=Xev_o[:,0])
log("saved auditB_preds.npz")
log("DONE")
