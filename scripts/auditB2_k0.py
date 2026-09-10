"""
AUDIT B2 (re-scoped) — k0 MODEL HEADROOM (item B2).
Baseline replication (v17b: linear 0.6287 / LGB 0.6273 / blend 0.6254) + <=6 legit feature
ideas, each reported as honest-CV delta for LINEAR and BLEND.
Protocol: fit <=2012 (labels strictly <= Dec-2012 via t_abs<=Nov-2012 cap), eval on 2013-15
unmasked (k0-like) rows; features never touch TWS(t+1) or masked TWS. Secondary eval subset:
rows at the 6 M1 mirror-anchor months (test-like k0 geometry), with M1-honest history
(trailing means see only fit-era + M1-anchor months).
Uses cached npz (no CSV loads). New file only.
"""
import numpy as np, pandas as pd, time, sys, warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
sys.path.insert(0, '/home/z/my-project/scripts')
from auditB_lib import load_cached, build_train_df, build_infra, build_k0_features, k0_linear, k0_lgb, COVS

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)
OUT = '/home/z/my-project/download/auditB2_k0.txt' if len(sys.argv) <= 1 else f'/home/z/my-project/download/auditB2_k0_{sys.argv[1]}.txt'
lines = []
def P(s=''):
    print(s, flush=True); lines.append(str(s))

PHASE = sys.argv[1].upper() if len(sys.argv) > 1 else 'A'
tr, te = load_cached()
cells = np.load('/home/z/my-project/scripts/auditB_cells.npz')
lat = cells['lat'].astype(np.float64); lon = cells['lon'].astype(np.float64)
n_cells = len(lat)
train = build_train_df(tr)
train['year'] = train['t_abs'] // 12

# calendar-month mask fractions from TEST (drives val protocol; identical to v17b)
tmonth = te['t_abs'] % 12 + 1
cal = pd.Series(te['masked'].astype(float)).groupby(tmonth).mean()

# ---------- protocol frames (v17b exact) ----------
val = train[(train['year'] >= 2013) & (train['year'] <= 2015)].copy()
val_msk = (val['t_abs'] % 12 + 1).map(cal).values > 0.5
val['masked'] = val_msk
val.loc[val['masked'], 'TWS_t'] = np.nan
log(f"val rows={len(val):,} masked={val_msk.sum():,} unmasked={(~val_msk).sum():,}")

fit = train[train['year'] <= 2012].copy()
log("building infra (fit<=2012)...")
infra = build_infra(fit, n_cells, 'cv')
mu_c = infra['mu_c']; clim = infra['clim']; tbar_c = infra['tbar_c']; beta_c = infra['beta_c']

# ---------- baseline k0 replication ----------
tmax_cv = 2012*12 + 10
fit_k0 = train[train['t_abs'] <= tmax_cv].copy()
Xtr, ytr, yr, tws_ok, mg_tr = build_k0_features(fit_k0, infra)
ok_tr = np.isfinite(ytr) & tws_ok
Xtr, ytr, yr = Xtr[ok_tr], ytr[ok_tr], yr[ok_tr]
mg_tr = mg_tr.loc[ok_tr].reset_index(drop=True)
wtr = np.where(yr <= 2009, 1.0, np.where(yr <= 2012, 2.0, 3.0)).astype(np.float32)
val_k0 = val.copy()
Xev, yev, _, tws_ok_ev, mg_ev = build_k0_features(val_k0, infra)
ok_ev = (~val_msk) & tws_ok_ev & np.isfinite(yev)
Xev_o, yev_o = Xev[ok_ev], yev[ok_ev]
mg_evo = mg_ev.loc[ok_ev].reset_index(drop=True)
log(f"k0 train rows={len(Xtr):,} (v17b log: 1,779,287)  eval rows={len(Xev_o):,} (v17b log: 218,534)")

p_lin0, _, _ = k0_linear(Xtr, ytr, wtr, Xev_o)
r_lin0 = float(np.sqrt(np.mean((p_lin0 - yev_o)**2)))
p_lgb0, _ = k0_lgb(Xtr, ytr, wtr, Xev_o, rounds=400)
r_lgb0 = float(np.sqrt(np.mean((p_lgb0 - yev_o)**2)))
p_bd0 = 0.5*p_lin0 + 0.5*p_lgb0
r_bd0 = float(np.sqrt(np.mean((p_bd0 - yev_o)**2)))
P(f"BASELINE replication: LINEAR {r_lin0:.4f} (v17b 0.6287) | LGBM(400) {r_lgb0:.4f} (0.6273) | blend {r_bd0:.4f} (0.6254)")

# M1-anchor eval subset (test-like k0 geometry)
M1_ANCHORS = [24156, 24160, 24166, 24172, 24183, 24186]
m1_sel = np.isin(mg_evo['t_abs'].values, M1_ANCHORS)
P(f"M1-anchor eval subset rows: {int(m1_sel.sum()):,} of {len(m1_sel):,}")
P(f"BASELINE on M1-anchor subset: lin {float(np.sqrt(np.mean((p_lin0[m1_sel]-yev_o[m1_sel])**2))):.4f} "
  f"lgb {float(np.sqrt(np.mean((p_lgb0[m1_sel]-yev_o[m1_sel])**2))):.4f} "
  f"blend {float(np.sqrt(np.mean((p_bd0[m1_sel]-yev_o[m1_sel])**2))):.4f}")

# ---------- features: build (phase A) or load cache (phase B) ----------
if PHASE == 'A':
    # ---------- feature engineering ----------
    log("building extra features...")
    # (G1) spatial neighbor mean TWS anomaly at t (ring 1deg w=.99, ring 2deg w=.97)
    ilat = np.round(lat*2).astype(np.int64)
    ilon = np.round((lon+180)*2).astype(np.int64) % 720
    buckets = {}
    for i in range(n_cells): buckets.setdefault((int(ilat[i]), int(ilon[i])), []).append(i)
    src1, dst1, src2, dst2 = [], [], [], []
    for i in range(n_cells):
        a, b = int(ilat[i]), int(ilon[i])
        for da in (-4, -2, 0, 2, 4):
            for db in (-4, -3, -2, -1, 0, 1, 2, 3, 4):
                if da == 0 and db == 0: continue
                dlon = min(abs(db), 720 - abs(db))
                cheb = max(abs(da), dlon)
                if cheb not in (2, 4): continue
                jj = buckets.get((a+da, (b+db) % 720))
                if not jj: continue
                for j in jj:
                    if cheb == 2: src1.append(j); dst1.append(i)
                    else: src2.append(j); dst2.append(i)
    src1 = np.array(src1); dst1 = np.array(dst1); src2 = np.array(src2); dst2 = np.array(dst2)
    P(f"neighbor pairs: ring1={len(src1):,} ring2={len(src2):,} (cells={n_cells:,})")

    def nbr_mean(field):
        v = np.where(np.isfinite(field), field, 0.0)
        f = np.isfinite(field).astype(np.float64)
        s1 = np.bincount(dst1, weights=v[src1], minlength=n_cells); c1 = np.bincount(dst1, weights=f[src1], minlength=n_cells)
        s2 = np.bincount(dst2, weights=v[src2], minlength=n_cells); c2 = np.bincount(dst2, weights=f[src2], minlength=n_cells)
        num = 0.99*s1 + 0.97*s2; den = 0.99*c1 + 0.97*c2
        return np.where(den > 0, num/np.maximum(den, 1e-9), 0.0).astype(np.float32)

    fit_all = train[train['year'] <= 2012]           # full history pool (all observed <=2012)
    def field_of(df):
        f = np.full(n_cells, np.nan, dtype=np.float64)
        f[df['cc'].values] = df['TWS_t'].values - mu_c[df['cc'].values]
        return f
    nbr_by_month = {}
    for m in np.sort(fit_k0['t_abs'].unique()):
        nbr_by_month[int(m)] = nbr_mean(field_of(fit_all[fit_all['t_abs'] == m]))
    vu = val[~val['masked']]
    for m in np.sort(vu['t_abs'].unique()):
        nbr_by_month[int(m)] = nbr_mean(field_of(vu[vu['t_abs'] == m]))

    # (G2) trailing observed anomaly means (3/6)
    def trailing(df_pool, df_rows, k):
        """Mean of the last k OBSERVED anomaly values at months <= t, INCLUDING t itself."""
        pool = pd.DataFrame({'cc': df_pool['cc'].values, 't_abs': df_pool['t_abs'].values,
                             'a': df_pool['TWS_t'].values - mu_c[df_pool['cc'].values]})
        pool = pool.sort_values(['cc', 't_abs'], kind='mergesort').reset_index(drop=True)
        roll = pool.groupby('cc', sort=False)['a'].rolling(k, min_periods=1).mean()
        pool['roll'] = roll.values
        rows = pd.DataFrame({'cc': df_rows['cc'].values, 't_abs': df_rows['t_abs'].values})
        mg = rows.merge(pool[['cc', 't_abs', 'roll']], on=['cc', 't_abs'], how='left')
        return np.nan_to_num(mg['roll'].values, nan=0.0).astype(np.float32)

    pool_ev = pd.concat([fit_all[['cc','t_abs','TWS_t']], vu[['cc','t_abs','TWS_t']]], ignore_index=True)
    tr3_tr = trailing(fit_all, mg_tr, 3); tr6_tr = trailing(fit_all, mg_tr, 6)
    tr3_ev = trailing(pool_ev, mg_evo, 3); tr6_ev = trailing(pool_ev, mg_evo, 6)
    vu_m1 = vu[np.isin(vu['t_abs'].values, M1_ANCHORS)]
    pool_m1 = pd.concat([fit_all[['cc','t_abs','TWS_t']], vu_m1[['cc','t_abs','TWS_t']]], ignore_index=True)
    tr3_m1 = trailing(pool_m1, mg_evo, 3); tr6_m1 = trailing(pool_m1, mg_evo, 6)

    # (G3) covariate interactions (anomaly products at t)
    Xtr0 = np.nan_to_num(Xtr, nan=0.0); Xev0 = np.nan_to_num(Xev_o, nan=0.0)
    i_tr = np.column_stack([Xtr0[:,1]*Xtr0[:,4], Xtr0[:,5]*Xtr0[:,2]]).astype(np.float32)
    i_ev = np.column_stack([Xev0[:,1]*Xev0[:,4], Xev0[:,5]*Xev0[:,2]]).astype(np.float32)

    # (G4) month-of-year of TARGET month (sin/cos)
    def mdummies(ta):
        m1 = (ta + 1) % 12 + 1
        ang = 2*np.pi*m1/12.0
        return np.column_stack([np.sin(ang), np.cos(ang)]).astype(np.float32)
    md_tr = mdummies(mg_tr['t_abs'].values); md_ev = mdummies(mg_evo['t_abs'].values)

    # (G5) cell trend extrapolation at t+1
    def trendex(mg):
        cc = mg['cc'].values
        return ((mg['t_abs'].values + 1 - tbar_c[cc]) * beta_c[cc]).astype(np.float32)
    tx_tr = trendex(mg_tr); tx_ev = trendex(mg_evo)

    # (G6) seasonal structure (fit-era monthly climatology, >=3 obs else 0)
    s_sum = np.zeros((n_cells, 12), dtype=np.float64); s_cnt = np.zeros((n_cells, 12), dtype=np.float64)
    fc = fit_all['cc'].values; ft = fit_all['t_abs'].values; fa = fit_all['TWS_t'].values - mu_c[fc]
    np.add.at(s_sum, (fc, ft % 12), fa); np.add.at(s_cnt, (fc, ft % 12), 1.0)
    seas = np.where(s_cnt >= 3, (s_sum/np.maximum(s_cnt, 1)).astype(np.float32), 0.0).astype(np.float32)
    def seas_feats(mg, X0):
        cc = mg['cc'].values; ta = mg['t_abs'].values
        return np.column_stack([X0[:,0] - seas[cc, ta % 12], seas[cc, (ta+1) % 12]]).astype(np.float32)
    se_tr = seas_feats(mg_tr, Xtr0); se_ev = seas_feats(mg_evo, Xev0)

    # (G1) assemble neighbor feature rows
    def nbr_feat(mg):
        ta = mg['t_abs'].values; cc = mg['cc'].values
        out = np.zeros(len(mg), dtype=np.float32)
        for m in np.unique(ta):
            s = ta == m
            out[s] = nbr_by_month[int(m)][cc[s]]
        return out
    nb_tr = nbr_feat(mg_tr); nb_ev = nbr_feat(mg_evo)

    FEATS = {
     'G1 nbr TWS@t (1-2deg)':    (nb_tr[:, None], nb_ev[:, None]),
     'G2 trailing 3/6-obs mean': (np.column_stack([tr3_tr, tr6_tr]), np.column_stack([tr3_ev, tr6_ev])),
     'G3 cov interactions':      (i_tr, i_ev),
     'G4 target-month dummies':  (md_tr, md_ev),
     'G5 cell trend @t+1':       (tx_tr[:, None], tx_ev[:, None]),
     'G6 seasonal resid+offset': (se_tr, se_ev),
    }
    np.savez_compressed('/home/z/my-project/scripts/auditB_cache/k0_feats.npz',
        nb_tr=nb_tr, nb_ev=nb_ev, tr3_tr=tr3_tr, tr6_tr=tr6_tr, tr3_ev=tr3_ev, tr6_ev=tr6_ev,
        tr3_m1=tr3_m1, tr6_m1=tr6_m1, tx_tr=tx_tr, tx_ev=tx_ev, md_tr=md_tr, md_ev=md_ev,
        i_tr=i_tr, i_ev=i_ev, se_tr=se_tr, se_ev=se_ev, m1_sel=m1_sel)
    log("features cached.")
else:
    z = np.load('/home/z/my-project/scripts/auditB_cache/k0_feats.npz')
    nb_tr, nb_ev = z['nb_tr'], z['nb_ev']
    tr3_tr, tr6_tr = z['tr3_tr'], z['tr6_tr']
    tr3_ev, tr6_ev = z['tr3_ev'], z['tr6_ev']
    tr3_m1, tr6_m1 = z['tr3_m1'], z['tr6_m1']
    tx_tr, tx_ev = z['tx_tr'], z['tx_ev']
    md_tr, md_ev = z['md_tr'], z['md_ev']
    i_tr, i_ev = z['i_tr'], z['i_ev']
    se_tr, se_ev = z['se_tr'], z['se_ev']
    log("features loaded from cache.")


# ---------- experiment harness ----------
def ext_linear(Xtr_c, ytr, wtr, Xev_c, n_extra):
    """v17b k0_linear + n_extra appended (always-finite) features, also in the reduced model.
    Xtr_c/Xev_c raw (may contain NaN in covs(t+1) cols 6..10)."""
    has_nxt_tr = np.isfinite(Xtr_c[:, 6:11]).all(axis=1)
    Xtr0 = np.nan_to_num(Xtr_c, nan=0.0)
    sw = np.sqrt(wtr)
    A = Xtr0[has_nxt_tr]*sw[has_nxt_tr, None]
    coefF = np.linalg.solve(A.T@A + 1e-3*np.eye(11+n_extra), A.T@(ytr[has_nxt_tr]*sw[has_nxt_tr]))
    colsR = np.array([0,1,2,3,4,5] + list(range(11, 11+n_extra)))
    A_r = Xtr0[:, colsR]*sw[:, None]
    coefR = np.linalg.solve(A_r.T@A_r + 1e-3*np.eye(6+n_extra), A_r.T@(ytr*sw))
    has_nxt_ev = np.isfinite(Xev_c[:, 6:11]).all(axis=1)
    Xe = np.nan_to_num(Xev_c, nan=0.0)
    pred = np.empty(len(Xev_c), dtype=np.float32)
    pred[has_nxt_ev] = Xe[has_nxt_ev] @ coefF
    pred[~has_nxt_ev] = Xe[~has_nxt_ev][:, colsR] @ coefR
    return pred

def run_config(extra_tr, extra_ev, tag):
    Xtr_c = np.column_stack([Xtr, extra_tr]).astype(np.float32) if extra_tr is not None else Xtr
    Xev_c = np.column_stack([Xev_o, extra_ev]).astype(np.float32) if extra_ev is not None else Xev_o
    n_extra = 0 if extra_tr is None else extra_tr.shape[1]
    pl = ext_linear(Xtr_c, ytr, wtr, Xev_c, n_extra)
    rl = float(np.sqrt(np.mean((pl - yev_o)**2)))
    pg, _ = k0_lgb(Xtr_c, ytr, wtr, Xev_c, rounds=400)
    rg = float(np.sqrt(np.mean((pg - yev_o)**2)))
    pb = 0.5*pl + 0.5*pg
    rb = float(np.sqrt(np.mean((pb - yev_o)**2)))
    rlm = float(np.sqrt(np.mean((pl[m1_sel] - yev_o[m1_sel])**2)))
    rgm = float(np.sqrt(np.mean((pg[m1_sel] - yev_o[m1_sel])**2)))
    rbm = float(np.sqrt(np.mean((pb[m1_sel] - yev_o[m1_sel])**2)))
    P(f"{tag:26s} LIN {rl:.4f} ({rl-r_lin0:+.4f})  LGB {rg:.4f} ({rg-r_lgb0:+.4f})  BLEND {rb:.4f} ({rb-r_bd0:+.4f}) | M1sub lin {rlm:.4f} lgb {rgm:.4f} blend {rbm:.4f}")
    return rl, rg, rb

P(f"\n=== k0 feature experiments phase {PHASE} (eval: 2013-15 unmasked rows, n={len(yev_o):,}) ===")
all_tr = np.column_stack([v[0] for v in FEATS.values()]); all_ev = np.column_stack([v[1] for v in FEATS.values()])
best = {'tag': 'base', 'rl': r_lin0, 'rg': r_lgb0, 'rb': r_bd0}
GROUPS = ['G1 nbr TWS@t (1-2deg)', 'G2 trailing 3/6-obs mean', 'G3 cov interactions'] if PHASE == 'A' \
         else ['G4 target-month dummies', 'G5 cell trend @t+1', 'G6 seasonal resid+offset']
for tag in GROUPS:
    etr, eev = FEATS[tag]
    rl, rg, rb = run_config(etr, eev, tag)
    if rb < best['rb']: best = {'tag': tag, 'rl': rl, 'rg': rg, 'rb': rb}
if PHASE == 'B':
    rl, rg, rb = run_config(all_tr, all_ev, 'ALL features')
    if rb < best['rb']: best = {'tag': 'ALL', 'rl': rl, 'rg': rg, 'rb': rb}
    # G2-only with M1-honest trailing (eval features swapped to M1 history)
    g2_m1_ev = np.column_stack([tr3_m1, tr6_m1])
    Xg2_tr = np.column_stack([Xtr, FEATS['G2 trailing 3/6-obs mean'][0]]).astype(np.float32)
    Xg2_ev = np.column_stack([Xev_o, g2_m1_ev]).astype(np.float32)
    pl2 = ext_linear(Xg2_tr, ytr, wtr, Xg2_ev, 2)
    pg2, _ = k0_lgb(Xg2_tr, ytr, wtr, Xg2_ev, rounds=400)
    pb2 = 0.5*pl2 + 0.5*pg2
    P(f"{'G2 (M1-honest trail)':26s} M1sub lin {float(np.sqrt(np.mean((pl2[m1_sel]-yev_o[m1_sel])**2))):.4f} "
      f"lgb {float(np.sqrt(np.mean((pg2[m1_sel]-yev_o[m1_sel])**2))):.4f} "
      f"blend {float(np.sqrt(np.mean((pb2[m1_sel]-yev_o[m1_sel])**2))):.4f}  (full-eval blend {float(np.sqrt(np.mean((pb2-yev_o)**2))):.4f})")

if PHASE == 'B':
    # M1-honest trailing swap for the ALL config (eval features only)
    order = list(FEATS.keys()); ti = order.index('G2 trailing 3/6-obs mean')
    off = sum(FEATS[k][0].shape[1] for k in order[:ti])
    all_ev_m1 = all_ev.copy()
    all_ev_m1[:, off:off+2] = np.column_stack([tr3_m1, tr6_m1])
    Xall_tr = np.column_stack([Xtr, all_tr]).astype(np.float32)
    Xall_ev_m1 = np.column_stack([Xev_o, all_ev_m1]).astype(np.float32)
    plA = ext_linear(Xall_tr, ytr, wtr, Xall_ev_m1, all_tr.shape[1])
    pgA, _ = k0_lgb(Xall_tr, ytr, wtr, Xall_ev_m1, rounds=400)
    pbA = 0.5*plA + 0.5*pgA
    P(f"{'ALL (M1-honest G2)':26s} M1sub lin {float(np.sqrt(np.mean((plA[m1_sel]-yev_o[m1_sel])**2))):.4f} "
      f"lgb {float(np.sqrt(np.mean((pgA[m1_sel]-yev_o[m1_sel])**2))):.4f} "
      f"blend {float(np.sqrt(np.mean((pbA[m1_sel]-yev_o[m1_sel])**2))):.4f}")


P(f"\nBEST config: {best['tag']} -> LINEAR {best['rl']:.4f}, LGB {best['rg']:.4f}, BLEND {best['rb']:.4f}")
P(f"base blend 0.6254 -> best {best['rb']:.4f} = {best['rb']-r_bd0:+.4f}")
with open(OUT, 'w') as f: f.write('\n'.join(lines))
log(f"saved {OUT}")
