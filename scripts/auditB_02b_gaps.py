"""
AUDIT B - 02b (clean): per-month fwd/bwd decomposition on HONEST rows + test-gap projection (B6).
"""
import numpy as np, pandas as pd, time, sys
sys.path.insert(0, '/home/z/my-project/scripts')
from auditB_lib import *
t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)

tr, te = load_cached()
train = build_train_df(tr)
n_cells = int(train['cc'].max())+1
test = pd.read_csv(f'{DATA}/Test (2).csv', usecols=['time','TWS_t_masked'])
test['time'] = pd.to_datetime(test['time'])
cal = test.assign(m=test['TWS_t_masked'].astype(bool)).groupby(test['time'].dt.month)['m'].mean()

train['year'] = train['t_abs']//12
val = train[(train['year'] >= 2013) & (train['year'] <= 2015)].copy()
val['masked'] = val['t_abs'].apply(lambda t: (t%12+1)).map(cal).values > 0.5
val_tws_visible = val['TWS_t'].copy()
val_target = val['target'].values.astype(np.float64)
val_msk = val['masked'].values; val_ta = val['t_abs'].values; val_cc = val['cc'].values
mfrac_v = val.groupby('t_abs')['masked'].mean()
val_anchor_tas = set(int(v) for v in mfrac_v[mfrac_v < 0.01].index)
shortcut = np.array([(t+1) in val_anchor_tas for t in val_ta])
honest = val_msk & ~shortcut

fit = train[train['year'] <= 2012].copy()
infra_cv = build_infra(fit, n_cells, 'cv')

p_fwd, _ = run_masked2(infra_cv, val, val_tws_visible, 0.80, use_bwd=False, use_denoise=False, lam=0.84, diag=True)
p_blend, DG = run_masked2(infra_cv, val, val_tws_visible, 0.80, use_bwd=True, use_denoise=False, lam=0.84, diag=True)

meta = DG['meta']; rows = DG['rows_list']
starts = np.cumsum([0]+[len(r) for r in rows])

def rmse(x): x=x[np.isfinite(x)]; return float(np.sqrt(np.mean(x**2)))

log("=== per honest month: fwd vs bwd vs blend (phi.80/lam.84, HONEST rows only) ===")
E_blend, E_fwd, E_bwd, GF, GB, TGT, ROWS = [], [], [], [], [], [], []
for j in range(len(meta)):
    m = int(meta['m'].iloc[j]); gb = int(meta['gap_back'].iloc[j]); gf = int(meta['gap_fwd'].iloc[j])
    s, e = starts[j], starts[j+1]
    rws = rows[j]
    hon = honest[rws]                      # honest subset of this masked month
    t_ = val_target[rws][hon]
    mu_ = infra_cv['mu_c'][val_cc[rws]][hon]
    dT_ = DG['Dtil'][s:e][hon]
    xf_ = DG['xf'][s:e][hon]; xb_ = DG['xb'][s:e][hon]; w_ = DG['wgt'][s:e][hon]
    e_fwd = mu_+dT_+xf_-t_
    e_bw = mu_+dT_+xb_-t_
    e_bl = mu_+dT_+(xf_+w_*(xb_-xf_))-t_
    log(f"  {m//12}-{m%12+1:02d} n={hon.sum():6,} back={gb} fwdGap={gf:>2}: blend={rmse(e_bl):.4f} fwd={rmse(e_fwd):.4f} bwd={rmse(e_bw) if np.isfinite(e_bw).any() else float('nan'):.4f} meanW_fwd={np.nanmean(w_):.3f}")
    E_blend.append(e_bl); E_fwd.append(e_fwd); E_bwd.append(e_bw)
    GF.append(np.full(hon.sum(), gf)); GB.append(np.full(hon.sum(), gb))
    TGT.append(t_); ROWS.append(rws[hon])
E_blend = np.concatenate(E_blend); E_fwd = np.concatenate(E_fwd); E_bwd = np.concatenate(E_bwd)
GF = np.concatenate(GF); GB = np.concatenate(GB); TGT = np.concatenate(TGT)
log(f"  POOLED honest: blend={rmse(E_blend):.4f} fwd={rmse(E_fwd):.4f}")

log("\n=== bwd gain by fwd-gap bucket (honest rows) ===")
tbl = {}
for gf in sorted(set(GF.tolist())):
    s = GF == gf
    rb = rmse(E_bwd[s]) if np.isfinite(E_bwd[s]).any() else float('nan')
    tbl[gf] = (rmse(E_blend[s]), rmse(E_fwd[s]), rb, int(s.sum()))
    log(f"  fwdGap={gf:>2}: n={int(s.sum()):6,} blend={rmse(E_blend[s]):.4f} fwd={rmse(E_fwd[s]):.4f} bwd={rb:.4f}")

log("\n=== fwd-only RMSE by back-gap bucket (honest rows) ===")
tblb = {}
for gb in sorted(set(GB.tolist())):
    s = GB == gb
    tblb[gb] = rmse(E_fwd[s])
    log(f"  backGap={gb}: n={int(s.sum()):6,} fwd={rmse(E_fwd[s]):.4f}")

# ---------------- TEST projection ----------------
# TEST masked-row structure (recomputed from census):
#   fwdGap(next anchor - tm): 2:31013, 3:31078, 4:15633, small:5..11 ~169, 12..17: ~92490, none:15666
#   backGap(tm - prev anchor): 2:62576, 3:46777, 4:31076, 5:15560, 6:15479, 7:15445
test_fwd = {2:31013, 3:31078, 4:15633+91+38+4+9+9+9+3+1, 12:15486, 13:15484, 14:15518, 15:15498, 16:15521, 17:15523, 18:83+90, 19:82+83, 20:0, 21:0}
n_none = 15666
fwd_g2 = tbl[2][1]; fwd_g3 = tbl[3][1]          # fwd-only RMSE at val fwd-gap 2/3
blend_g1, blend_g2, blend_g3 = tbl[1][0], tbl[2][0], tbl[3][0]
fwd_by_back = {k: tblb[k] for k in tblb}
# 2017-block (fwdGap>=12): bwd contributes ~nothing -> fwd-only, with test back-gap mix
mix_back_2017 = {2:0, 3:0, 4:15560+0, 5:15479, 6:15445, 7:15486}  # 2017-02..07 targets: back 2..7
# precise: 2017-01..06 rows -> tm=2017-02..07, prev anchor 2016-12 -> backGap = tm-2016-12 = 2..7 (approx equal counts)
mix_back_2017 = {2:15486, 3:15484, 4:15518, 5:15498, 6:15521, 7:15523}
mix_back_2016 = {2:31013+31078, 3:0, 4:15560, 5:15479, 6:15445, 7:0}  # rough: 2016 block back gaps 2..6
n_none_back = {2:15666}
def se_mix(mix):
    se, n = 0.0, 0
    for gb, cnt in mix.items():
        r = fwd_by_back.get(gb, fwd_by_back[max(fwd_by_back)])
        se += cnt*r**2; n += cnt
    return se, n
# 2016 block: use blend at matching fwd gap (gap2->blend_g2, gap3->blend_g3, gap4->blend_g3 conservative-optimistic)
se_2016 = 31013*blend_g2**2 + 31078*blend_g3**2 + (15633+176)*blend_g3**2
n_2016 = 31013+31078+15633+176
# 2017 block: fwd-only with back-gap mix
se_2017, n_2017 = se_mix(mix_back_2017)
# none block: fwd-only, back-gap 2
se_none, n_none_ = 15666*fwd_by_back[2]**2, 15666
tot_n = n_2016+n_2017+n_none_
tot_se_blend = se_2016+se_2017+se_none
tot_se_fwd = (n_2016*(0.5*(fwd_g2**2+fwd_g3**2)) ) + se_2017 + se_none
log("\n=== TEST-STRUCTURE PROJECTION (val-measured per-gap RMSEs, test gap mix) ===")
log(f"  2016 block (n={n_2016:,}, fwdGap 2-4): blend assumed = val blend at gap2/3")
log(f"  2017 block (n={n_2017:,}, fwdGap 12-17): bwd ~useless -> fwd-only, back-gap mix 2..7")
log(f"  2018-12 block (n={n_none_:,}, no bwd): fwd-only back-gap 2")
log(f"  => projected TEST masked RMSE: with bwd = {np.sqrt(tot_se_blend/tot_n):.4f} | fwd-only = {np.sqrt(tot_se_fwd/tot_n):.4f}")
log(f"  (val honest measured: blend={rmse(E_blend):.4f}, fwd={rmse(E_fwd):.4f})")
log(f"  => bwd gain shrinks from {rmse(E_fwd)-rmse(E_blend):.4f} (val) to ~{np.sqrt(tot_se_fwd/tot_n)-np.sqrt(tot_se_blend/tot_n):.4f} (test-structure)")

np.savez_compressed('/home/z/my-project/scripts/auditB_b6.npz',
    E_blend=E_blend, E_fwd=E_fwd, E_bwd=E_bwd, GF=GF, GB=GB, TGT=TGT,
    ROWS=np.concatenate(ROWS))
log("DONE")
