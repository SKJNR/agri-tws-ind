"""SPLIT DECODE — what do probe_PK=0.9365 and probe_PE=1.1226 tell us?

Measured facts (PUBLIC scores, RMSE lower-is-better):
  v8a  = 0.70702259    (probe base)
  v6c  = 0.703041139
  v10b = 0.699997215   (new best)
  PK   = 0.936486593   (v8a with all k0 rows -> climatology mu_c)
  PE   = 1.122620331   (v8a with months 2015-09..2017-04 -> mu_c)

Questions this script answers (offline, no submission slots):
  Q1  Is the public split random, or time-blocked at the start of the test era?
  Q2  What are our masked-class / k0-class PUBLIC RMSEs?
  Q3  Does the D-offset field GROW over the test era (small early, big late)?
      -> if yes, era-weighted Dhat is the biggest untapped axis, and the
         private LB (majority late months) will be harder than public.

Method:
  A. Test anchor fields AF[a] = TWS_a - mu_c at the 6 fully-unmasked months.
     std(AF[a]) = climatology RMSE at month a = sqrt(D(t)^2 + V_fast)
     -> direct measurement of the D(t) profile (V_fast from train).
  B. Train: per-calendar-month detrended-anomaly variance -> seasonality of V_fast
     (k0-calendar months 1,6,7,9,11,12 vs masked-calendar months 2,3,4,5,8).
  C. Solve the probe equations under each split hypothesis; the implied public
     k0-share / early-share must match the expected share of that window.
  D. Decode class RMSEs; MOHAR's implied class RMSEs; what-if table.
  E. Val-window sanity checks (mu fitted on 2002-2012 only).
"""
import numpy as np, pandas as pd

DATA = '/home/z/my-project/data'
COVS = ['SPEI_01_t','SPEI_03_t','SPEI_06_t','SPEI_12_t','SOIL_MOISTURE_t']

V8A, V6C, V10B, PK, PE = 0.70702259, 0.703041139, 0.699997215, 0.936486593, 1.122620331

# ---------------- load train ----------------
print("loading train...", flush=True)
train = pd.read_csv(f'{DATA}/Train (1).csv', usecols=['time','lat','lon','TWS_t'])
train['time'] = pd.to_datetime(train['time'])
train['TWS_t'] = pd.to_numeric(train['TWS_t'], errors='coerce').astype('float32')
train['cc'] = (train['lat'].round(2).astype(str)+'_'+train['lon'].round(2).astype(str)).astype('category').cat.codes.astype('int32')
train['ym'] = train['time'].dt.year*100 + train['time'].dt.month
train['t_abs'] = (train['ym']//100)*12 + (train['ym']%100) - 1
n_cells = int(train['cc'].max())+1
yms = np.sort(train['ym'].unique())
ym_to_i = {int(v):i for i,v in enumerate(yms)}
t_abs_train = np.array([(int(v)//100)*12 + (int(v)%100) - 1 for v in yms], dtype=np.float64)

F = np.full((len(yms), n_cells), np.nan, dtype=np.float32)
F[train['ym'].map(ym_to_i).values, train['cc'].values] = train['TWS_t'].values
mu_c = np.nanmean(F, axis=0)

# detrended anomaly (per-cell linear trend removed) -> fast+noise marginal variance
F64 = F.astype(np.float64); ok_t = np.isfinite(F64)
t_mat = np.where(ok_t, t_abs_train[:, None], np.nan)
tbar_c = np.nanmean(t_mat, axis=0); td = t_mat - tbar_c[None, :]
beta_c = np.where((np.nansum(td*td, axis=0) > 100), np.nansum(td*F64, axis=0)/np.maximum(np.nansum(td*td, axis=0),1), 0)
A_dt = F64 - mu_c[None, :] - td*beta_c[None, :]

# ---------------- load test ----------------
print("loading test...", flush=True)
test = pd.read_csv(f'{DATA}/Test (2).csv', usecols=['time','lat','lon','TWS_t','TWS_t_masked'])
test['time'] = pd.to_datetime(test['time'])
test['TWS_t'] = pd.to_numeric(test['TWS_t'], errors='coerce').astype('float32')
codes = train[['cc','lat','lon']].drop_duplicates('cc').sort_values('cc')
pos = {(int(round(float(la)*1000)), int(round(float(lo)*1000))): int(cc) for cc,la,lo in zip(codes['cc'],codes['lat'],codes['lon'])}
test['cc'] = [pos.get((int(round(float(a)*1000)), int(round(float(b)*1000))), -1) for a,b in zip(test['lat'], test['lon'])]
test['ym'] = test['time'].dt.year*100 + test['time'].dt.month
test['t_abs'] = (test['ym']//100)*12 + (test['ym']%100) - 1
test['masked'] = test['TWS_t_masked'].astype(bool)
ta = test['t_abs'].values; cc_t = test['cc'].values; msk = test['masked'].values
test_months = np.array(sorted(test['t_abs'].unique()))
month0 = int(test_months.min())

# k0 share overall
s_k0_all = float((~msk).mean())
print(f"\ntest rows={len(test)}  k0 share (all 40 months)={s_k0_all:.4f}")

# ---------------- B. seasonal variance profile from train ----------------
print("\n=== B. per-calendar-month fast+noise variance (train, detrended anomaly) ===")
cal = (t_abs_train.astype(int) % 12) + 1
vm = {}
for m in range(1, 13):
    vm[m] = float(np.nanvar(A_dt[cal == m]))
med = np.median(list(vm.values()))
for m in range(1, 13):
    bar = '#' * int(40*vm[m]/med)
    print(f"  month {m:2d}: var={vm[m]:.4f}  std={np.sqrt(vm[m]):.4f}  {bar}")
# which calendar months are ~100% masked in test?
mfrac_cal = test.groupby(test['ym'] % 100)['masked'].mean()
masked_cals = sorted(mfrac_cal[mfrac_cal > 0.9].index.astype(int))
k0_cals = [m for m in range(1,13) if m not in masked_cals]
v_masked = float(np.mean([vm[m] for m in masked_cals]))
v_k0 = float(np.mean([vm[m] for m in k0_cals]))
print(f"  masked-calendar months {masked_cals}: mean var={v_masked:.4f} std={np.sqrt(v_masked):.4f}")
print(f"  k0-calendar months     {k0_cals}: mean var={v_k0:.4f} std={np.sqrt(v_k0):.4f}")
print(f"  variance ratio k0/masked = {v_k0/v_masked:.3f}  (1.0 = no seasonality)")

# ---------------- A. anchor fields + D profile ----------------
print("\n=== A. test anchor fields: clim-RMSE profile over the test era ===")
mfrac = test.groupby('t_abs')['masked'].mean()
anchors = sorted(int(v) for v in mfrac[mfrac < 0.01].index)
AF = {}
for a in anchors:
    sel = (ta == a) & (~msk)
    fa = np.full(n_cells, np.nan, dtype=np.float32); fa[cc_t[sel]] = test['TWS_t'].values[sel]
    AF[a] = fa - mu_c
print(f"anchors (t_abs, ym): {[(a, int(a//12)*100+a%12+1) for a in anchors]}")
print(f"{'anchor':>8} {'ym':>7} {'std(AF)':>8} {'D^2 est':>8} {'D std est':>9}")
D2 = {}
for a in anchors:
    cal_m = (a % 12) + 1
    vfast = vm[cal_m]
    s = float(np.nanstd(AF[a]))
    D2[a] = max(s*s - vfast, 0.0)
    print(f"{a:>8} {int(a//12)*100+a%12+1:>7} {s:>8.4f} {D2[a]:>8.4f} {np.sqrt(D2[a]):>9.4f}")

early_anchors = [a for a in anchors if a < month0 + 20]
late_anchors = [a for a in anchors if a >= month0 + 20]
print(f"early anchors: {[int(a//12)*100+a%12+1 for a in early_anchors]}  late anchors: {[int(a//12)*100+a%12+1 for a in late_anchors]}")
Dh_e = np.nanmean(np.array([AF[a] for a in early_anchors]), axis=0)
Dh_l = np.nanmean(np.array([AF[a] for a in late_anchors]), axis=0)
ok = np.isfinite(Dh_e) & np.isfinite(Dh_l)
print(f"std(Dhat_early)={np.nanstd(Dh_e):.4f}  std(Dhat_late)={np.nanstd(Dh_l):.4f}  "
      f"std(Dh_late-Dh_early)={np.nanstd(Dh_l[ok]-Dh_e[ok]):.4f}  corr={np.corrcoef(Dh_e[ok], Dh_l[ok])[0,1]:.4f}")

# ---------------- C. hypothesis tests ----------------
print("\n=== C. probe equations under each split hypothesis ===")
# expected clim RMSE at each test month (extrapolate D^2 linearly in t between anchors)
anchor_t = np.array(anchors, dtype=float)
anchor_D2 = np.array([D2[a] for a in anchors])
def clim_rmse2_at(t_abs_arr, cal_arr):
    # D^2(t): linear fit through anchor D^2 values (t in months)
    tt = np.asarray(t_abs_arr, dtype=float)
    d2 = np.interp(tt, anchor_t, anchor_D2)
    return d2 + np.array([vm[c] for c in cal_arr])

cals_te = (test['t_abs'].values % 12) + 1
E2 = clim_rmse2_at(ta, cals_te)          # per-row clim RMSE^2 (class-independent up to seasonality)

def window_stats(idx, name):
    n = int(idx.sum())
    s_k0 = float((~msk[idx]).mean())
    e2 = float(np.mean(E2[idx]))
    print(f"  {name:<28} rows-share={n/len(test):.3f}  k0-share={s_k0:.3f}  pred-clim-RMSE={np.sqrt(e2):.4f}")
    return s_k0, np.sqrt(e2)

print("window k0-shares and predicted clim RMSE:")
w_all = np.ones(len(test), bool)
w_12  = ta < month0 + 12
w_20  = ta < month0 + 20
s_all, C_all = window_stats(w_all, "all 40 months")
s_12,  C_12  = window_stats(w_12,  "first 12 months (30%)")
s_20,  C_20  = window_stats(w_20,  "first 20 months (50%)")

print(f"\nFACTS: PK={PK:.6f}  PE={PE:.6f}  v8a={V8A:.6f}  v6c={V6C:.6f}  v10b={V10B:.6f}")
print(f"  PK^2 - v8a^2 = {PK**2 - V8A**2:.6f} = s_k0_pub * (C_pub^2 - K8_pub^2)")
print(f"  PE^2 - v8a^2 = {PE**2 - V8A**2:.6f} = s_early_pub * (E_pub^2 - V8_early^2)")

print("\n-- H1: public = RANDOM (representative) --")
print(f"   needs s_k0=0.335 -> C_pub^2 - K8^2 = {(PK**2-V8A**2)/s_k0_all:.4f}")
for K8 in (0.55, 0.60, 0.65):
    C_imp = np.sqrt((PK**2-V8A**2)/s_k0_all + K8**2)
    print(f"   K8={K8:.2f} -> implied C_pub(k0 clim RMSE)={C_imp:.4f}   [predicted C_all={C_all:.4f}]")
# PE under random: PE^2 = 0.5*E_early^2 + 0.5*V8_late^2, V8_late~V8A
E_e2 = float(np.mean(E2[w_20]))
for V8l in (0.70, 0.75, 0.80):
    pe_pred = np.sqrt(0.5*E_e2 + 0.5*V8l**2)
    print(f"   V8_late={V8l:.2f} -> predicted PE={pe_pred:.4f}  (actual {PE:.4f})")

print("\n-- H2: public = TIME-BLOCKED first-N months --")
for N, (wn, sw, Cw) in [(12, (w_12, s_12, C_12)), (20, (w_20, s_20, C_20))]:
    # PE: public months all inside 2015-09..2017-04 (first 20) -> PE = C_pub
    print(f"   N={N}: predicted PE = C_pub = {Cw:.4f}  (actual {PE:.4f})")
    # PK: implied s_k0 for K8 in range
    for K8 in (0.55, 0.60, 0.65):
        s_imp = (PK**2 - V8A**2)/(Cw**2 - K8**2)
        print(f"      K8={K8:.2f} -> implied s_k0_pub={s_imp:.3f}   [expected k0-share of window={sw:.3f}]")

print("\n-- H3: public = first 20 months, v8a has early/late asymmetry --")
# v8a^2 = s20*K8e^2 + (1-s20)*M8e^2 ; PK^2 = s20*C_pub^2 + (1-s20)*M8e^2
# -> s20*(C^2-K8e^2) = PK^2-v8a^2 ; try K8e, solve M8e
for K8e in (0.58, 0.62, 0.66):
    s_imp = (PK**2 - V8A**2)/(C_20**2 - K8e**2)
    M8e2 = (V8A**2 - s_imp*K8e**2)/max(1-s_imp, 1e-3)
    print(f"   K8e={K8e:.2f} -> s_k0={s_imp:.3f} (exp {s_20:.3f}), M8e={np.sqrt(max(M8e2,0)):.4f}")

# ---------------- D. decode + what-if ----------------
print("\n=== D. class-RMSE decode (under best-fit hypothesis) + what-if ===")
# use v6c/v10b pair (k0-only change) to get K10, then MOHAR
d2 = V6C**2 - V10B**2
print(f"v6c->v10b gain: {V6C-V10B:+.6f}  (k0-axis, masked rows unchanged)")
for s in (s_12, s_20, s_k0_all):
    print(f"   s_k0={s:.3f}: dK^2={d2/s:.5f} -> e.g. K6=0.61->K10={np.sqrt(0.61**2-d2/s):.4f}")
# MOHAR under first-20 hypothesis
print(f"\nMOHAR 0.5596 under first-20 (s_k0={s_20:.3f}):")
for K_m in (0.45, 0.50, 0.55):
    M_m = np.sqrt((0.5596**2 - s_20*K_m**2)/(1-s_20))
    print(f"   if their k0={K_m:.2f} -> their masked={M_m:.4f}  (ours ~0.75)")

# ---------------- E. val sanity ----------------
print("\n=== E. val-window sanity (mu from 2002-2012, val 2013-2015) ===")
tr_mask = train['ym'] < 201301
mu_h = train[tr_mask].groupby('cc')['TWS_t'].mean().reindex(range(n_cells)).values.astype('float32')
val = train[~tr_mask]
vc = val['cc'].values
err = val['TWS_t'].values - mu_h[vc]
vh = val['ym'].values
h1 = vh <= 201406; h2 = vh > 201406
print(f"val clim RMSE: all={np.sqrt(np.nanmean(err**2)):.4f}  2013H1-2014H1={np.sqrt(np.nanmean(err[h1]**2)):.4f}  "
      f"2014H2-2015={np.sqrt(np.nanmean(err[h2]**2)):.4f}")
# val D estimate: mean val field - honest mu
Fc = np.full(n_cells, np.nan, dtype=np.float32)
Fc[np.unique(val['cc'].values)] = val.groupby('cc')['TWS_t'].mean().reindex(range(n_cells)).values
Dv = Fc - mu_h
print(f"val-era offset std (32-month mean, incl. fast-avg noise)={np.nanstd(Dv):.4f}")
print("\nDONE.")
