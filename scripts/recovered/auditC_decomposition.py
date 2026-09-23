"""
AUDIT C (strategy & red team) — numeric checks.
C1: LB decomposition sensitivity + noise floors
C2: v12b/v13b anomaly arithmetic
C3: protocol realism — horizon / backward-gap distributions, test vs honest-CV val
C6: leader-gap bounds

Outputs printed; summarized in download/auditC_report.md.
"""
import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')

DATA = '/home/z/my-project/data'
W_M, W_K = 186913/280961, 94048/280961   # masked / k0 row weights (full test)

# ---------------- C1: decomposition sensitivity ----------------
print("="*72)
print("C1. LB DECOMPOSITION SENSITIVITY   LB^2 = %.4f*masked^2 + %.4f*k0^2" % (W_M, W_K))
print("="*72)
lb = {'v12b': 0.695357171, 'v17b': 0.704955918, 'v2b': 0.7137, 'v17a_proj': 0.6894}
def masked_given_k0(lb_score, k0):
    m2 = (lb_score**2 - W_K*k0**2)/W_M
    return np.sqrt(m2) if m2 > 0 else np.nan

print(f"\n{'k0 assumed':>12} | {'v12b masked':>12} {'v17b masked':>12} {'gap(17b-12b)':>13}")
for k0 in [0.60, 0.62, 0.6254, 0.64, 0.66]:
    m12, m17 = masked_given_k0(lb['v12b'], k0), masked_given_k0(lb['v17b'], k0)
    print(f"{k0:>12.4f} | {m12:>12.4f} {m17:>12.4f} {m17-m12:>13.4f}")

# same-k0 invariance: if both submissions share the same k0, masked gap is fixed
g = (lb['v17b']**2 - lb['v12b']**2)/W_M
print(f"\nIf v12b & v17b share the SAME k0 (any value):")
print(f"  masked_17b^2 - masked_12b^2 = {g:.5f}  -> gap ~ {g/(2*0.73):.4f} at masked~0.73")
# but if k0s differ by dk0:
for dk in [-0.02, -0.01, 0.01, 0.02]:
    gg = g + W_K*((0.64+dk/2)**2 - (0.64-dk/2)**2)/W_M   # k0_17 = k0_12 + dk
    print(f"  if k0_17b - k0_12b = {dk:+.2f}: masked gap ~ {gg/(2*0.73):+.4f}")

# ---------------- noise floors ----------------
print("\n" + "="*72)
print("NOISE FLOORS (RMSE standard errors, iid-Gaussian approx, kurtosis=0)")
print("="*72)
n_honest_cv = 93829
cv_best = 0.6535
se_cv = cv_best/np.sqrt(2*n_honest_cv)
print(f"honest-CV masked (n={n_honest_cv:,}): SE(RMSE) ~ {se_cv:.5f}  -> 2SE = {2*se_cv:.5f}")
print(f"  => CV deltas below ~{2*se_cv:.3f} (e.g. phi 0.6535 vs 0.6544) are NOT resolvable")
n_pub = int(0.30*280961)
se_lb = 0.70/np.sqrt(2*n_pub)
print(f"public LB (n~{n_pub:,}, 30% split): SE ~ {se_lb:.5f} across *different* splits;")
print(f"  for FIXED public rows, identical predictions => identical score EXACTLY.")
print(f"  paired 2-submission comparison on same rows: SE(diff) << {se_lb:.5f}")

# ---------------- C2: anomaly arithmetic ----------------
print("\n" + "="*72)
print("C2. v12b/v13b ANOMALY ARITHMETIC")
print("="*72)
d = 0.697421091 - 0.695357171
print(f"reported delta = {d:.6f} ({d/0.695357171*100:.3f}%)")
print(f"implied mean-squared-error difference per row (if public=full test):")
print(f"  dMSE = LB13^2 - LB12^2 = {0.697421091**2 - 0.695357171**2:.6f}")
print(f"  over 280,961 rows -> total squared-error diff = {280961*(0.697421091**2 - 0.695357171**2):,.0f}")
print(f"  = e.g. {280961*(0.697421091**2-0.695357171**2)/ (2*0.7):,.0f} rows changed by ~1.0 RMSE, or")
print(f"  ~{280961*(0.697421091**2-0.695357171**2)/(2*0.7*0.01):,.0f} rows changed by 0.01")
print("  -> a 0.0021 LB delta REQUIRES large real changes on scored rows (if files differ)")
print("     or is impossible (if files identical & scoring deterministic)")

# ---------------- C3: protocol realism ----------------
print("\n" + "="*72)
print("C3. PROTOCOL REALISM: horizon & backward-gap distributions")
print("="*72)
t = pd.read_csv(f'{DATA}/Test (2).csv', usecols=['time','TWS_t_masked'])
t['time'] = pd.to_datetime(t['time'])
t['ym'] = t['time'].dt.year*100 + t['time'].dt.month
t['t_abs'] = (t['ym']//100)*12 + (t['ym']%100) - 1
t['masked'] = t['TWS_t_masked'].astype(bool)
mfrac = t.groupby('t_abs')['masked'].mean()
anchors_t = np.array(sorted(mfrac[mfrac < 0.01].index))
print(f"test anchors (t_abs): {anchors_t.tolist()}")
ta = t['t_abs'].values; msk = t['masked'].values
h_test, bgap_test = [], []
for m in sorted(set(ta[msk])):
    sel = (ta == m) & msk
    prev = anchors_t[anchors_t <= m]
    nxt = anchors_t[anchors_t > m]
    h = m - prev[-1]
    bg = (nxt[0] - (m+1)) if len(nxt) else np.inf
    h_test += [h]*int(sel.sum())
    bgap_test += [bg]*int(sel.sum())
h_test = np.array(h_test); bgap_test = np.array(bgap_test)
print(f"\nTEST masked rows n={len(h_test):,}: horizon h = months since last anchor")
for h in sorted(set(h_test)):
    print(f"  h={h}: {(h_test==h).sum():>7,} rows ({(h_test==h).mean()*100:.1f}%)")
fin = np.isfinite(bgap_test)
print(f"backward gap (next anchor - target month), rows with a later anchor: {fin.sum():,}")
for g in sorted(set(bgap_test[fin])):
    print(f"  gap={int(g)}: {(bgap_test==g).sum():>7,} rows ({(bgap_test==g).mean()*100:.1f}%)")
print(f"  no later anchor (fwd-only rows): {(~fin).sum():,} ({(~fin).mean()*100:.1f}%)")

# val honest protocol replication (2013-15, calendar-month mask, honest rows)
tr = pd.read_csv(f'{DATA}/Train (1).csv', usecols=['time','lat','lon','TWS_t'])
tr['time'] = pd.to_datetime(tr['time'])
val = tr[(tr['time'].dt.year >= 2013) & (tr['time'].dt.year <= 2015)].copy()
cal = t.groupby(t['time'].dt.month)['masked'].mean()
val['masked'] = val['time'].dt.month.map(cal).values > 0.5
val['ym'] = val['time'].dt.year*100 + val['time'].dt.month
val['t_abs'] = (val['ym']//100)*12 + (val['ym']%100) - 1
mfrac_v = val.groupby('t_abs')['masked'].mean()
anchors_v = np.array(sorted(mfrac_v[mfrac_v < 0.01].index))
ta_v = val['t_abs'].values; msk_v = val['masked'].values
shortcut = np.array([(tt+1) in set(anchors_v.tolist()) for tt in ta_v])
honest = msk_v & ~shortcut
h_val, bgap_val = [], []
for m in sorted(set(ta_v[honest])):
    sel = (ta_v == m) & honest
    prev = anchors_v[anchors_v <= m]
    nxt = anchors_v[anchors_v > m]
    hh = m - prev[-1]
    bg = (nxt[0] - (m+1)) if len(nxt) else np.inf
    h_val += [hh]*int(sel.sum())
    bgap_val += [bg]*int(sel.sum())
h_val = np.array(h_val); bgap_val = np.array(bgap_val)
print(f"\nVAL honest rows n={len(h_val):,} (anchors every ~{np.diff(anchors_v).mean():.1f} months):")
for h in sorted(set(h_val)):
    print(f"  h={h}: {(h_val==h).sum():>7,} rows ({(h_val==h).mean()*100:.1f}%)")
finv = np.isfinite(bgap_val)
print(f"backward gap distribution (val):")
for g in sorted(set(bgap_val[finv])):
    print(f"  gap={int(g)}: {(bgap_val==g).sum():>7,} rows ({(bgap_val==g).mean()*100:.1f}%)")
print(f"  no later anchor: {(~finv).sum():,} ({(~finv).mean()*100:.1f}%)")
print(f"\nMAX h: test={h_test.max()}, val={h_val.max()}")
print(f"MAX bwd gap: test={np.nanmax(bgap_test)}, val={np.nanmax(bgap_val[finv]) if finv.any() else 0}")
sh_test = np.histogram(h_test, bins=[0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5])[0]/len(h_test)
sh_val = np.histogram(h_val, bins=[0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5])[0]/len(h_val)
print(f"h-distribution L1 distance (test vs val): {np.abs(sh_test[:len(sh_val)]-sh_val).sum()/2:.3f}")

# ---------------- C6: leader bounds ----------------
print("\n" + "="*72)
print("C6. LEADER-GAP BOUNDS (leader LB 0.5596)")
print("="*72)
L = 0.5596
print(f"masked_max (if k0=0, impossible): {np.sqrt(L**2/W_M):.4f}")
for k0 in [0.50, 0.55, 0.58, 0.60, 0.62]:
    print(f"  if leader k0={k0:.2f}: leader masked <= {masked_given_k0(L, k0):.4f}")
print("\nfloor decomposition scenarios (masked^2 = noise^2 + D^2 + fast^2):")
for noise in [0.36, 0.40, 0.456]:
    for D in [0.42, 0.30, 0.20]:
        for f in [0.37, 0.25]:
            m = np.sqrt(noise**2 + D**2 + f**2)
            lbm = np.sqrt(W_M*m**2 + W_K*0.62**2)
            print(f"  noise={noise:.3f} D={D:.2f} fast={f:.2f} -> masked={m:.3f}, LB~{lbm:.3f}"
                  + ("   <- leader range" if m < 0.60 else ""))
print("\nimplied LB gain per masked-only improvement (at masked~0.72, LB~0.70):")
for dm in [0.005, 0.01, 0.02, 0.04]:
    dl = (W_M*(0.72**2 - (0.72-dm)**2))/(2*0.70)
    print(f"  masked -{dm:.3f} -> LB -{dl:.4f}")
print("implied LB gain per k0-only improvement (at k0~0.64):")
for dk in [0.005, 0.01, 0.02]:
    dl = (W_K*(0.64**2 - (0.64-dk)**2))/(2*0.70)
    print(f"  k0 -{dk:.3f} -> LB -{dl:.4f}")
