"""
Verify other AI's claim: test-period phi might be 0.859, not 0.95.
Recompute cross-anchor correlations using ONLY strictly-fully-unmasked months.
"""
import numpy as np, pandas as pd

DATA = '/home/z/my-project/data'
train = pd.read_csv(f'{DATA}/Train (1).csv', usecols=['time','lat','lon','TWS_t'])
train['time'] = pd.to_datetime(train['time'])
train['TWS_t'] = pd.to_numeric(train['TWS_t'], errors='coerce').astype('float32')
train['cc'] = (train['lat'].round(1).astype(str) + '_' + train['lon'].round(1).astype(str)).astype('category').cat.codes.astype('int32')
train['ym'] = train['time'].dt.year*100 + train['time'].dt.month
n_cells = int(train['cc'].max())+1

test = pd.read_csv(f'{DATA}/Test (2).csv')
test['time'] = pd.to_datetime(test['time'])
test['TWS_t'] = pd.to_numeric(test['TWS_t'], errors='coerce').astype('float32')
codes = train[['cc','lat','lon']].drop_duplicates()
cmap = {(round(float(la),1),round(float(lo),1)): int(cc) for cc,la,lo in zip(codes['cc'],codes['lat'],codes['lon'])}
test['cc'] = [cmap.get((round(float(la),1),round(float(lo),1)),-1) for la,lo in zip(test['lat'],test['lon'])]
test['ym'] = test['time'].dt.year*100 + test['time'].dt.month
test['masked'] = test['TWS_t_masked'].astype(bool)

# strict: a month is "fully unmasked" if >99% of cells are unmasked
month_mask_frac = test.groupby('ym')['masked'].mean()
print("All test months with mask fraction:")
for ym, frac in month_mask_frac.items():
    flag = "FULL" if frac < 0.01 else f"partial ({frac*100:.1f}% masked)"
    print(f"  {ym}: {flag}")

strict_anchors = sorted(month_mask_frac[month_mask_frac < 0.01].index.tolist())
print(f"\nStrict fully-unmasked months: {strict_anchors}")
print(f"Count: {len(strict_anchors)}")

# also check what "lenient" anchors (frac < 0.5) look like
lenient_anchors = sorted(month_mask_frac[month_mask_frac < 0.5].index.tolist())
print(f"\nLenient anchors (other AI's original criterion, mask<50%): {lenient_anchors}")

# build TWS field per ym (cell, ym) -> TWS_t value (use train+test combined)
all_yms_train = sorted(train['ym'].unique())
all_yms_test = sorted(test['ym'].unique())
all_yms = sorted(set(all_yms_train + all_yms_test))
ym_to_i = {v:i for i,v in enumerate(all_yms)}

# field[ym_idx, cell_idx] -> TWS_t
F = np.full((len(all_yms), n_cells), np.nan, dtype=np.float32)
F[train['ym'].map(ym_to_i).values, train['cc'].values] = train['TWS_t'].values
unm_test = ~test['masked'].values
F_idx = test['ym'].map(ym_to_i).values
F[F_idx[unm_test], test['cc'].values[unm_test]] = test['TWS_t'].values[unm_test]

# compute pairwise cross-anchor correlations
print("\n--- Cross-anchor cell-wise correlations (strict, fully-unmasked only) ---")
pairs = []
for i, a in enumerate(strict_anchors):
    for j, b in enumerate(strict_anchors):
        if j <= i: continue
        k = (b//100 - a//100)*12 + (b%100 - a%100)
        if k <= 0: continue
        xa = F[ym_to_i[a]]
        xb = F[ym_to_i[b]]
        ok = np.isfinite(xa) & np.isfinite(xb)
        if ok.sum() < 100: continue
        r = np.corrcoef(xa[ok], xb[ok])[0,1]
        phi_est = r ** (1.0/k)
        pairs.append((a,b,k,r,phi_est, ok.sum()))
        print(f"  {a} -> {b} (k={k}): r={r:.4f}  -> phi_est={phi_est:.4f}  (n={ok.sum()})")

# aggregate
if pairs:
    rs = np.array([p[3] for p in pairs])
    ks = np.array([p[2] for p in pairs])
    phis = np.array([p[4] for p in pairs])
    # joint fit: log(r) = k * log(phi)  -> log(phi) = sum(k*log(r)) / sum(k^2)
    log_phi = np.sum(ks * np.log(rs)) / np.sum(ks*ks)
    phi_joint = np.exp(log_phi)
    print(f"\nJoint fit (strict anchors): phi = {phi_joint:.4f}")
    print(f"Mean of per-pair phi estimates: {phis.mean():.4f}")

# Now compare with LENIENT anchors (other AI's original criterion, includes partial months)
print("\n--- Lenient anchors (mask<50%, includes partial months) ---")
lenient_pairs = []
for i, a in enumerate(lenient_anchors):
    for j, b in enumerate(lenient_anchors):
        if j <= i: continue
        k = (b//100 - a//100)*12 + (b%100 - a%100)
        if k <= 0: continue
        xa = F[ym_to_i[a]]
        xb = F[ym_to_i[b]]
        ok = np.isfinite(xa) & np.isfinite(xb)
        if ok.sum() < 100: continue
        r = np.corrcoef(xa[ok], xb[ok])[0,1]
        phi_est = r ** (1.0/k)
        lenient_pairs.append((a,b,k,r,phi_est, ok.sum()))

if lenient_pairs:
    rs = np.array([p[3] for p in lenient_pairs])
    ks = np.array([p[2] for p in lenient_pairs])
    log_phi = np.sum(ks * np.log(rs)) / np.sum(ks*ks)
    phi_joint_lenient = np.exp(log_phi)
    print(f"Joint fit (lenient anchors): phi = {phi_joint_lenient:.4f}")
    print(f"  n_pairs={len(lenient_pairs)}, range of r: [{rs.min():.3f}, {rs.max():.3f}]")

# Also: recompute lambda (signal fraction) from strict pairs
# var(TWS) = signal_var + noise_var; cov(TWS_t, TWS_{t+k}) = signal_var * phi^k
# lambda = cov(TWS_t, TWS_{t+1}) / var(TWS_t) -- but only valid for k=1
# From strict anchors at k=4,5: cov = signal_var * phi^k, so lambda = cov_at_k0 / var
# Need a k=1 pair ideally. Check if any strict anchors are 1 month apart.

print("\n--- Check for k=1 strict pairs ---")
for i, a in enumerate(strict_anchors):
    for j, b in enumerate(strict_anchors):
        if j <= i: continue
        k = (b//100 - a//100)*12 + (b%100 - a%100)
        if k == 1:
            print(f"  {a} -> {b} (k=1)!")

# Seasonality check
print("\n--- Seasonality check (monthly mean across full train period) ---")
train['month'] = train['time'].dt.month
monthly_means = train.groupby('month')['TWS_t'].mean()
print(f"Monthly means range: {monthly_means.min():.4f} to {monthly_means.max():.4f}")
print(f"Range: {monthly_means.max() - monthly_means.min():.4f}")
print(f"Std: {monthly_means.std():.4f}")
print(f"Global TWS_t mean: {train['TWS_t'].mean():.4f}, std: {train['TWS_t'].std():.4f}")

# Save summary
with open('/home/z/my-project/download/phi_recalc.txt', 'w') as f:
    f.write("PHI RECOMPUTATION SUMMARY\n")
    f.write("="*50 + "\n\n")
    f.write(f"Strict anchors (fully unmasked, <1% masked): {strict_anchors}\n")
    f.write(f"Count: {len(strict_anchors)}\n\n")
    f.write("Strict cross-anchor pairs:\n")
    for p in pairs:
        f.write(f"  {p[0]} -> {p[1]} (k={p[2]}): r={p[3]:.4f} -> phi={p[4]:.4f}\n")
    if pairs:
        f.write(f"\nJoint fit (strict): phi = {phi_joint:.4f}\n")
    f.write(f"\nLenient anchors (mask<50%): {lenient_anchors}\n")
    f.write(f"Count: {len(lenient_anchors)}\n")
    if lenient_pairs:
        f.write(f"Joint fit (lenient): phi = {phi_joint_lenient:.4f}\n")
    f.write(f"\nSeasonality: monthly mean range = {monthly_means.max()-monthly_means.min():.4f}\n")
print("\nSaved summary to download/phi_recalc.txt")
