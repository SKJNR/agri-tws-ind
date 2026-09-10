#!/usr/bin/env python3
"""
TASK 19b: ENSO/ONI PROBE — the last untested legal external covariate family.

Question (user): "are you sure nothing was missed in the clean lane?"
E1 tested IN-FILE covariates (SPEI_12 etc.) for D-evolution -> dead (R 0.19-0.27).
External climate indices (ONI = Nino3.4 SST anomaly) are LEGAL (not GRACE/TWS)
and were never tested. The data is synthetic-with-real-GRACE-anchored D, so a
real ENSO teleconnection COULD in principle live inside D's drift.

Tests (mirroring E1's protocol):
  A: test-anchor pairs — dD(c) ~ g(c) * dONI, LOO-by-pair (n=5 pairs)
  C: train 24-month anomaly change vs mean ONI over window (high power)
  S: sanity — global-mean train anomaly vs ONI (any signal at all?)

Pre-registered bar: pooled LOO R >= 0.5 => viable lever (open v22 D-channel).
R < 0.3 => dead, same verdict family as E1.
"""
import numpy as np
import pandas as pd

DATA = '/home/z/my-project/data'

# ---------- load ONI ----------
rows = []
with open(f'{DATA}/oni.data') as f:
    lines = [l.strip() for l in f if l.strip()]
for l in lines[1:-1]:  # skip header and trailing nan-line
    parts = l.split()
    try:
        yr = int(parts[0])
    except ValueError:
        continue
    if yr < 1950 or yr > 2026:
        continue
    for i, v in enumerate(parts[1:13]):
        try:
            rows.append((yr * 12 + i, float(v)))
        except ValueError:
            pass
oni = pd.DataFrame(rows, columns=['t_abs', 'oni']).set_index('t_abs').sort_index()
print(f"ONI loaded: {len(oni)} months, {oni.index.min()}..{oni.index.max()}, "
      f"std={oni['oni'].std():.2f}")

# ---------- load competition data ----------
test = pd.read_csv(f'{DATA}/Test (2).csv')
test.columns = [c.strip() for c in test.columns]
test['t_abs'] = test['time'].str[:4].astype(int) * 12 + test['time'].str[5:7].astype(int) - 1
tw_col = 'TWS_t' if 'TWS_t' in test.columns else 'TWS_t_masked'
test['masked'] = test[tw_col].isna()

train = pd.read_csv(f'{DATA}/Train (1).csv')
train['t_abs'] = train['time'].str[:4].astype(int) * 12 + train['time'].str[5:7].astype(int) - 1

def ckey(lat, lon):
    return np.round(lat * 10).astype(int) * 10000 + np.round(lon * 10).astype(int)
for df in (train, test):
    df['cc'] = ckey(df['lat'], df['lon'])
base = train.groupby('cc').agg(mu=('TWS_t', 'mean')).reset_index()

# ================= TEST A: anchor pairs, dD ~ g(c)*dONI =================
gm = test.groupby('t_abs').agg(n=('ID', 'count'), n_masked=('masked', 'sum'))
gm['vis'] = gm['n'] - gm['n_masked']
anchor_ts = sorted(gm[gm['vis'] > 1000].index.tolist())
print(f"\nTest anchors: {[f'{t//12}-{t%12+1:02d}' for t in anchor_ts]}")

test_v = test[~test['masked']].merge(base, on='cc', how='left')
test_v['anom'] = test_v['TWS_t'] - test_v['mu']
anchor_fields = {t: test_v[test_v['t_abs'] == t].set_index('cc')['anom'] for t in anchor_ts}

pairs = [(anchor_ts[i], anchor_ts[i + 1]) for i in range(len(anchor_ts) - 1)]
dD, dONI = {}, {}
for i, k in pairs:
    dd = (anchor_fields[k] - anchor_fields[i]).dropna()
    dD[(i, k)] = dd
    oni_i = oni['oni'].reindex(range(i, k + 1)).mean()
    oni_k = oni['oni'].reindex(range(i, k + 1)).mean()  # same window
    # dONI = mean ONI over (i,k] minus ONI at i (state change proxy)
    oni_at_i = oni['oni'].get(i, np.nan)
    dONI[(i, k)] = oni_k - oni_at_i
    print(f"pair {i//12}-{i%12+1:02d}->{k//12}-{k%12+1:02d}: dONI={dONI[(i,k)]:+.2f}")

# LOO-by-pair: fit g(c) on 4 pairs, predict 5th
print("\n--- TEST A: LOO-by-pair dD ~ g(c)*dONI ---")
y_true, y_pred = [], []
for j, (i, k) in enumerate(pairs):
    train_pairs = [p for p in pairs if p != (i, k)]
    cc_common = dD[(i, k)].index
    num, den = pd.Series(0.0, index=cc_common), pd.Series(0.0, index=cc_common)
    for (pi, pk) in train_pairs:
        dd = dD[(pi, pk)].reindex(cc_common).fillna(0.0)
        num += dd * dONI[(pi, pk)]
        den += dONI[(pi, pk)] ** 2
    g = num / den.clip(lower=1e-9)
    pred = g * dONI[(i, k)]
    y_true.append(dD[(i, k)].values)
    y_pred.append(pred.values)
y_true = np.concatenate(y_true); y_pred = np.concatenate(y_pred)
rA = np.corrcoef(y_true, y_pred)[0, 1]
print(f"LOO-by-pair R = {rA:+.3f}   (bar: 0.5 viable / <0.3 dead; E1 SPEI_12 got 0.271)")

# global-mean version (scalar-scalar, n=5)
gdD = [float(dD[p].mean()) for p in pairs]
gdONI = [dONI[p] for p in pairs]
r_glob = np.corrcoef(gdD, gdONI)[0, 1]
print(f"global-mean corr(dD, dONI) over {len(pairs)} pairs = {r_glob:+.3f}")

# ================= TEST C: train 24m change vs mean ONI =================
print("\n--- TEST C: train 24-month anomaly change vs mean ONI (high power) ---")
t_min, t_max = train['t_abs'].min(), train['t_abs'].max()
piv_tws = train.pivot_table(index='cc', columns='t_abs', values='TWS_t')
rng = np.random.default_rng(42)
cells = piv_tws.index.values
sample = rng.choice(len(cells), size=min(4000, len(cells)), replace=False)
gap = 24
tgt, cov = [], []
s = t_min
while s + 2 * gap <= t_max:
    e = s + gap
    c1 = [t for t in range(s, e + 1) if t in piv_tws.columns]
    c2 = [t for t in range(e + 1, e + gap + 1) if t in piv_tws.columns]
    if len(c1) >= gap * 0.8 and len(c2) >= gap * 0.8:
        a1 = piv_tws.iloc[sample].loc[:, c1].mean(axis=1).values
        a2 = piv_tws.iloc[sample].loc[:, c2].mean(axis=1).values
        oni_w = oni['oni'].reindex(range(e + 1, e + gap + 1)).mean()
        if np.isfinite(oni_w):
            ok = np.isfinite(a1) & np.isfinite(a2)
            tgt.append(a2[ok] - a1[ok])
            cov.append(np.full(ok.sum(), oni_w))
    s += 12
tgt = np.concatenate(tgt); cov = np.concatenate(cov)
rC = np.corrcoef(tgt, cov)[0, 1]
print(f"n pairs: {len(tgt):,}   corr(24m anomaly change, mean ONI) = {rC:+.3f}")
print(f"(E1's SPEI_12 analog measured +0.195)")

# ================= TEST S: sanity, global mean anomaly vs ONI =================
print("\n--- TEST S: sanity — global-mean monthly train anomaly vs ONI ---")
gmean = train.groupby('t_abs')['TWS_t'].mean()
j = pd.concat([gmean.rename('gm'), oni['oni']], axis=1).dropna()
rS = np.corrcoef(j['gm'], j['oni'])[0, 1]
print(f"n months: {len(j)}   corr(global mean TWS, ONI) = {rS:+.3f}")
for lag in (0, 3, 6, 12):
    jj = pd.concat([gmean.rename('gm'), oni['oni'].shift(lag)], axis=1).dropna()
    print(f"  lag {lag:>2}m: corr = {np.corrcoef(jj['gm'], jj['oni'])[0,1]:+.3f}")

print("\n===== VERDICT (pre-registered) =====")
best = max(abs(rA), abs(rC))
if best >= 0.5:
    print(f"VIABLE (R={best:.2f} >= 0.5): open the v22 ENSO-D channel.")
elif best < 0.3:
    print(f"DEAD (R={best:.2f} < 0.3): same verdict family as E1. The D-evolution")
    print("channel has no legal driver: in-file covs (E1) and external climate")
    print("indices (this probe) both fail the 0.5 bar. Clean-lane ledger closes.")
else:
    print(f"AMBIGUOUS (R={best:.2f}): below the 0.5 bar — not actionable.")
