"""
GATE REVIEW v18 — file-level checks + pre-registration projection recompute.
New file (gate review). Does NOT modify any existing file.
"""
import numpy as np, pandas as pd, hashlib
DATA = '/home/z/my-project/data'; DL = '/home/z/my-project/download'
OUT = []
def P(s=''):
    print(s, flush=True); OUT.append(str(s))

sub = pd.read_csv(f'{DATA}/SampleSubmission (4).csv', dtype={'ID': str})
test = pd.read_csv(f'{DATA}/Test (2).csv', dtype={'ID': str})
assert (sub['ID'].values == test['ID'].values).all(), "SampleSubmission vs Test ID order"
t = pd.to_datetime(test['time'])
ym = t.dt.year*100 + t.dt.month
t_abs = (ym//100)*12 + (ym%100) - 1
masked = test['TWS_t_masked'].astype(bool).values
P(f"rows={len(test):,}  masked={masked.sum():,}  unmasked={(~masked).sum():,}")
P(f"LB weights: masked {masked.sum()/len(test):.5f}  k0 {(~masked).sum()/len(test):.5f}")

RUNS = {'run1 Feb16-Mar16': [24193, 24194], 'run2 Jul16-Sep16': [24198, 24199, 24200],
        'run3 Jan17-Jun17 (2017 block)': [24204, 24205, 24206, 24207, 24208, 24209],
        'run4 Dec18 (fwd-only)': [24227]}

files = {}
for tag in ['v18a', 'v18b', 'v17b']:
    df = pd.read_csv(f'{DL}/submission_{tag}.csv', dtype={'ID': str})
    files[tag] = df
    md5 = hashlib.md5(open(f'{DL}/submission_{tag}.csv', 'rb').read()).hexdigest()
    P(f"\n=== {tag} (md5 {md5}) ===")
    P(f"  ID bit-exact vs SampleSubmission (set+order): {bool((df['ID'].values == sub['ID'].values).all())}")
    v = df['Target'].values.astype(np.float64)
    P(f"  n={len(v):,}  NaN={int(np.isnan(v).sum())}  inf={int(np.isinf(v).sum())}  "
      f"min={v.min():.4f} max={v.max():.4f} mean={v.mean():.4f} std={v.std():.4f}")
    P(f"  masked rows: mean={v[masked].mean():.4f} std={v[masked].std():.4f} "
      f"p1/p50/p99={np.percentile(v[masked],[1,50,99]).round(4).tolist()}")
    P(f"  k0 rows    : mean={v[~masked].mean():.4f} std={v[~masked].std():.4f} "
      f"p1/p50/p99={np.percentile(v[~masked],[1,50,99]).round(4).tolist()}")
    for rname, months in RUNS.items():
        s = masked & np.isin(t_abs, months)
        P(f"  {rname:34s}: n={s.sum():>7,}  std={v[s].std():.4f}  mean={v[s].mean():+.4f}")

# v18 vs v17b change magnitude on masked rows
a = files['v18a']['Target'].values.astype(np.float64); b = files['v17b']['Target'].values.astype(np.float64)
bb = files['v18b']['Target'].values.astype(np.float64)
d = a[masked] - b[masked]; d2 = bb[masked] - b[masked]
P(f"\nmasked-row change vs v17b: v18a mean|d|={np.abs(d).mean():.4f} max|d|={np.abs(d).max():.4f} "
  f"frac|d|>0.1={np.mean(np.abs(d)>0.1):.3f} corr={np.corrcoef(a[masked], b[masked])[0,1]:.4f}")
P(f"masked-row change vs v17b: v18b mean|d|={np.abs(d2).mean():.4f} max|d|={np.abs(d2).max():.4f} corr={np.corrcoef(bb[masked], b[masked])[0,1]:.4f}")
for rname, months in RUNS.items():
    s = masked & np.isin(t_abs, months)
    P(f"  {rname:34s}: v18a-vs-v17b RMSE-of-change={np.sqrt(np.mean((a[s]-b[s])**2)):.4f}")

# ---------------- G: pre-registration recompute ----------------
P("\n================ PROJECTION RECOMPUTE ================")
LB17 = 0.704955918; M2_17 = 0.6905; M2_18 = 0.6694; M2_18b = 0.6688
WM, WK = 0.6652, 0.3348
P(f"v17b: implied masked RMSE = sqrt((LB^2 - {WK}*k0^2)/{WM})")
for k0 in [0.625, 0.640, 0.650]:
    m = np.sqrt((LB17**2 - WK*k0**2)/WM)
    P(f"  k0={k0:.3f}: masked={m:.4f}  gap vs M2 {M2_17:.4f} = {m-M2_17:+.4f}")
m_c = np.sqrt((LB17**2 - WK*0.640**2)/WM)
gap_c = m_c - M2_17
P(f"central gap (k0=0.640): {gap_c:+.4f}  [build log claims +0.0450]")
P(f"\nv18a projection: masked = M2_ens {M2_18:.4f} + gap band [+0.030, +0.050] = "
  f"[{M2_18+0.030:.4f}, {M2_18+0.050:.4f}]")
P("LB = sqrt(0.6652*masked^2 + 0.3348*k0^2):")
P(f"{'masked':>8s} | " + " | ".join(f"k0={k:.3f}" for k in [0.630, 0.640, 0.650]))
for mk in [M2_18+0.030, M2_18+0.040, M2_18+0.050, 0.698, 0.708, 0.719]:
    row = [np.sqrt(WM*mk**2 + WK*k**2) for k in [0.630, 0.640, 0.650]]
    P(f"{mk:8.4f} | " + " | ".join(f"{r:.4f}" for r in row))
P(f"reference: v12b=0.695357  v17b=0.704956")
worst = np.sqrt(WM*(M2_18+0.050)**2 + WK*0.650**2); best = np.sqrt(WM*(M2_18+0.030)**2 + WK*0.630**2)
P(f"joint best (gap+0.030,k0.63)={best:.4f} | joint worst (gap+0.050,k0.65)={worst:.4f} "
  f"-> worst {'BEATS' if worst < 0.695357 else 'DOES NOT BEAT'} v12b 0.6954")
P(f"pre-registered table check: masked 0.698->LB {np.sqrt(WM*0.698**2+WK*0.640**2):.4f} (log says 0.6791) | "
  f"0.708->{np.sqrt(WM*0.708**2+WK*0.640**2):.4f} (0.6860) | 0.719->{np.sqrt(WM*0.719**2+WK*0.640**2):.4f} (0.6936)")

# ---------------- E: public LB sampling noise ----------------
P("\n================ PUBLIC-LB SAMPLING NOISE (random 30/70 by row) ================")
n_pub_m = 0.30*masked.sum(); n_pub_k = 0.30*(~masked).sum()
for m_rmse, k_rmse, tag in [(0.710, 0.640, 'central'), (M2_18+0.045, 0.640, 'gap+0.045')]:
    se_m = m_rmse/np.sqrt(2*n_pub_m); se_k = k_rmse/np.sqrt(2*n_pub_k)
    lb = np.sqrt(WM*m_rmse**2 + WK*k_rmse**2)
    se_lb = np.sqrt((WM*m_rmse*se_m)**2 + (WK*k_rmse*se_k)**2)/lb
    P(f"{tag}: public rows ~{n_pub_m:,.0f} masked + {n_pub_k:,.0f} k0 -> SE(masked)={se_m:.4f} "
      f"SE(k0)={se_k:.4f} SE(LB)={se_lb:.4f}")

with open('/home/z/my-project/scripts/gate_files_out.txt', 'w') as f:
    f.write('\n'.join(OUT))
P("\nGATE FILES DONE")
