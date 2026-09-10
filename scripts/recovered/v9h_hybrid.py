"""H1/H2 HYBRID PROBES — merge-decomposition of the two record lines.
Requires: download/submission_v6c.csv (the 0.7030 record file, built outside this env)
          download/submission_v8a.csv (our 0.7070 file, already on disk)
Produces:
  probe_H1 = v6c on MASKED rows + v8a on k0 rows   (which line owns masked rows?)
  probe_H2 = v8a on MASKED rows + v6c on k0 rows   (which line owns k0 rows?)
Read-out (with v6c=0.7030, v8a=0.7070, noise floor ~0.002):
  H1 < 0.699          -> v8a's masked rows are better; merge lane = v8a-masked + best k0
  H2 < 0.699          -> v6c's masked rows are better; merge lane = v6c-masked + best k0
  both ~0.703-0.707   -> the two lines' gains OVERLAP; merging adds little
Also prints corr(v6c, v8a) by row class -> overlap diagnostic.
"""
import numpy as np, pandas as pd, os

DATA = '/home/z/my-project/data'
DL = '/home/z/my-project/download'
P6 = f'{DL}/submission_v6c.csv'
P8 = f'{DL}/submission_v8a.csv'

if not os.path.exists(P6):
    print("ERROR: download/submission_v6c.csv not found.")
    print("Ask the collaborator for the exact v6c submission CSV (the 0.7030 record file),")
    print("drop it at that path, then re-run:  python scripts/v9h_hybrid.py")
    raise SystemExit(1)

sub = pd.read_csv(f'{DATA}/SampleSubmission (4).csv')
test = pd.read_csv(f'{DATA}/Test (2).csv', usecols=['TWS_t_masked'])
masked = test['TWS_t_masked'].astype(bool).values

a6 = pd.read_csv(P6); a8 = pd.read_csv(P8)
assert len(a6) == len(a8) == 280961
assert (a6['ID'].values == sub['ID'].values).all()
assert (a8['ID'].values == sub['ID'].values).all()
v6 = a6['Target'].values.astype(np.float64)
v8 = a8['Target'].values.astype(np.float64)

print(f"corr(v6c, v8a) overall = {np.corrcoef(v6, v8)[0,1]:.4f}")
print(f"corr on MASKED rows    = {np.corrcoef(v6[masked], v8[masked])[0,1]:.4f}")
print(f"corr on k0 rows        = {np.corrcoef(v6[~masked], v8[~masked])[0,1]:.4f}")
print(f"mean|v6c-v8a| masked   = {np.abs(v6[masked]-v8[masked]).mean():.4f}")
print(f"mean|v6c-v8a| k0       = {np.abs(v6[~masked]-v8[~masked]).mean():.4f}")

def save(p, tag):
    out = sub[['ID']].copy()
    out['Target'] = p.astype(np.float32)
    out.to_csv(f'{DL}/probe_{tag}.csv', index=False)
    assert out['Target'].notna().all()
    print(f"saved probe_{tag}.csv (mean={p.mean():.4f} std={p.std():.4f})")

h1 = v8.copy(); h1[masked] = v6[masked]   # H1 = v6c masked + v8a k0
save(h1, 'H1')
h2 = v8.copy(); h2[~masked] = v6[~masked] # H2 = v8a masked + v6c k0
save(h2, 'H2')
print("\nSubmit H1 + H2 (2 slots). Read-out table in the script docstring.")
