"""Verify v11a/v11b CSVs and diff vs v10b (per class + public window)."""
import numpy as np, pandas as pd

DATA = '/home/z/my-project/data'
DL = '/home/z/my-project/download'

sub = pd.read_csv(f'{DATA}/SampleSubmission (4).csv')
test = pd.read_csv(f'{DATA}/Test (2).csv', usecols=['time','lat','lon','TWS_t_masked'])
test['ym'] = pd.to_datetime(test['time']).dt.year*100 + pd.to_datetime(test['time']).dt.month
test['t_abs'] = (test['ym']//100)*12 + (test['ym']%100) - 1
msk = test['TWS_t_masked'].astype(bool).values
ta = test['t_abs'].values
test_months = np.array(sorted(test['t_abs'].unique()))
pub = ta < (int(test_months.min()) + 12)   # first 7 test months = public window
print(f"public rows: {pub.sum()} ({pub.mean():.3f})  k0-share in public: {(~msk[pub]).mean():.3f}")

base = pd.read_csv(f'{DL}/submission_v10b.csv')['Target'].values
for tag in ['v11a', 'v11b']:
    o = pd.read_csv(f'{DL}/submission_{tag}.csv')
    assert len(o) == 280961, tag
    assert (o['ID'].values == sub['ID'].values).all(), tag
    assert o['Target'].notna().all() and np.isfinite(o['Target']).all(), tag
    d = o['Target'].values - base
    print(f"{tag}: corr vs v10b={np.corrcoef(o['Target'].values, base)[0,1]:.5f} "
          f"mean|d|={np.abs(d).mean():.4f} | masked|d|={np.abs(d[msk]).mean():.4f} "
          f"k0|d|={np.abs(d[~msk]).mean():.4f} | PUBLIC mean|d|={np.abs(d[pub]).mean():.4f} "
          f"| PRIVATE mean|d|={np.abs(d[~pub]).mean():.4f} | range [{o['Target'].min():.3f}, {o['Target'].max():.3f}]")
print("ALL CHECKS PASSED")
