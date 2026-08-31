"""
GATE REVIEW v18 — extra quantifications: mirror realism (obs/cell), M1-vs-M2
prediction similarity on shared rows (protocol non-independence).
"""
import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
DATA = '/home/z/my-project/data'
COVS = ['SPEI_01_t','SPEI_03_t','SPEI_06_t','SPEI_12_t','SOIL_MOISTURE_t']
OUT = []
def P(s=''):
    print(s, flush=True); OUT.append(str(s))

train = pd.read_csv(f'{DATA}/Train (1).csv', usecols=['time','lat','lon','TWS_t','target']+COVS)
train['time'] = pd.to_datetime(train['time'])
for c in ['TWS_t','target']+COVS:
    train[c] = pd.to_numeric(train[c], errors='coerce').astype('float32')
train['cc'] = (train['lat'].round(2).astype(str)+'_'+train['lon'].round(2).astype(str)).astype('category').cat.codes.astype('int32')
train['ym'] = train['time'].dt.year*100 + train['time'].dt.month
train['t_abs'] = (train['ym']//100)*12 + (train['ym']%100) - 1
n_cells = int(train['cc'].max())+1

test = pd.read_csv(f'{DATA}/Test (2).csv')
test['time'] = pd.to_datetime(test['time'])
test['ym'] = test['time'].dt.year*100 + test['time'].dt.month
test['t_abs'] = (test['ym']//100)*12 + (test['ym']%100) - 1
test['masked'] = test['TWS_t_masked'].astype(bool)
obs_cell_test = (~test['masked']).sum()/15715
P(f"test obs/cell (unmasked rows / 15715 cells) = {obs_cell_test:.2f}")

val_all = train[(train['time'].dt.year >= 2013) & (train['time'].dt.year <= 2015)].copy()
val_tws_visible = val_all['TWS_t'].copy()
M1_ANCHORS = [24156, 24160, 24166, 24172, 24183, 24186]
M1_RUNS = {24157:1, 24161:2, 24162:2, 24167:2, 24168:2, 24173:3, 24176:3, 24177:3, 24178:3, 24179:3, 24180:3, 24187:4}
M2_ANCHORS = [24156, 24160, 24166, 24172, 24186]
M2_RUNS = dict(M1_RUNS); M2_RUNS.update({24181:3, 24182:3, 24183:3})
def make_mirror(anchors, runs):
    months = set(anchors) | set(runs)
    mir = val_all[val_all['t_abs'].isin(months)].copy()
    mir['masked'] = mir['t_abs'].isin(runs)
    mir['TWS_t'] = val_tws_visible.loc[mir.index].values
    mir.loc[mir['masked'], 'TWS_t'] = np.nan
    return mir
M1 = make_mirror(M1_ANCHORS, M1_RUNS); M2 = make_mirror(M2_ANCHORS, M2_RUNS)
for nm, mir in [('M1', M1), ('M2', M2)]:
    obs = (~mir['masked']).sum()/15715
    P(f"{nm}: obs/cell = {obs:.2f}  (auditA: M1 5.96, M2 4.96, test 5.98)")

# prediction similarity on shared scored rows (BASE config both protocols)
src = open('/home/z/my-project/scripts/build_v18_final.py').read()
i0 = src.index('def build_infra'); i1 = src.index('# ---------- k0 machinery')
import time
ns = dict(np=np, pd=pd, gc=__import__('gc'), warnings=warnings,
          lgb=__import__('lightgbm'), cKDTree=__import__('scipy.spatial', fromlist=['cKDTree']).cKDTree,
          csr_matrix=__import__('scipy.sparse', fromlist=['csr_matrix']).csr_matrix,
          COVS=COVS, n_cells=n_cells,
          cell_xy=train[['cc','lat','lon']].drop_duplicates('cc').sort_values('cc')[['lat','lon']].values.astype(np.float64),
          log=lambda m: None, P=P, t0=time.time(), time=time)
exec(compile(src[i0:i1], 'src', 'exec'), ns)
fit = train[train['time'].dt.year <= 2012].copy()
infra_cv = ns['build_infra'](fit, 'cv')
BASE = dict(phi=0.80, use_bwd=True, dtil_w=(0.70,0.45,0.073), lam=0.84)
p1 = ns['run_masked_v18'](infra_cv, M1, M1['TWS_t'], **BASE)
p2 = ns['run_masked_v18'](infra_cv, M2, M2['TWS_t'], **BASE)
CFG = dict(phi=0.80, lam=0.80, bwd_max_gap=8, dhat_tau=24, dhat_mode='full', smooth_sigma=2.0)
c1 = ns['run_masked_v18'](infra_cv, M1, M1['TWS_t'], **CFG)
c2 = ns['run_masked_v18'](infra_cv, M2, M2['TWS_t'], **CFG)
key1 = pd.Series(list(zip(M1['cc'].values, M1['t_abs'].values)))
key2 = pd.Series(list(zip(M2['cc'].values, M2['t_abs'].values)))
m1s = M1['masked'].values; m2s = M2['masked'].values
df1 = pd.DataFrame({'k': key1[m1s], 'pb': p1[m1s], 'pc': c1[m1s]})
df2 = pd.DataFrame({'k': key2[m2s], 'pb': p2[m2s], 'pc': c2[m2s]})
mg = df1.merge(df2, on='k', suffixes=('_1', '_2'))
P(f"\nshared scored rows: {len(mg):,}")
P(f"BASE pred corr M1-vs-M2 on shared rows: {mg['pb_1'].corr(mg['pb_2']):.4f}, "
  f"RMSE of difference = {np.sqrt(np.mean((mg['pb_1']-mg['pb_2'])**2)):.4f}")
P(f"cfg0 pred corr M1-vs-M2 on shared rows: {mg['pc_1'].corr(mg['pc_2']):.4f}, "
  f"RMSE of difference = {np.sqrt(np.mean((mg['pc_1']-mg['pc_2'])**2)):.4f}")
y = M1.set_index(pd.Series(list(zip(M1['cc'].values, M1['t_abs'].values))))['target']
mg['y'] = mg['k'].map(y)
r_base = np.sqrt(np.mean((mg['pb_1']-mg['y'])**2)); r_cfg = np.sqrt(np.mean((mg['pc_1']-mg['y'])**2))
P(f"sanity: M1-view RMSE on shared rows BASE={r_base:.4f} cfg0={r_cfg:.4f}")
with open('/home/z/my-project/scripts/gate_extra_out.txt', 'w') as f:
    f.write('\n'.join(OUT))
P("DONE")
