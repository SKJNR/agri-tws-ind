"""a15_1b.py — refine the 1-step factor predictor on the val 18-pair protocol.

Questions:
 1. Oracle per-mode linear (in-sample): cap of pred = a*y + r family.
 2. Dtil variants: pure trendex  vs  v4-style combo (w1*Dhat_val + w2*S + w3*trendex).
 3. VarD scale sweep; H shrinkage sweep for the cov(t+1) update.
 4. Per-year error breakdown (ramp growth).
 5. Val-era innovation std estimate.
"""
import sys, time
import numpy as np
sys.path.insert(0, '/home/z/my-project/scripts')
from a15_common import load_train, FactorCore, ModeKF, estimate_VarD, COVS

t0 = time.time()
OUT = '/home/z/my-project/download/a15_1b_refine.txt'
log = []
def P(s=''):
    print(s, flush=True)
    log.append(str(s))

train = load_train()
n_cells = int(train['cc'].max()) + 1
fit = train[train['time'].dt.year <= 2012]
val = train[(train['time'].dt.year >= 2013) & (train['time'].dt.year <= 2015)]

core = FactorCore(fit, n_cells, K=100, rec_edges=(2010, 2013), verbose=False)
V, Vx, phi, q = core.V, core.Vx, core.phi, core.q
full, nfull = core.full, core.nfull

val_months = np.sort(val['t_abs'].unique())
vm2i = {int(m): i for i, m in enumerate(val_months)}
Tval = len(val_months)
Xv = np.full((Tval, n_cells), np.nan, dtype=np.float32)
mi = val['t_abs'].map(vm2i).values
Xv[mi, val['cc'].values] = val['TWS_t'].values
vmonths, vfields = core.era_cov(val)
vz = core.zscores(vfields, vmonths)
# v4-style S (combined cov-reg field, era mean)
S_val = np.nanmean(np.array([vfields[int(m)][0] for m in vmonths]), axis=0)

# val anchor anomaly fields (unmasked months; val anchors = cal in ANCHOR_CAL)
ANCHOR_CAL = {1, 6, 7, 9, 11, 12}
val_anchor_months = [int(m) for m in val_months if (m % 12 + 1) in ANCHOR_CAL]
AF = {}
for a in val_anchor_months:
    i = vm2i[a]
    f = np.full(n_cells, np.nan, dtype=np.float32)
    f[val['cc'].values[val['t_abs'].values == a]] = val['TWS_t'].values[val['t_abs'].values == a]
    AF[a] = f - core.mu_c
Dhat = np.nanmean(np.array([AF[a] for a in val_anchor_months]), axis=0)

# LOO-fit Dtil weights on val anchors (cv_lb protocol)
Xs, ys = [], []
for a in val_anchor_months:
    dloo = np.nanmean(np.array([AF[b] for b in val_anchor_months if b != a]), axis=0)
    tx = core.trendex(a)
    ok = np.isfinite(dloo) & np.isfinite(S_val) & np.isfinite(AF[a]) & np.isfinite(tx)
    Xs.append(np.column_stack([dloo[ok], S_val[ok], tx[ok]]))
    ys.append(AF[a][ok])
w_ = np.linalg.solve(np.vstack(Xs).T @ np.vstack(Xs) + 1e-3 * np.eye(3),
                     np.vstack(Xs).T @ np.concatenate(ys))
W1, W2, W3 = float(w_[0]), float(w_[1]), float(w_[2])
P(f"Dtil-val LOO weights: W1={W1:.3f} W2={W2:.3f} W3={W3:.3f}   [cv_lb ref 0.786/0.218/0.049]")

def Dtil_combo(t):
    return W1 * Dhat + W2 * S_val + W3 * core.trendex(t)

# anomaly fields per Dtil variant
TDv = val_months[:, None].astype(np.float64) - core.tbar_c[None, :]
Av_raw = Xv.astype(np.float64) - core.mu_c[None, :]          # TWS - mu (no trend)
def anomalies(Dfun):
    G = np.zeros((Tval, n_cells))
    for i, m in enumerate(val_months):
        G[i] = Av_raw[i] - Dfun(int(m))
    return G

ev = np.where(np.diff(val_months) == 1)[0]
tau, nxt = ev, ev + 1
y_true_full = Av_raw[nxt]                                     # TWS - mu at t+1
varT = np.nanvar(y_true_full[:, full])
P(f"eval pairs={len(ev)} target (TWS-mu) std={np.sqrt(varT):.4f}")

def eval_pred_unused():
    pass

# we need r for ALL cells: r = G - V_ext @ f  (V_ext: (n_cells,K))
def r_fields(G, f):
    return G - f @ core.V_ext.T

Dfun_ = None
def run(tag, Dfun, mode, **kw):
    global Dfun_
    Dfun_ = Dfun
    G = anomalies(Dfun)
    gv = G[:, full]
    gv = np.where(np.isfinite(gv), gv, 0.0)
    f = gv @ V
    r = np.where(np.isfinite(G), G, 0.0) - f @ core.V_ext.T
    YA = np.stack([V.T @ gv[vm2i[a]] for a in val_anchor_months])
    gaps = [(i, i + 1, val_anchor_months[i + 1] - val_anchor_months[i])
            for i in range(len(val_anchor_months) - 1)]
    VarD = estimate_VarD(YA, gaps, phi, Vx) * kw.get('VarD_scale', 1.0)
    kf = ModeKF(phi, Vx, q, core.H * kw.get('H_scale', 1.0), core.Rcov, VarD)
    preds = np.zeros((len(ev), core.K))
    for e in range(len(ev)):
        kf.reset()
        kf.anchor_update(f[tau[e]])
        kf.step(1)
        m1 = int(val_months[nxt[e]])
        if mode == 'cov' and m1 in vz:
            kf.cov_update(vz[m1])
        preds[e] = kf.mean_var()[0]
    Dn = np.stack([Dfun(int(val_months[nxt[e]])) for e in range(len(ev))])
    pred_full = Dn[:, full] + preds @ V.T + r[tau][:, full]
    d = pred_full - y_true_full[:, full]
    out = float(np.sqrt(np.nanmean(d ** 2)))
    yrs = val_months[nxt] // 12
    extra = ''
    if kw.get('per_year'):
        extra = ' | ' + ' '.join(f"{yr}:{np.sqrt(np.nanmean(d[yrs == yr] ** 2)):.3f}" for yr in sorted(set(yrs.tolist())))
    P(f"{tag:44s} RMSE={out:.4f}{extra}")
    return out, f, r, gv

# persistence reference in this space (TWS-mu): pred = A(t)-mu
pers = np.sqrt(np.nanmean((Av_raw[tau][:, full] - y_true_full[:, full]) ** 2))
P(f"persistence (TWS-mu):                        RMSE={pers:.4f}")

P("\n--- Dtil = pure trendex ---")
run("trendex / no cov", core.trendex, 'nocov')
run("trendex / cov(t+1)", core.trendex, 'cov')
P("\n--- Dtil = combo (LOO weights) ---")
run("combo / no cov", Dtil_combo, 'nocov', per_year=True)
run("combo / cov(t+1)", Dtil_combo, 'cov')

P("\n--- VarD scale sweep (combo, cov) ---")
for sc in [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]:
    run(f"combo / cov / VarDx{sc}", Dtil_combo, 'cov', VarD_scale=sc)

P("\n--- H scale sweep (combo, VarDx1) ---")
for hs in [0.0, 0.5, 0.75, 1.0]:
    run(f"combo / cov / Hx{hs}", Dtil_combo, 'cov', H_scale=hs)

P("\n--- oracle per-mode linear (in-sample, combo) ---")
G = anomalies(Dtil_combo)
gv = np.where(np.isfinite(G[:, full]), G[:, full], 0.0)
f = gv @ V
r = np.where(np.isfinite(G), G, 0.0) - f @ core.V_ext.T
num = (f[tau] * f[nxt]).sum(0)
den = (f[tau] ** 2).sum(0)
a_or = num / np.maximum(den, 1e-9)
Dn = np.stack([Dtil_combo(int(val_months[nxt[e]])) for e in range(len(ev))])
pred_full = Dn[:, full] + (f[tau] * a_or[None, :]) @ V.T + r[tau][:, full]
P(f"oracle per-mode a* (in-sample):              RMSE={np.sqrt(np.nanmean((pred_full - y_true_full[:, full]) ** 2)):.4f}")
P(f"oracle a* top10={np.round(a_or[:10], 3)}")

# val innovation std estimate: resid of oracle prediction in cell space
d = pred_full - y_true_full[:, full]
P(f"empirical 1-step resid std (oracle, in-sample): {np.sqrt(np.nanmean(d ** 2)):.4f}")

with open(OUT, 'w') as fh:
    fh.write('\n'.join(log) + '\n')
P(f"\nsaved {OUT} [{time.time()-t0:.0f}s]")
