import pandas as pd, numpy as np
TR=pd.read_csv("/home/z/my-project/scripts/15a_TR.csv"); TE=pd.read_csv("/home/z/my-project/scripts/15a_TE.csv")
def corr(a,b):
    m=~np.isnan(a)&~np.isnan(b)
    if m.sum()<50: return np.nan
    return np.corrcoef(a[m],b[m])[0,1]
# head-to-head: per-cell corr(sm4, TWS) vs corr(best given cov, TWS), train era
rows=[]
for key,g in TR.groupby(["lat","lon"]):
    if len(g)<100: continue
    r={}
    for c in ["SPEI_01_t","SPEI_03_t","SPEI_06_t","SPEI_12_t","SOIL_MOISTURE_t","sm3","sm4"]:
        r[c]=corr(g[c].values,g["TWS"].values)
    rows.append({"lat":key[0],"lon":key[1],**r})
H=pd.DataFrame(rows)
best_given=H[["SPEI_01_t","SPEI_03_t","SPEI_06_t","SPEI_12_t","SOIL_MOISTURE_t"]].max(axis=1)
print("head-to-head per-cell corr with TWS (train era, deseason):")
print("  sm4 beats best given cov in %.1f%% of cells" % ((H.sm4>best_given).mean()*100))
print("  sm4 > SPEI_12 in %.1f%%" % ((H.sm4>H.SPEI_12_t).mean()*100))
print("  sm4 > SOIL_MOISTURE in %.1f%%" % ((H.sm4>H.SOIL_MOISTURE_t).mean()*100))
# redundancy sm4 vs given covs
r_sm_soi=[]; r_sm_s12=[]
for key,g in TR.groupby(["lat","lon"]):
    if len(g)<100: continue
    r_sm_soi.append(corr(g["sm4"].values,g["SOIL_MOISTURE_t"].values))
    r_sm_s12.append(corr(g["sm4"].values,g["SPEI_12_t"].values))
print("redundancy: corr(sm4, given SOIL_MOISTURE) median %.3f | corr(sm4, SPEI_12) median %.3f" % (np.nanmedian(r_sm_soi), np.nanmedian(r_sm_s12)))
# test-era coupling per month (how well does sm4 nowcast TWS at each anchor month)
print("\ntest-era monthly nowcast corr (sm4 vs TWS_ds, all cells pooled per month):")
for ym,g in TE.groupby("mi"):
    r=corr(g["sm4"].values,g["TWS"].values); r12=corr(g["SPEI_12_t"].values,g["TWS"].values)
    print(f"  mi={int(ym)}: corr(sm4,TWS)={r:+.3f}  corr(SPEI_12,TWS)={r12:+.3f}")
