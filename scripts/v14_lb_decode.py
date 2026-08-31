"""v14 LB decode: 0.694044494 — implied public k0 class RMSE + val->public transfer ratio."""
import math

# Board
v10b = 0.699997215
v12b = 0.695357171   # prev best
v13  = 0.696325144
v14  = 0.694044494   # NEW BEST

# Public class structure (established Task 18/19):
# public rows: k0 = 43.0%, masked = 57.0% (h2 31,151 / h3 31,132 — 50/50)
# masked rows are BIT-IDENTICAL across v10b/v12b/v14 (all static Dtil) ->
# the ENTIRE v12b->v14 public delta sits on k0 rows.
w_k0, w_msk = 0.43, 0.57
sigma_msk = 0.767  # established masked-class estimate (Task 19)

mse = lambda s: s * s
msk_mse_contrib = w_msk * mse(sigma_msk)

def k0_rmse(pub):
    return math.sqrt((mse(pub) - msk_mse_contrib) / w_k0)

k0_v12b = k0_rmse(v12b)
k0_v14  = k0_rmse(v14)

print("=== v14 LB decode ===")
print(f"v14 = {v14}  NEW BEST (delta vs v12b = {v14 - v12b:+.7f})")
print(f"delta vs v10b = {v14 - v10b:+.7f}")
print()
print(f"total public MSE: v12b {mse(v12b):.6f} -> v14 {mse(v14):.6f}  (dMSE = {mse(v14)-mse(v12b):+.6f})")
print(f"implied public k0 RMSE: v12b {k0_v12b:.4f} -> v14 {k0_v14:.4f}  (class delta = {k0_v14-k0_v12b:+.4f})")
print()
# Pre-registered band
print("=== vs pre-registered band ===")
print(f"pre-registered v14 public: 0.6905-0.6925 (center 0.6910, from k0-analog val 0.5753)")
print(f"actual 0.694044494 -> landed ABOVE band by {v14 - 0.6925:+.4f}")
val_k0_gain = 0.5876 - 0.5753   # val class gain of new k0 recipe (shipped stack 0.5876 -> blend 0.5753)
pub_k0_gain = k0_v14 - k0_v12b
print(f"val k0 class gain: {val_k0_gain:+.4f} | public k0 class gain: {pub_k0_gain:+.4f}")
print(f"transfer ratio = {pub_k0_gain/val_k0_gain:.2f}  (~{abs(pub_k0_gain/val_k0_gain)*100:.0f}% of val promise)")
print()
# Rule evaluation
print("=== pre-registered rule (Task 19) ===")
print("rule: <=0.6930 -> build v15 carrier (v14 + h>=4 gate); >=0.6954 -> revert k0 to v12b recipe")
print(f"actual 0.694044494 -> MIDDLE ZONE (0.6930 < x < 0.6954):")
print("  - k0-new recipe (0.6*lgb + 0.4*kalman) is REAL (+ board gain) but transfers at ~30% -> KEEP it")
print("  - era h>=4 gate remains untested on public (all h>=4 rows are private) -> still the private payload")
print("  -> v15 = v14 k0 + h>=4 era gate is STILL the right carrier build (dominates v13b on private k0)")
print()
# Forward calibration
print("=== recalibrated transfer discount ===")
print("val k0-lane class gains should be discounted ~3x when predicting public:")
for gain in [0.005, 0.010, 0.015]:
    pub_cls = -abs(gain) * 0.30
    pub_total = math.sqrt(mse(v14) - w_k0*mse(k0_v14) + w_k0*mse(k0_v14 + pub_cls))
    print(f"  val class gain -{gain:.3f} -> expect public ~{pub_total:.4f}")
print()
# k0 lane headroom to top-10
print("=== k0 lane headroom (public) ===")
for target_k0 in [0.55, 0.505, 0.46]:
    pub = math.sqrt(w_k0*mse(target_k0) + msk_mse_contrib)
    print(f"  k0 {k0_v14:.4f} -> {target_k0:.3f}: public {pub:.4f}")
print(f"  (top-10 ~0.667, MOHAR 0.5596; v14 rank ~ mid-40s est.)")
