"""Interpret the FIRST probe score: probe_m201601_plus = 1.052248929.

Probe (Task 22, build_split_probes.py): v24 + 3.0 on ALL rows of month 2016-01
(n_m = 15,647 of 280,961). Base score s0 = 0.683791578 (v24, exact).

Single-sided equation (e = pred - truth on public rows of that month):
    s_plus^2 - s0^2 = 9*f + 6*f*e_bar
    f = |P cap M| / |P|   (public fraction of the month's rows)
One equation, two unknowns -> scenario table + predicted MINUS score for each.
The minus probe resolves f EXACTLY:  f = (s_plus^2 + s_minus^2 - 2*s0^2)/18
"""
import math

S0 = 0.683791578            # v24 public score (exact, in-band adopted)
SP = 1.052248929            # probe_m201601_plus (user-pasted)
DELTA = 3.0
N_M = 15647                 # rows of 2016-01 in Test.csv
N_TEST = 280961
P_OLD = 109222              # old (wrong) decode |P|
P_RULES = 84288             # rules text "approximately 30%"

d = SP ** 2 - S0 ** 2
print(f"s_plus^2   = {SP**2:.9f}")
print(f"s0^2       = {S0**2:.9f}")
print(f"Delta_plus = {d:.9f}   (= 9f + 6f*e_bar)")
print(f"f if e_bar = 0        : {d/9:.5f}")
print()
print("HARD FACTS from the plus score alone:")
print(f"  1. Delta > 0  ->  f(2016-01) > 0 : the month INTERSECTS the public set")
print(f"  2. f(2016-01) <= {N_M/P_OLD:.4f} @ |P|=109,222 ({N_M:,}/{P_OLD:,})")
print(f"     f(2016-01) <= {N_M/P_RULES:.4f} @ |P|={P_RULES:,} (30% rules text)")
print()

rows = [
    ("uniform random split (f = n_m/N_test)", N_M / N_TEST),
    ("no bias (f = Delta/9)", d / 9),
    ("half-month public @ |P|=109,222", 0.5 * N_M / P_OLD),
    ("OLD BLOCK: fully public @ |P|=109,222", N_M / P_OLD),
    ("fully public @ |P|=84,288 (30% rules)", N_M / P_RULES),
]
print(f"{'scenario':<40}{'f':>8}{'e_bar':>9}{'pred s_minus':>14}")
for name, f in rows:
    ebar = (d / f - DELTA ** 2) / (2 * DELTA)
    sm2 = S0 ** 2 + 9 * f - 6 * f * ebar
    print(f"{name:<40}{f:>8.4f}{ebar:>9.4f}{math.sqrt(sm2):>14.4f}")

print()
print("INSTANT READ when the minus score lands:")
print("  s_minus ~ 0.911  -> uniform-like split (f~0.056, month over-biased +0.41)")
print("  s_minus ~ 1.052  -> zero bias, f ~ 0.071 (half-month public)")
print("  s_minus ~ 1.552  -> OLD BLOCK holds (f=0.143, month under-biased -0.76)")
print("  s_minus > 1.60   -> even more public mass than the old block")
print()
print("Exact pair formula (use once both scores exist):")
print(f"  f = (s_plus^2 + s_minus^2 - 2*{S0}^2)/18.0")
print(f"  bias bonus: (s_plus^2 - s_minus^2)/12 = public mean error on the month")
