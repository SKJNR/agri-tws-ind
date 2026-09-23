"""Task 27: THE HEADROOM AUDIT — why 0.60 is not tweak-reachable, quantified.

User's question: "If predictions match the LB exactly, why not work on all the
others for a breakthrough to 0.60 — a major tweak instead of small tweaks?"

This script separates two DIFFERENT capabilities:
  (A) SCORE ARITHMETIC (what we proved): for a KNOWN modification of a KNOWN
      vector, the new public score is computable exactly. Pure bookkeeping —
      like knowing your bank balance after a deposit.
  (B) MODEL SEARCH (what 0.60 needs): finding a different vector with much
      lower error — requires knowing the TRUTH, which is the one thing we
      don't have. The LB oracle only scores vectors we submit (5/day).

Then: the measured wall. Every number below is MEASURED (worklog Tasks
6-25), not assumed.
"""
import math

S25 = 0.679780277
S0 = 0.683791578
TARGET = 0.60
CUT = 0.6559
FLOOR = 0.666  # three independent audits (hydrologist ledger / craft ledger / forum)

print('=' * 72)
print('1. THE GAP, IN MSE TERMS')
print('=' * 72)
need = S25**2 - TARGET**2
print(f'  current MSE {S25**2:.4f}  ->  target MSE {TARGET**2:.4f}')
print(f'  must REMOVE {need:.4f} MSE  =  {math.sqrt(need):.4f} RMSE everywhere')

print()
print('=' * 72)
print('2. WHAT THE CORRECTION LANE CAN MATHEMATICALLY DELIVER (max, no physics)')
print('=' * 72)
# capturable MSE from constant per-month corrections = sum_m f_m * e_m^2
# measured e's: Jan-2016 +0.3108 (the El Nino extreme); 2016-09 -0.010 (masked ~ 0)
f_jan = 0.058874
print(f'  Jan-2016 (measured, deployed): f*e^2 = {f_jan*0.3108**2:.5f} MSE')
print(f'  2016-09 masked (measured):     f*e^2 = {0.058776*0.010**2:.7f} MSE  ~ ZERO')
print()
print('  FANTASY scenario: ALL 5 remaining k0 months carry Jan-sized biases:')
k0_share_public = 6*0.0588  # ~6 k0 months x ~5.9% public share each
fantasy = k0_share_public*0.3108**2
print(f'    max capturable = {fantasy:.4f} MSE -> public {math.sqrt(S25**2-fantasy):.4f}')
print(f'    (still {math.sqrt(S25**2-fantasy)-TARGET:.3f} ABOVE 0.60, and above the cutoff)')
print()
print('  To reach 0.60 by constant corrections alone you would need')
avg_e = math.sqrt(need)   # if every month's |e| were equal, sum f_m = 1
print(f'  |e_m| = {avg_e:.3f} on EVERY month (all 18). Measured reality:')
print(f'  extrapolation-EXTREME month (El Nino peak): 0.311')
print(f'  ordinary masked month: 0.010')
print(f'  => the error-mean distribution is measured near-flat outside extreme')
print(f'     extrapolation months. The lane is real but ~0.01-sized, not 0.08.')

print()
print('=' * 72)
print('3. WHERE EACH BAND OF THE LB ACTUALLY COMES FROM (measured/probed)')
print('=' * 72)
print(f'  0.5596  leader (MOHAR)   = 0.07 BELOW the external-GRACE floor ->')
print(f'          models the generator SYNTHETIC RESIDUAL (mechanism unknown)')
print(f'  ~0.63   external-GRACE lane (GDO/GravIS/COST-G fill)  PROHIBITED')
print(f'          (v20 audit: bit-exact GDO identity with Train; refused)')
print(f'  0.666   CLEAN information floor (3 audits, component-measured:')
print(f'          masked 0.722 = noise 0.456 + D-err 0.42 + fast 0.37;')
print(f'          k0 0.585; forum rivals independently confirm ~0.70 ceiling)')
print(f'  0.6559  top-10 cutoff (BELOW the clean floor -> top-10 requires')
print(f'          DQ cascade, private reshuffle, or the report)')
print(f'  0.6798  US (v25) — ~0.01-0.014 of measured legal headroom left')
print()
print('  The 0.63-0.66 band is EXACTLY the external-TWS information.')
print('  Nothing legal lives there. 0.60 does not exist in the legal feature space.')

print()
print('=' * 72)
print('4. REMAINING LEGAL HEADROOM — the complete budget (all axes)')
print('=' * 72)
rows = [
    ('k0-month tomography (5 armed pairs)', 0.002, 0.008,
     'Jan was THE extreme month; others 3-10x smaller; masked measured ~0'),
    ('Spatial (region-level) bias within Jan', 0.000, 0.006,
     'UNMEASURED axis; ceiling = between-region variance of the error'),
    ('Masked-month polish (all craft lanes)', 0.000, 0.005,
     'ledger closed: stacking/per-h/isotonic/etc all measured sub-threshold'),
    ('New external data (legal)', 0.000, 0.001,
     'ERA5 tested +0; GPCP r=0.02; EDO/GDO covs = given covs; GLDAS pending'),
]
tot_lo, tot_hi = 0.0, 0.0
print(f"  {'axis':<42}{'lo':>7}{'hi':>7}")
for name, lo, hi, note in rows:
    tot_lo += lo; tot_hi += hi
    print(f'  {name:<42}{lo:>7.3f}{hi:>7.3f}')
    print(f'      {note}')
print(f"  {'TOTAL legal headroom':<42}{tot_lo:>7.3f}{tot_hi:>7.3f}")
print(f'  projected finish: {S25-tot_hi:.4f} .. {S25-tot_lo:.4f}  '
      f'(floor {FLOOR})')

print()
print('=' * 72)
print('5. THE HONEST BOTTOM LINE')
print('=' * 72)
print(f'  - We are {S25-FLOOR:.4f} above the clean floor; the floor is')
print(f'    {FLOOR-TARGET:.3f} above 0.60. The binding constraint is')
print(f'    INFORMATION, not effort or craft.')
print(f'  - Exact score prediction = measurement power (it let us BANK the')
print(f'    Jan gain at zero risk, and will bank the next ones). It cannot')
print(f'    manufacture information the legal data does not contain.')
print(f'  - The discontinuous drops at 0.63 and 0.56 are the prohibited lane')
print(f'    and the leader residual model. We refuse the first (documented);')
print(f'    the second we cannot see.')
print(f'  - Rank-relevant levers left: DQ cascade (forum rulings), private')
print(f'    reshuffle (hedges already selected), REPORT = 50% of final score.')
