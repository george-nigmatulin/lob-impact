"""Phase 1 scorecard: response ratio, banded Hurst exponents and short-lag gamma per source.

Reads a cache written by lob_impact.flow_balance and prints, for every source in it:

  R(1), R(100)     response to a trade after 1 and 100 trades, in ticks. The drift-corrected
                   Rc is used when the cache has it: the raw R contains a pure-drift term
                   for any source with a sign imbalance.
  R100/R1          how far the price keeps moving in the direction of a trade after it.
                   Not identified (n/i) when R(1) or R(100) is unavailable, when R(1) = 0,
                   or, if the cache has the reference row, when |R(1)| is below
                   R1_FLOOR_VS_REF times the reference's |R(1)| (the denominator is then
                   noise).
  H 2-10, H 10-30  Hurst exponent refitted from the cached variance curve on two lag bands
                   separately, since V(l) need not be a single power law across them and
                   one blended exponent would hide which end deviates.
  gamma2-15        sign-ACF exponent refitted on short lags, where the ACF is usually still
                   above noise. Informational only.
Values that cannot be computed are printed as n/a.

A second table gives the signed differences (source - reference) of the ratio and of both
H bands. The response ratio is the primary signal, short-lag H the secondary one.

Inputs
  --cache PATH, or by default <LOB_IMPACT_RESULTS>/flow_balance/flow_balance_<STOCK><tag>.npz.
  Keys used per source <name>: <name>_Rc or <name>_R (lag 1..200), <name>_V (variance
  curve on flow_balance.var_lags()), <name>_C (ACF at lags 1..300). Sources are listed in
  cache order, with the reference row first.
Output
  printed tables only.

Example
  python -m lob_impact.scorecard --stock XYZ
  python -m lob_impact.scorecard --cache results/flow_balance/flow_balance_XYZ_short.npz --ref Real
"""
import argparse
import os

import numpy as np

from lob_impact.flow_balance import REAL, var_lags
from lob_impact.paths import LOB_IMPACT_RESULTS

H_BANDS = ((2, 10), (10, 30))   # lag bands (trades) of the two H fits
GAMMA_FIT = (2, 15)             # short-lag window of the gamma refit
R_AT = (1, 100)                 # lags (trades) of the response ratio R(100)/R(1)
R1_FLOOR_VS_REF = 0.10          # |R(1)| must reach this fraction of the reference's |R(1)|


def _resp_key(z, m):
    """Cache key of the response curve: drift-corrected Rc if present, else raw R."""
    for suffix in ('_Rc', '_R'):
        if m + suffix in z.files:
            return m + suffix
    return None


def r_ratio(z, m, ref_r1=None):
    """R(1), R(100) and their ratio (None when not identified); None if no usable curve.

    The floor is set on the reference's R(1), not on the source's own curve: a source
    that barely reacts to its own trade would otherwise clear a self-relative floor.
    """
    k = _resp_key(z, m)
    if k is None:
        return None
    R = np.asarray(z[k], float)
    lo, hi = R_AT
    if len(R) < hi or not np.isfinite(R[lo - 1]) or not np.isfinite(R[hi - 1]):
        return None
    r1, r100 = float(R[lo - 1]), float(R[hi - 1])
    if r1 == 0:
        return dict(r1=r1, r100=r100, ratio=None)
    if ref_r1 is not None and abs(r1) < R1_FLOOR_VS_REF * abs(ref_r1):
        return dict(r1=r1, r100=r100, ratio=None)
    return dict(r1=r1, r100=r100, ratio=float(r100 / r1))


def h_bands(z, m):
    """H per lag band, refitted from the cached V(l) rather than reusing the blended H."""
    k = m + '_V'
    if k not in z.files:
        return {}
    V = np.asarray(z[k], float)
    ls = var_lags()                 # the grid flow_balance.var_H writes V against
    ls = ls[:len(V)]
    V = V[:len(ls)]
    ok = np.isfinite(V) & (V > 0)
    if ok.sum() < 4:
        return {}
    x, y = np.log(ls[ok]), np.log(V[ok])
    out = {}
    for a, b in H_BANDS:
        m_ = (ls[ok] >= a) & (ls[ok] <= b)
        out[(a, b)] = float(np.polyfit(x[m_], y[m_], 1)[0]) / 2.0 if m_.sum() > 2 else float('nan')
    return out


def gamma_short(z, m):
    """Sign-ACF exponent refitted over GAMMA_FIT (informational, not a decision variable)."""
    k = m + '_C'
    if k not in z.files:
        return float('nan')
    C = np.asarray(z[k], float)
    ls = np.arange(1, len(C) + 1)
    lo, hi = GAMMA_FIT
    m_ = (ls >= lo) & (ls <= hi) & (C > 0) & np.isfinite(C)
    if m_.sum() < 5:
        return float('nan')
    return -float(np.polyfit(np.log(ls[m_]), np.log(C[m_]), 1)[0])


def sources_in(z, ref):
    """Source names in cache order (those with a response or variance curve), reference first."""
    models = [m for m in dict.fromkeys(k.rsplit('_', 1)[0] for k in z.files)
              if _resp_key(z, m) or (m + '_V') in z.files]
    return ([ref] if ref in models else []) + [m for m in models if m != ref]


def print_main_table(z, models, ref_r1):
    hdr = '%-13s %7s %8s %9s %9s %9s %9s' % (
        'model', 'R(1)', 'R(100)', 'R100/R1', 'H 2-10', 'H 10-30', 'gamma2-15')
    print(hdr)
    print('-' * len(hdr))
    for m in models:
        rr, hh = r_ratio(z, m, ref_r1), h_bands(z, m)
        g = gamma_short(z, m)
        f = lambda v: ('%9.3f' % v) if v is not None and np.isfinite(v) else '      n/a'
        print('%-13s %7s %8s %9s %9s %9s %9s' % (
            m,
            ('%7.3f' % rr['r1']) if rr else '    n/a',
            ('%8.3f' % rr['r100']) if rr else '     n/a',
            ('%9.2f' % rr['ratio']) if (rr and rr['ratio'] is not None) else '      n/i',
            f(hh.get(H_BANDS[0], float('nan'))),
            f(hh.get(H_BANDS[1], float('nan'))),
            f(g)))


def print_distance_table(z, models, ref, ref_r, ref_r1, ref_h):
    print('distance to "%s" (sign: + means the source is above the reference)' % ref)
    hdr2 = '%-13s %12s %10s %10s' % ('model', 'd(R100/R1)', 'd(H 2-10)', 'd(H 10-30)')
    print(hdr2)
    print('-' * len(hdr2))
    for m in models:
        if m == ref:
            continue
        rr, hh = r_ratio(z, m, ref_r1), h_bands(z, m)
        d_r = (rr['ratio'] - ref_r['ratio']) \
            if (rr and rr['ratio'] is not None and ref_r['ratio'] is not None) \
            else float('nan')
        ds = [(hh.get(b, float('nan')) - ref_h.get(b, float('nan'))) for b in H_BANDS]
        f = lambda v: ('%+10.3f' % v) if np.isfinite(v) else '       n/a'
        print('%-13s %12s %10s %10s' % (
            m, ('%+12.2f' % d_r) if np.isfinite(d_r) else '         n/a', f(ds[0]), f(ds[1])))


def parse_args(argv):
    ap = argparse.ArgumentParser(
        prog='python -m lob_impact.scorecard',
        description='Response ratio, banded H and short-lag gamma per source of a '
                    'flow_balance cache, with distances to the reference row.')
    ap.add_argument('--cache', default=None,
                    help='flow_balance_<STOCK><tag>.npz (default: built from --stock/--tag)')
    ap.add_argument('--stock', default=None,
                    help='locate the cache LOB_IMPACT_RESULTS/flow_balance/flow_balance_<STOCK><tag>.npz')
    ap.add_argument('--tag', default='', help='filename suffix of the cache, with --stock')
    ap.add_argument('--ref', default=REAL, help='row every other row is measured against')
    a = ap.parse_args(argv)
    if a.cache is None:
        if not a.stock:
            ap.error('give --cache or --stock')
        a.cache = os.path.join(LOB_IMPACT_RESULTS, 'flow_balance',
                               f'flow_balance_{a.stock}{a.tag}.npz')
    return a


def main(argv=None):
    a = parse_args(argv)
    z = np.load(a.cache, allow_pickle=False)
    models = sources_in(z, a.ref)

    ref_r = r_ratio(z, a.ref) if a.ref in models else None
    ref_r1 = ref_r['r1'] if ref_r else None
    ref_h = h_bands(z, a.ref) if a.ref in models else {}

    print('cache: %s' % os.path.basename(a.cache))
    print('primary signal: response ratio R(100)/R(1); secondary: H on two lag bands;')
    print('gamma2-15 is informational. n/a = not available; n/i = ratio not identified '
          '(R(1) or R(100) missing, R(1) = 0, or |R(1)| < %g x reference |R(1)|).'
          % R1_FLOOR_VS_REF)
    print()
    print_main_table(z, models, ref_r1)

    if ref_r is None or ref_r.get('ratio') is None:
        print('\nreference row "%s" is missing from the cache or its R100/R1 is not '
              'identified: distances not computed' % a.ref)
        return
    print()
    print_distance_table(z, models, a.ref, ref_r, ref_r1, ref_h)


if __name__ == '__main__':
    main()
