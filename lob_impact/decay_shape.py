"""Shape of the relaxation segment of the impact master curve, and its law where identifiable.

The segment is classified as a rise, flat, or decline, and the decay law (power law or
exponential) is named only where the data can tell the two apart.

The input is a master curve master(v) = <I(v)> / <I(1)>: the mean k-clock impact of a
relaxation-shape rollout set divided by its mean at execution end, on v = k / n_ins
(v = 1: last child executed). Only the cooling segment z = v - 1 >= Z0 is used.

1. Materiality first, without a fit: drop = median of the first EDGE_FRAC of the segment
   minus the median of the last EDGE_FRAC, in peak units (master = 1 at execution end).
       drop <= -RISE_T     -> 'rise'
       |drop| < FLAT_T     -> 'flat'
2. Only a materially declining segment is asked about its form. The fits a + b*z^(-g)
   (g on GGRID) and a + b*exp(-z/t) (t on TGRID) are compared with a constant by AIC on an
   autocorrelation-corrected sample size, n_eff = n / (1 + 2 * sum_k rho_k), from the better
   fit's residuals (lags summed while rho_k > 0, up to n/4): the curve is a mean over
   rollouts and neighbouring points are strongly correlated, so the raw n would overstate
   the evidence. A form is named only if the winner beats the constant by dAIC <= DAIC_MIN,
   beats the other form by at least MARGIN, has a positive amplitude b, and its parameter
   is not on the edge of its grid:
       -> 'pl' or 'exp'
   otherwise the segment declines but its form is not identified:
       -> 'down'

Inputs
------
One master-curve npz per stock; default path
    LOB_IMPACT_RESULTS/master_curve/master_curve_<STOCK>_relaxation.npz
(--master_curve takes a pattern containing {stock}). Keys read:
    vgrid        float  v = k / n_ins, k = 0..n_ins + n_cool
    <M>_master   float  master curve of model M on vgrid (NaN where not measured)
    <M>_sig      bool   whether the execution-end impact <I(1)> of M cleared its gate
Every model with a <M>_master key is evaluated; a model with fewer than MIN_POINTS finite
points on the cooling segment is skipped.

Outputs
-------
<out_dir>/decay_shape_<STOCK>.md (out_dir default LOB_IMPACT_RESULTS/master_curve), one row per
    model: sig (y/n), e/p (last finite master value: end of window over execution end),
    drop, slope per unit z (least squares), n_eff, dAIC(PL), dAIC(EXP), ghat, verdict.
<out_dir>/decay_shape_<STOCK>.npz: <M>_drop, <M>_slope, <M>_n, <M>_n_eff, <M>_dAIC_pl,
    <M>_dAIC_exp, <M>_ghat, <M>_tau, <M>_ep (numbers) and <M>_verdict (string).

Usage
-----
    python -m lob_impact.decay_shape --stock XYZ
    python -m lob_impact.decay_shape --stock XYZ,ABC \\
        --master_curve 'caches/master_curve_{stock}_relaxation.npz'
"""
import argparse
import os

import numpy as np

from lob_impact.paths import LOB_IMPACT_RESULTS

GGRID = np.arange(0.05, 1.51, 0.05)   # power-law exponent search grid
TGRID = np.arange(0.1, 5.01, 0.1)     # exponential timescale search grid (units of z)
Z0 = 0.2          # start of the cooling segment, in execution durations past execution end
EDGE_FRAC = 0.15  # fraction of the segment averaged (median) at each end for the drop
FLAT_T = 0.08     # |drop| below this: no material move (peak units)
RISE_T = 0.08     # drop below -RISE_T: the segment rises
MARGIN = 10.0     # dAIC margin required to tell power law from exponential
DAIC_MIN = -10    # the winning form must beat the constant by at least this much
MIN_POINTS = 10   # finite segment points needed to evaluate a model


def aic(rss, n, k):
    return n * np.log(max(rss, 1e-12) / n) + 2 * k


def fit_two_param(z, y, regressor):
    """Least squares y ~ a + b * regressor; returns ((a, b), rss)."""
    X = np.column_stack([np.ones_like(z), regressor])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    rss = float(np.sum((y - X @ coef) ** 2))
    return coef, rss


def n_effective(resid, max_lag=None):
    """Effective sample size under serial correlation: n / (1 + 2 * sum rho_k), summing
    lags while the autocorrelation stays positive (Bartlett-style cut), floor 2."""
    n = len(resid)
    r = resid - resid.mean()
    denom = float(np.sum(r * r))
    if denom <= 0:
        return float(n)
    max_lag = max_lag or max(1, n // 4)
    s = 0.0
    for k in range(1, max_lag + 1):
        rho = float(np.sum(r[:-k] * r[k:]) / denom)
        if rho <= 0:
            break
        s += rho
    return float(max(2.0, n / (1.0 + 2.0 * s)))


def verdict_for(zz, y):
    """Return dict(drop, slope, n, n_eff, dAIC_pl, dAIC_exp, ghat, tau, verdict)."""
    n = len(y)
    slope = float(np.polyfit(zz, y, 1)[0])

    # 1. materiality, fit-free
    k = max(2, int(round(EDGE_FRAC * n)))
    drop = float(np.median(y[:k]) - np.median(y[-k:]))

    # fits (recorded even when the form is not named)
    rss0 = float(np.sum((y - y.mean()) ** 2))
    best_pl = (np.inf, np.nan, None)
    for g in GGRID:
        coef, rss = fit_two_param(zz, y, zz ** (-g))
        if rss < best_pl[0]:
            best_pl = (rss, float(g), coef)
    best_ex = (np.inf, np.nan, None)
    for t in TGRID:
        coef, rss = fit_two_param(zz, y, np.exp(-zz / t))
        if rss < best_ex[0]:
            best_ex = (rss, float(t), coef)

    # autocorrelation-corrected sample size from the better fit's residuals
    if best_pl[0] <= best_ex[0]:
        X = np.column_stack([np.ones_like(zz), zz ** (-best_pl[1])])
        resid = y - X @ best_pl[2]
    else:
        X = np.column_stack([np.ones_like(zz), np.exp(-zz / best_ex[1])])
        resid = y - X @ best_ex[2]
    n_eff = n_effective(resid)

    a0 = aic(rss0, n_eff, 1)
    d_pl = aic(best_pl[0], n_eff, 3) - a0
    d_ex = aic(best_ex[0], n_eff, 3) - a0

    ghat, tau = best_pl[1], best_ex[1]
    if drop <= -RISE_T:
        v = 'rise'
    elif abs(drop) < FLAT_T:
        v = 'flat'
    else:
        # material decline: name the form only if the data can tell the forms apart
        pl_wins = d_pl <= d_ex
        winner_d = min(d_pl, d_ex)
        margin = abs(d_pl - d_ex)
        amp_ok = (best_pl[2][1] > 0) if pl_wins else (best_ex[2][1] > 0)
        edge = ((ghat >= GGRID[-1] - 1e-9 or ghat <= GGRID[0] + 1e-9) if pl_wins
                else tau >= TGRID[-1] - 1e-9)
        if winner_d <= DAIC_MIN and margin >= MARGIN and amp_ok and not edge:
            v = 'pl' if pl_wins else 'exp'
        else:
            v = 'down'
    return dict(drop=drop, slope=slope, n=n, n_eff=n_eff, dAIC_pl=d_pl, dAIC_exp=d_ex,
                ghat=ghat, tau=tau, verdict=v)


def run(stock, src, out_dir):
    """Evaluate every model of one master-curve cache; write the .md and .npz outputs."""
    if not os.path.exists(src):
        print('%s: no relaxation master-curve cache at %s, skipped' % (stock, src))
        return None
    z = np.load(src, allow_pickle=False)
    v = z['vgrid']
    models = sorted({k[:-len('_master')] for k in z.files if k.endswith('_master')})
    lines = ['| model | sig | e/p | drop | slope/z | n_eff | dAIC(PL) | dAIC(EXP) | ghat '
             '| verdict |',
             '|---|---|---|---|---|---|---|---|---|---|']
    out = {}
    for m in models:
        y_full = np.asarray(z[m + '_master'], dtype=float)
        sig = bool(z[m + '_sig'])
        sel = v - 1.0 >= Z0
        zz, y = (v[sel] - 1.0), y_full[sel]
        ok = np.isfinite(y)
        zz, y = zz[ok], y[ok]
        if len(y) < MIN_POINTS:
            continue
        r = verdict_for(zz, y)
        ep = float(y_full[np.isfinite(y_full)][-1])
        lines.append(f'| {m} | {"y" if sig else "n"} | {ep:.2f} | {r["drop"]:+.3f} | '
                     f'{r["slope"]:+.3f} | {r["n_eff"]:.1f} | {r["dAIC_pl"]:+.1f} | '
                     f'{r["dAIC_exp"]:+.1f} | {r["ghat"]:.2f} | {r["verdict"]} |')
        out[m] = dict(r, ep=ep)
    txt = '\n'.join(lines)
    print('\n=== %s ===\n%s' % (stock, txt))
    os.makedirs(out_dir, exist_ok=True)
    dst = os.path.join(out_dir, 'decay_shape_%s.md' % stock)
    with open(dst, 'w') as f:
        f.write(txt + '\n')
    np.savez_compressed(os.path.splitext(dst)[0] + '.npz',
                        **{f'{m}_{k}': val for m, d in out.items() for k, val in d.items()
                           if not isinstance(val, str)},
                        **{f'{m}_verdict': d['verdict'] for m, d in out.items()})
    print('saved -> %s (+ .npz)' % dst)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog='python -m lob_impact.decay_shape',
                                 description='Shape of the relaxation segment of the impact '
                                             'master curve: rise, flat or decline, and the '
                                             'decay law where the data identify it. Full '
                                             'method: python -m pydoc lob_impact.decay_shape')
    ap.add_argument('--stock', required=True, help='ticker, or several comma-separated')
    ap.add_argument('--master_curve',
                    default=os.path.join(LOB_IMPACT_RESULTS, 'master_curve',
                                         'master_curve_{stock}_relaxation.npz'),
                    help='master-curve npz path; {stock} is replaced by the ticker')
    ap.add_argument('--out_dir', '--outdir', dest='out_dir',
                    default=os.path.join(LOB_IMPACT_RESULTS, 'master_curve'),
                    help='output directory (default: LOB_IMPACT_RESULTS/master_curve)')
    args = ap.parse_args(argv)
    done = [run(st, args.master_curve.replace('{stock}', st), args.out_dir)
            for st in [s for s in args.stock.split(',') if s]]
    if not any(r is not None for r in done):
        raise SystemExit('no master-curve cache found (build it with python -m lob_impact.master_curve '
                         '--shape relaxation, or pass --master_curve)')


if __name__ == '__main__':
    main()
