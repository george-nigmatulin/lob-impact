"""Build-up exponent delta of the square-root law, with a day-clustered bootstrap CI.

The square-root law of market impact reads I ~ sigma * (Q / V)^delta, with
delta = 0.5 for a square root. This tool estimates delta per model from the
points cache written by lob_impact.points and reports a 95% confidence interval that
resamples trading days, the independent unit of the rollouts.

Inputs:
    --points   points cache from lob_impact.points (default <LOB_IMPACT_RESULTS>/points_<STOCK>.npz),
               keys <model>_Q, <model>_I, <model>_days (<model>_kc is not used here)
    --daily    daily statistics CSV, format as in lob_impact.vol (ticker, day, open_price,
               highest_price, lowest_price, close_price, execution_sum)
    --stock    ticker whose daily rows are joined to the points by day
    --models   comma-separated model names, processed in this order

Transformation (per point; points with non-finite x or y are dropped):
    x = log(Q / V)    V = execution_sum of the point's day
    y = I / sigma     sigma = daily Parkinson volatility of that day (lob_impact.vol)

Point estimate (signed binned estimator): sort the points by x, cut them into
NBINS equal-count bins, take per bin the mean x and the signed mean y, and fit
log(mean y) = alpha + delta * mean x by OLS over the bins whose mean y is
positive. x takes few distinct values (one per child count and day), so many
points tie; tied points keep their cache order (a stable sort), which makes the
estimate independent of the platform's sort implementation. An unstable sort
splits ties at bin edges arbitrarily and can move the estimate on small samples. Negative responses are averaged in rather than discarded (no I > 0
selection), so the bin means are not biased by conditioning on the sign of the
response, and averaging within bins damps the heavy tails of single points.
Fewer than MIN_BINS positive bins, or a degenerate x range, gives NaN (no
identifiable law).

Confidence interval: fixed bin edges at the NBINS + 1 quantiles of x (the top
edge raised by 1e-9 so the maximum falls inside), per-day sums of y, x and counts
in each bin, then N_BOOT resamples of the days with replacement. Each resample
refits delta on the summed bins (empty bins dropped); resamples that return NaN
are discarded; the CI is the 2.5 and 97.5 percentiles of the rest. Fixed edges
make one resample cost O(days * bins). Because the interval uses these fixed
edges and the point estimate uses equal-count chunks, the two can disagree near
the identification limit: an interval can exist while delta is NaN.

A single numpy RandomState(SEED) stream is advanced across the models in
--models order; models absent from the cache do not consume draws, so the
interval of a model depends on which models precede it.

Outputs (under --out_dir, default LOB_IMPACT_RESULTS):
    delta_<STOCK>.json   {"<STOCK>:<model>": {"delta", "lo", "hi", "hw", "n_day", "n_pts"}, ...,
                          "absent": [...], "degenerate": [...]}
                         delta  point estimate (NaN when not identified; written as
                                the non-standard JSON token NaN)
                         lo/hi  2.5 / 97.5 percentiles of the finite bootstrap deltas
                         hw     (hi - lo) / 2
                         n_day  days with at least one usable point
                         n_pts  usable points
                         absent      models missing from the points cache
                         degenerate  models whose N_BOOT resamples were all NaN
                         (absent and degenerate models have no per-model entry)
    delta_<STOCK>.md     the same as a Markdown table

Usage:
    python -m lob_impact.delta --stock XYZ --models MyModel,Baseline --daily daily_stats.csv \\
        [--points points_XYZ.npz] [--out_dir DIR]
"""
import argparse
import json
import os

import numpy as np

from lob_impact.paths import LOB_IMPACT_RESULTS
from lob_impact.vol import daily_sigmas

NBINS = 18          # equal-count bins in log(Q / V)
MIN_BINS = 5        # fewest positive bins for an identifiable slope
N_BOOT = 400        # day-bootstrap resamples
SEED = 20260820     # seed of the bootstrap stream (fixed for reproducibility)


def load_points(points_path, stock, model, sig):
    """(x, y, days) of one model with x = log(Q / V), y = I / sigma; None if not cached.

    sig is the output of lob_impact.vol.daily_sigmas; V and the Parkinson sigma are looked
    up by (stock, day). Points without daily statistics (NaN) or with non-finite x or
    y are dropped.
    """
    with np.load(points_path, allow_pickle=False) as d:
        if model + '_Q' not in d.files:
            return None
        Q = d[model + '_Q']; I = d[model + '_I']
        days = d[model + '_days']
    V = np.array([sig.get((stock, dd), {}).get('V', np.nan) for dd in days])
    sg = np.array([sig.get((stock, dd), {}).get('parkinson', np.nan) for dd in days])
    x = np.log(Q / V); y = I / sg
    ok = np.isfinite(x) & np.isfinite(y)
    return x[ok], y[ok], days[ok]


def delta_binned_exact(x, y, nbins=NBINS):
    """Signed binned delta: equal-count bins of x, signed mean y, OLS through positive means.

    Always uses nbins chunks of argsort(x, kind='stable') (empty chunks skipped).
    Returns (delta or NaN, bin mean x, bin mean y).
    """
    order = np.argsort(x, kind='stable')
    xc, ym = [], []
    for chunk in np.array_split(order, nbins):
        if chunk.size:
            xc.append(x[chunk].mean()); ym.append(y[chunk].mean())
    xc, ym = np.array(xc), np.array(ym)
    pos = ym > 0
    if pos.sum() < MIN_BINS or (xc[pos].max() - xc[pos].min()) < 1e-6:
        return np.nan, xc, ym
    return float(np.polyfit(xc[pos], np.log(ym[pos]), 1)[0]), xc, ym


def day_stats_fixed_edges(x, y, days, edges):
    """Per-day, per-bin sums (SY, SX, N), each (n_days, NBINS), for fixed bin edges.

    Days are indexed in sorted order of their labels. A point goes to bin
    searchsorted(edges, x, 'right') - 1, clipped to [0, len(edges) - 2].
    """
    di = {d: i for i, d in enumerate(np.unique(days))}
    nd = len(di)
    b = np.clip(np.searchsorted(edges, x, side='right') - 1, 0, len(edges) - 2)
    didx = np.array([di[d] for d in days])
    SY = np.zeros((nd, NBINS)); SX = np.zeros((nd, NBINS)); N = np.zeros((nd, NBINS))
    for k in range(NBINS):
        m = b == k
        np.add.at(SY[:, k], didx[m], y[m])
        np.add.at(SX[:, k], didx[m], x[m])
        np.add.at(N[:, k], didx[m], 1)
    return SY, SX, N


def delta_from_daysum(SY, SX, N, pick):
    """delta from the bins summed over the day indices in pick (repeats allowed); NaN if
    fewer than MIN_BINS non-empty bins have a positive mean y or the x range is degenerate."""
    sy = SY[pick].sum(0); sx = SX[pick].sum(0); n = N[pick].sum(0)
    ok = n > 0
    ym = np.where(ok, sy / np.maximum(n, 1), np.nan)
    xc = np.where(ok, sx / np.maximum(n, 1), np.nan)
    pos = ok & (ym > 0)
    if pos.sum() < MIN_BINS or (xc[pos].max() - xc[pos].min()) < 1e-6:
        return np.nan
    return float(np.polyfit(xc[pos], np.log(ym[pos]), 1)[0])


def day_bootstrap(x, y, days, rng, n_boot=N_BOOT):
    """Point estimate and day-bootstrap deltas of one model.

    Returns (delta, finite bootstrap deltas, number of days). Consumes n_boot
    draws of rng.randint(0, n_days, n_days).
    """
    d0, _, _ = delta_binned_exact(x, y)
    edges = np.quantile(x, np.linspace(0, 1, NBINS + 1))
    edges[-1] += 1e-9
    SY, SX, N = day_stats_fixed_edges(x, y, days, edges)
    nd = SY.shape[0]
    ds = [delta_from_daysum(SY, SX, N, rng.randint(0, nd, nd)) for _ in range(n_boot)]
    ds = np.array([v for v in ds if np.isfinite(v)])
    return d0, ds, nd


def run(points_path, sig, stock, models):
    """delta and CI per model, in the order given.

    Returns (results, rows, absent, degenerate): results maps "<stock>:<model>" to
    the entry dict, rows lists (model, entry) in processing order.
    """
    rng = np.random.RandomState(SEED)
    res, rows, absent, degenerate = {}, [], [], []
    for model in models:
        r = load_points(points_path, stock, model, sig)
        if r is None:
            absent.append(model)
            print('%s %-10s absent from points cache' % (stock, model), flush=True)
            continue
        x, y, days = r
        if x.size == 0:
            raise SystemExit('%s %s: no point has daily statistics; check that the daily CSV '
                             'has rows for ticker %s on the cached days' % (stock, model, stock))
        d0, ds, nd = day_bootstrap(x, y, days, rng)
        if ds.size == 0:
            # every resample NaN: no identifiable law in the day sums
            degenerate.append(model)
            print('%s %-10s bootstrap degenerate (all resamples NaN): no identifiable law'
                  % (stock, model), flush=True)
            continue
        lo, hi = np.percentile(ds, [2.5, 97.5])
        entry = dict(delta=float(d0), lo=float(lo), hi=float(hi),
                     hw=float(0.5 * (hi - lo)), n_day=int(nd), n_pts=int(x.size))
        res['%s:%s' % (stock, model)] = entry
        rows.append((model, entry))
        print('%s %-10s delta %.3f [%.3f, %.3f]  hw %.3f  %dd/%d pts'
              % (stock, model, d0, lo, hi, 0.5 * (hi - lo), nd, x.size), flush=True)
    return res, rows, absent, degenerate


def markdown(stock, rows, absent, degenerate):
    """Markdown report: one table row per model with a CI, then the absent/degenerate lists."""
    L = ['# %s: build-up exponent delta with day-clustered 95%% CIs '
         '(fixed-edge day bootstrap, B=%d)' % (stock, N_BOOT), '']
    L.append('| model | delta (binned) | 95% CI (day) | half-width | n_day | n_pts |')
    L.append('|---|---|---|---|---|---|')
    for model, v in rows:
        L.append('| %s | %.2f | [%.2f, %.2f] | %.3f | %d | %d |'
                 % (model, v['delta'], v['lo'], v['hi'], v['hw'], v['n_day'], v['n_pts']))
    L.append('')
    if absent:
        L.append('Absent from the points cache (run lob_impact.points for them): ' + ', '.join(absent))
    if degenerate:
        L.append('Degenerate bootstrap (every resample NaN, no identifiable law): '
                 + ', '.join(degenerate))
    return '\n'.join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog='python -m lob_impact.delta',
        description='Build-up exponent delta (signed binned estimator) with a day-clustered '
                    'bootstrap CI, per model. See the module docstring for the method.')
    ap.add_argument('--stock', required=True, help='ticker of the points cache and the daily rows')
    ap.add_argument('--models', required=True, help='comma-separated model names, in report order')
    ap.add_argument('--daily', required=True, help='daily statistics CSV (format: lob_impact.vol)')
    ap.add_argument('--points', default=None,
                    help='points cache from lob_impact.points (default: <LOB_IMPACT_RESULTS>/points_<STOCK>.npz)')
    ap.add_argument('--out_dir', '--outdir', dest='out_dir', default=LOB_IMPACT_RESULTS,
                    help='output directory (default: LOB_IMPACT_RESULTS)')
    args = ap.parse_args(argv)

    models = [m for m in args.models.split(',') if m]
    if not models:
        ap.error('--models is empty')
    points = args.points or os.path.join(LOB_IMPACT_RESULTS, 'points_%s.npz' % args.stock)
    if not os.path.isfile(points):
        ap.error('points cache not found: %s (run python -m lob_impact.points first)' % points)
    if not os.path.isfile(args.daily):
        ap.error('daily statistics CSV not found: %s' % args.daily)

    sig = daily_sigmas(args.daily)
    res, rows, absent, degenerate = run(points, sig, args.stock, models)
    if not rows and not degenerate:
        raise SystemExit('none of %s is in %s (run python -m lob_impact.points for them)'
                         % (', '.join(models), points))

    os.makedirs(args.out_dir, exist_ok=True)
    md_path = os.path.join(args.out_dir, 'delta_%s.md' % args.stock)
    json_path = os.path.join(args.out_dir, 'delta_%s.json' % args.stock)
    with open(md_path, 'w') as fh:
        fh.write(markdown(args.stock, rows, absent, degenerate))
    with open(json_path, 'w') as fh:
        json.dump(dict(res, absent=absent, degenerate=degenerate), fh, indent=1)
    print('delta -> %s, %s' % (json_path, md_path))


if __name__ == '__main__':
    main()
