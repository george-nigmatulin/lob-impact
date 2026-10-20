"""Impact master curve: mean signed impact on the executed-volume clock, normalised at execution end.

For every metaorder rollout the signed relative mid-price impact (bps) is sampled at the
child-order boundaries k = 1..n_ins and then at the ends of the cooling windows, giving
one value per point of the grid v = k / n_ins (v = 1 is the end of execution). Sampling
at boundaries needs no interpolation, and every rollout has the same length. Averaging
first and normalising second,

    master(v) = <I(v)> / <I(1)>,

avoids the blow-ups of a per-rollout I / I_peak when a rollout's own peak is near zero.

  * shape 'beta'        100 children, no cooling: v in [0, 1], the build-up.
  * any other shape     10 children, then 100 cooling windows: v in [0, 11], build-up
                        followed by relaxation (lob_impact.decay_shape reads this cache).

Gates, per model:
  * a rollout whose |I| exceeds COLLAPSE_BPS anywhere is dropped as a collapsed book;
  * a grid point with fewer than max(MIN_COUNT, MIN_FRAC * rollouts) finite values is NaN;
  * the curve is normalisable only if <I(1)> is finite and not ~0; it is flagged
    significant (sig) when <I(1)> >= 3 standard errors and positive. A negative or
    insignificant denominator turns the ratio into noise, so such curves are reported
    but not drawn.

Inputs (headerless LOBSTER CSVs; see lob_impact.points for the full file contract)
  <grid>/<STOCK>-<MODEL>-<shape>/{buy,sell}/**/data_gen/*orderbook*gen*.csv
  <grid>/<STOCK>-<MODEL>-<shape>/{buy,sell}/**/aggressive_indices_<YYYY-MM-DD>.csv
  The day is the '_YYYY-MM-DD_' in the orderbook file name; buy rollouts enter with sign
  +1, sell rollouts with -1. The reference mid is the row before the first child; the
  cooling-window length is the smallest gap between child indices.

Output: <LOB_IMPACT_RESULTS>/master_curve/master_curve_<STOCK>_<shape>.npz (or --out stem)
  vgrid                 grid v = k / n_ins, k = 0..n_ins + n_cool
  <model>_master        master(v)
  <model>_peak          <I(1)>, bps
  <model>_peak_se       its standard error, bps
  <model>_n             rollouts collected (before the collapse filter)
  <model>_sig           <I(1)> positive and >= 3 standard errors
  plus <stem>.png, a diagnostic figure (skipped with --no_plot).

Usage:
    python -m lob_impact.master_curve --stock XYZ --models MyModel,Baseline --shape relaxation
"""
import argparse
import csv
import glob
import os
import re

import numpy as np

from lob_impact.paths import LOB_IMPACT_GRID, LOB_IMPACT_RESULTS

SENTINEL = 2147483647       # price the simulator writes for an empty book side
DATE_RE = re.compile(r'_(\d{4}-\d{2}-\d{2})_')
COLLAPSE_BPS = 2000.0       # |I| above this anywhere in a rollout marks a collapsed book
MIN_COUNT, MIN_FRAC = 30, 0.3   # a grid point needs this many finite rollouts
PALETTE = ('#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
           '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf')


def shape_counts(shape):
    """(children, cooling windows) of a shape: 'beta' is build-up only."""
    return (100, 0) if shape == 'beta' else (10, 100)


def _read(f):
    with open(f) as fh:
        return np.array([r for r in csv.reader(fh) if r], dtype=float)


def _aggr_by_day(side_dir):
    """day -> child-order row indices, from the per-day aggressive_indices_<day>.csv files."""
    out = {}
    for f in sorted(glob.glob(os.path.join(side_dir, '**', 'aggressive_indices_*.csv'),
                              recursive=True)):
        d = os.path.basename(f)[len('aggressive_indices_'):-len('.csv')]
        if re.fullmatch(r'\d{4}-\d{2}-\d{2}', d):
            out[d] = np.loadtxt(f, dtype=int, ndmin=1)
    return out


def collect_paths(side_dir, sign, n_ins, n_cool):
    """Per rollout: signed impact in bps at the child boundaries, then the cooling-window ends.

    Returns a list of arrays of length n_ins + n_cool + 1 (index 0 is the origin, 0 bps).
    Cooling-window ends are spaced by the smallest gap between child indices; a final
    boundary that falls just past the end of the file is moved to the last row. Rollouts
    that end before their first child, or have fewer than 5 finite boundary values, are
    dropped.
    """
    ab = _aggr_by_day(side_dir)
    obs = sorted(glob.glob(os.path.join(side_dir, '**', 'data_gen', '*orderbook*gen*.csv'),
                           recursive=True))
    nb = n_ins + n_cool
    out = []
    for ob in obs:
        m = DATE_RE.search(os.path.basename(ob))
        if not m:
            continue
        aggr = ab.get(m.group(1))
        if aggr is None or len(aggr) < 2 or aggr[0] < 1:
            continue
        a = _read(ob)
        if a.ndim != 2 or a.shape[1] < 4:
            continue
        ask, bid = a[:, 0].copy(), a[:, 2].copy()
        bad = (ask >= SENTINEL) | (bid >= SENTINEL) | (ask <= 0) | (bid <= 0)
        mid = (ask + bid) / 2.0
        mid[bad] = np.nan
        if aggr[0] > len(mid):                  # the rollout ends before its first child
            continue
        ref = mid[aggr[0] - 1]
        if not np.isfinite(ref) or ref <= 0:
            continue
        I = sign * (mid - ref) / ref * 1e4      # bps
        mb = int(np.diff(aggr).min())
        vals = np.full(nb + 1, np.nan)
        vals[0] = 0.0
        idx = list(aggr[:n_ins])
        last = aggr[min(n_ins, len(aggr)) - 1]
        if n_cool and mb > 0:
            idx += [int(last + j * mb) for j in range(1, n_cool + 1)]
        for j, ix in enumerate(idx, start=1):
            if ix >= len(I) and mb > 0 and ix - (len(I) - 1) <= mb:
                ix = len(I) - 1                 # final boundary lands exactly at the array end
            if ix < len(I):
                vals[j] = I[ix]
        if np.isfinite(vals[1:]).sum() < 5:
            continue
        out.append(vals)
    return out


def model_curve(grid, stock, model, shape, n_ins, n_cool):
    """Master curve of one model, or None when it has no data or <I(1)> is ~0."""
    exp_dir = os.path.join(grid, f'{stock}-{model}-{shape}')
    paths = collect_paths(os.path.join(exp_dir, 'buy'), +1, n_ins, n_cool) + \
        collect_paths(os.path.join(exp_dir, 'sell'), -1, n_ins, n_cool)
    if not paths:
        print(f'{model}: no data')
        return None
    M = np.vstack(paths)
    bad = np.nanmax(np.abs(M), axis=1) > COLLAPSE_BPS
    if bad.any():
        print(f'  [collapse] {model}: dropped {int(bad.sum())}/{len(M)} rollouts '
              f'(|I| > {COLLAPSE_BPS:.0f} bps; worst {np.nanmax(np.abs(M[bad])):.0f})', flush=True)
        M = M[~bad]
    mean = np.nanmean(M, axis=0)
    cnt = np.sum(np.isfinite(M), axis=0)
    mean[cnt < max(MIN_COUNT, MIN_FRAC * len(paths))] = np.nan
    i1 = n_ins                                  # index of v = 1, the end of execution
    peak = mean[i1]
    peak_se = float(np.nanstd(M[:, i1]) / max(np.sqrt(cnt[i1]), 1.0))
    sig = bool(np.isfinite(peak) and peak >= 3 * peak_se and peak > 1e-9)
    if not np.isfinite(peak) or abs(peak) < 1e-9:
        print(f'{model}: peak~0, skipping normalise')
        return None
    master = mean / peak
    n = len(paths)
    print(f'  {model:10s}: n={n}, peak<I(v=1)>={peak:.3f}+-{peak_se:.3f} bps '
          f'{"SIG" if sig else "NOT significant (curve not drawn)"}, '
          f'end/peak={master[np.isfinite(master)][-1]:.2f}')
    return dict(master=master, peak=peak, peak_se=peak_se, n=n, sig=sig)


def plot(stock, shape, vgrid, curves, png):
    """Diagnostic figure: significant master curves with two references, the square-root
    build-up v^0.5 and, for relaxation shapes, a propagator-shaped decay towards 2/3 of the
    peak, the permanent level that fair pricing implies for square-root impact (Farmer,
    Gerig, Lillo and Waelbroeck, "How efficiency shapes market impact", Quantitative
    Finance, 2013)."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib not available: figure skipped')
        return None
    from lob_impact.trajectory import propagator_curve
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for i, (m, c) in enumerate(curves.items()):
        color = PALETTE[i % len(PALETTE)]
        if c['sig']:
            ax.plot(vgrid, c['master'], color=color, lw=1.8, label=f"{m} (n={c['n']})")
        else:
            ax.plot([], [], color=color, lw=1.0, alpha=0.4,
                    label=f"{m} (n={c['n']}; <I(1)>={c['peak']:.2f} bps, not normalisable)")
    vb = np.linspace(0.01, 1.0, 60)
    ax.plot(vb, vb ** 0.5, color='k', ls='--', lw=1.2, label=r'square-root build-up $v^{0.5}$')
    if shape != 'beta':
        px, prop = propagator_curve(1.0, 1.0, 0.5, float(vgrid[-1]))
        ax.plot(px, 2 / 3 + prop / 3, color='0.4', ls='-.', lw=1.5,
                label=r'relaxation to $\frac{2}{3}$ of peak (propagator shape)')
    ax.axvline(1.0, color='#9a9a9a', lw=1.0, ls=':')
    ax.axhline(1.0, color='#9a9a9a', lw=0.7)
    ax.set_xlabel(r'$v$ = fraction of the metaorder executed ($v=1$: execution end)')
    ax.set_ylabel(r'$\langle I(v)\rangle / \langle I(1)\rangle$')
    ax.set_title(f'{stock}: impact master curve ({shape})')
    ax.margins(x=0)
    ax.legend(loc='upper left' if shape == 'beta' else 'lower left', fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(png, dpi=150)
    plt.close(fig)
    return png


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog='python -m lob_impact.master_curve',
        description='Mean signed impact on the executed-volume clock, normalised at the end '
                    'of execution. See the module docstring for the method and the layout.')
    ap.add_argument('--stock', required=True, help='ticker; experiment directories are '
                                                   '<STOCK>-<MODEL>-<shape>')
    ap.add_argument('--models', required=True, help='comma-separated model names')
    ap.add_argument('--shape', default='beta',
                    help="'beta' (100 children, build-up) or e.g. 'relaxation' "
                         '(10 children + 100 cooling windows)')
    ap.add_argument('--grid', default=LOB_IMPACT_GRID,
                    help='root of the metaorder-injection rollouts (default: LOB_IMPACT_GRID)')
    ap.add_argument('--out', default=None, help='output stem (default: '
                    '<LOB_IMPACT_RESULTS>/master_curve/master_curve_<STOCK>_<shape>)')
    ap.add_argument('--no_plot', action='store_true', help='skip the diagnostic figure')
    args = ap.parse_args(argv)

    models = [m for m in args.models.split(',') if m]
    if not models:
        ap.error('--models is empty')
    n_ins, n_cool = shape_counts(args.shape)
    vgrid = np.arange(n_ins + n_cool + 1) / n_ins
    print(f'=== {args.stock} {args.shape} master curve ===')

    cache, curves = {'vgrid': vgrid}, {}
    for m in models:
        c = model_curve(args.grid, args.stock, m, args.shape, n_ins, n_cool)
        if c is None:
            continue
        curves[m] = c
        cache[f'{m}_master'] = c['master']
        cache[f'{m}_peak'] = c['peak']
        cache[f'{m}_peak_se'] = c['peak_se']
        cache[f'{m}_n'] = c['n']
        cache[f'{m}_sig'] = c['sig']
    if not curves:
        raise SystemExit('no normalisable model for %s, shape %s, under %s: check --stock, '
                         '--models, --shape and the layout (python -m pydoc lob_impact.master_curve)'
                         % (args.stock, args.shape, args.grid))

    stem = args.out or os.path.join(LOB_IMPACT_RESULTS, 'master_curve',
                                    f'master_curve_{args.stock}_{args.shape}')
    stem = os.path.splitext(stem)[0]
    os.makedirs(os.path.dirname(os.path.abspath(stem)), exist_ok=True)
    np.savez(stem + '.npz', **cache)
    print(f'saved -> {stem}.npz')
    if not args.no_plot:
        plot(args.stock, args.shape, vgrid, curves, stem + '.png')


if __name__ == '__main__':
    main()
