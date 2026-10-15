"""Visible-vs-blind attribution of metaorder impact: one run, two switches, three arms.

The same generation is repeated with two switches flipped: are the child orders applied to
the book, and are they visible to the model?

    visible  applied, visible   book mechanics + the model's reaction (the metaorder grid)
    blind    applied, hidden    book mechanics only          (stored under 'invisible-*')
    noins    no child orders    the model's unconditional drift

For each arm (visible buy/sell, blind buy/sell, noins) and an evenly strided sample of its
rollouts, the mid-price move in trade direction is measured in ticks from row 0:

* the trajectory on a common progress grid (GRID01: 241 points over the rollout);
* the final displacement (last row, or the mean of the last FINAL_TAIL rows if the last row
  has no valid mid), aggregated nan-robustly so a rollout whose book dies does not poison
  the arm;
* the mechanical part, sum over children of mid[child row] - mid[row before];
* the drift part, what moved between children: row 0 to the row before the first child,
  then each child to the row before the next one (the tail after the last child is not
  included);
* the per-child response R(m) = mid m rows after a child minus mid just before it, for
  m = 1..M_MAX, cut before the next child so its impact never leaks in; pooled over all
  children with streaming sums, reported where more than R_MIN_EVENTS children contribute;
* the mean bid-ask spread on the same progress grid.

The visible/blind ratio is the mean visible final displacement (buy and sell averaged)
over the mean blind mechanical part. It is reported only when the visible arm clears
DETECTION_SE standard errors; below that the visible mean is noise and so is the ratio.

A model that cannot condition on the injected child orders (a replay or parametric
baseline, --baselines) has no separate blind generation: visible and blind coincide by
construction.
For such a model, --blind_analog BASE=OTHER reads the blind arm from OTHER's grid rollouts,
e.g. the reference replay `Historic`, which applies the child orders with no reaction rule.

Inputs
------
    visible  <grid>/<STOCK>-<MODEL>-beta/{buy,sell}/
    blind    <controls>/<blind_sub>/<STOCK>-<MODEL>-beta/{buy,sell}/
    noins    <controls>/<noins_sub>/<STOCK>-<MODEL>-beta/buy/
--grid defaults to LOBI_GRID, --controls to LOBI_CONTROLS, --blind_sub to 'invisible' and
--noins_sub to 'noins'. Each side directory holds either
data_gen/ itself or exp_* directories that do; empty exp_* directories are ignored and an
arm spread over several exp_* directories is read as a whole.
    data_gen/*message*gen*.csv  with the matching orderbook file next to it ('message'
                                replaced by 'orderbook' in the file name); day as _YYYY-MM-DD_
    aggressive_indices_<YYYY-MM-DD>.csv, else aggressive_indices.csv, next to data_gen/:
                                0-based child-order rows, one integer per line
Orderbook CSV: headerless LOBSTER, column 0 best ask, column 2 best bid, integer price units
on a TICK-unit grid; a price >= 2147483647 or <= 0 marks an empty side. The message CSV is
used only for its row count (rows beyond the shorter of the two files are ignored).
Optional inputs: --emp_json FILE, any JSON (e.g. a response measured on real executions),
copied into numbers.json as 'empirical'; --cond_spread, the mean spread of the real
conditioning windows <visible-buy exp>/data_cond/*orderbook*.csv, stored as 'hist_spread'.

Outputs
-------
<out_dir>/<MODEL>/numbers.json   (out_dir default: LOBI_RESULTS/triangle/<STOCK>)
    model, stock, baseline, invisible_analog [, empirical] [, hist_spread], and per arm
    'visible-buy', 'visible-sell', 'invisible-buy', 'invisible-sell', 'noins':
      n                     rollouts read
      traj_mean, traj_se    mean trajectory and its SE on GRID01, ticks
      spr_mean              mean spread on GRID01, ticks
      final_mean, final_se  final displacement, ticks
      mech_mean             mechanical part, ticks (0.0 for an arm without children)
      drift_mean            between-children part, ticks (final_mean without children)
      min_gap               smallest gap between consecutive children, rows (null if none)
      resp_mean, resp_se    R(m), m = 0..M_MAX (index 0 is NaN), ticks   [arms with children]
      resp_n_events         children contributing at m = 1               [arms with children]
      spread_first, spread_last  mean spread over the first / last SPREAD_EDGE grid points
<out_dir>/<MODEL>/trajectories.png   mean trajectory per arm (skipped with --no_plot)
The visible/blind ratio is printed, not stored. Undefined values are written as the bare
token NaN, which Python's json module reads but strict JSON parsers reject.

Usage
-----
    python -m lobi.triangle build --stock XYZ --models MyModel
    python -m lobi.triangle build --stock XYZ --models Historic,MyBaseline \\
        --baselines Historic,MyBaseline --blind_analog MyBaseline=Historic
    python -m lobi.triangle ratio --stock XYZ --models MyModel
"""
import argparse
import csv
import glob
import json
import os
import re

import numpy as np

from lobi.paths import LOBI_CONTROLS, LOBI_GRID, LOBI_RESULTS

try:
    import pandas as _pd
except ImportError:          # pragma: no cover - csv fallback below
    _pd = None

TICK = 100                   # price units per tick (LOBSTER prices are 1e-4 USD; 1 cent)
SENT = 2147483647            # price the simulator writes for an empty book side
GRID01 = np.linspace(0.0, 1.0, 241)   # common progress grid for rollouts of any length
M_MAX = 131                  # longest per-child response horizon, rows
R_MIN_EVENTS = 10            # R(m) is reported only where more children than this contribute
FINAL_TAIL = 20              # rows averaged when the last row has no valid mid
SPREAD_EDGE = 24             # grid points averaged for spread_first / spread_last
DETECTION_SE = 3             # visible arm must clear this many SE for the ratio to be reported
ARMS = ('visible-buy', 'visible-sell', 'invisible-buy', 'invisible-sell', 'noins')


def read_csv_np(f):
    if _pd is not None:
        return _pd.read_csv(f, header=None, dtype=float).to_numpy()
    with open(f) as fh:
        return np.array([r for r in csv.reader(fh) if r], dtype=float)


def discover_exp(side_dir):
    """Directory to read an arm from: the single exp_* directory holding rollouts; the side
    directory itself if several exp_* directories hold rollouts (an arm generated in
    slices, read as a whole) or if it holds data_gen/ directly; else None."""
    nonempty = [d for d in sorted(glob.glob(os.path.join(side_dir, 'exp_*')))
                if glob.glob(os.path.join(d, 'data_gen', '*orderbook*gen*.csv'))]
    if len(nonempty) == 1:
        return nonempty[0]
    if len(nonempty) > 1:
        return side_dir
    if os.path.isdir(os.path.join(side_dir, 'data_gen')):
        return side_dir
    return None


def aggr_for(exp, date, L):
    """Child-order rows (< L) for one day: aggressive_indices_<date>.csv, else the shared
    aggressive_indices.csv, else none."""
    f = os.path.join(exp, 'aggressive_indices_%s.csv' % date)
    if not os.path.exists(f):
        f = os.path.join(exp, 'aggressive_indices.csv')
    if not os.path.exists(f):
        return np.array([], dtype=int)
    a = np.loadtxt(f, dtype=int, ndmin=1)
    return a[a < L]


def book_mid_spread(b):
    """Mid (price units) and spread (ticks) per row; NaN where a side is empty."""
    ask, bid = b[:, 0], b[:, 2]
    bad = (ask >= SENT) | (bid >= SENT) | (ask <= 0) | (bid <= 0)
    mid = (ask + bid) / 2.0
    spr = (ask - bid) / TICK
    mid[bad] = np.nan
    spr[bad] = np.nan
    return mid, spr


def load_regime(exp, sign, n_samples, with_insertions, m_max=M_MAX):
    """Aggregates of one arm (see the module docstring for every field).

    Rollouts are taken with an even stride so that at most n_samples are read; a rollout
    whose first row has no valid mid is skipped."""
    gens = sorted(glob.glob(os.path.join(exp, 'data_gen', '*message*gen*.csv')))
    if not gens:      # arm split across exp_* directories: walk all of them
        gens = sorted(glob.glob(os.path.join(exp, '**', 'data_gen', '*message*gen*.csv'),
                                recursive=True))
    step = max(len(gens) // n_samples, 1)
    trajs, sprs, mechs, drifts, finals = [], [], [], [], []
    # streaming accumulators for R(m): pooled children do not fit in memory as lists
    r_sum = np.zeros(m_max + 1)
    r_sumsq = np.zeros(m_max + 1)
    r_cnt = np.zeros(m_max + 1, dtype=np.int64)
    min_gap = 10 ** 9
    for mf in gens[::step][:n_samples]:
        bf = os.path.join(os.path.dirname(mf),
                          os.path.basename(mf).replace('message', 'orderbook'))
        if not os.path.exists(bf):
            continue
        m_date = re.search(r'_(\d{4}-\d{2}-\d{2})_', os.path.basename(mf))
        date = m_date.group(1) if m_date else ''
        msg = read_csv_np(mf)
        book = read_csv_np(bf)
        L = min(len(msg), len(book))
        book = book[:L]
        mid, spr = book_mid_spread(book)
        if np.isnan(mid[0]):
            continue
        x = np.arange(L) / max(L - 1, 1)
        ok = ~np.isnan(mid)
        trajs.append(np.interp(GRID01, x[ok], (sign * (mid[ok] - mid[0]) / TICK)))
        oks = ~np.isnan(spr)
        sprs.append(np.interp(GRID01, x[oks], spr[oks]))
        # child rows live next to each rollout's own data_gen/, so a split arm reads the
        # right day's file
        ai = aggr_for(os.path.dirname(os.path.dirname(mf)), date, L)
        ai = ai[ai > 0]
        if with_insertions and len(ai) > 1:
            min_gap = min(min_gap, int(np.diff(ai).min()))
            mech = np.nansum(mid[ai] - mid[ai - 1]) / TICK
            segs = np.concatenate([[mid[ai[0] - 1] - mid[0]],
                                   mid[ai[1:] - 1] - mid[ai[:-1]]])
            mechs.append(sign * mech)
            drifts.append(sign * np.nansum(segs) / TICK)
            nxt = np.append(ai[1:], L)
            for i, nx in zip(ai, nxt):
                base = mid[i - 1]
                if np.isnan(base):
                    continue
                # stop before the NEXT child so its impact never leaks into R(m)
                hi = min(i - 1 + m_max, nx - 1, L - 1)
                seg = sign * (mid[i:hi + 1] - base) / TICK        # values at m = 1..len(seg)
                ok = ~np.isnan(seg)
                mm = np.arange(1, len(seg) + 1)[ok]
                v = seg[ok]
                r_sum[mm] += v
                r_sumsq[mm] += v * v
                r_cnt[mm] += 1
        end = mid[-1] if not np.isnan(mid[-1]) else np.nanmean(mid[-FINAL_TAIL:])
        finals.append(sign * (end - mid[0]) / TICK)
    trajs, sprs = np.array(trajs), np.array(sprs)
    n = len(trajs)
    if n == 0:
        raise SystemExit('%s: no usable rollout (a message/orderbook pair whose first row has '
                         'a valid mid); check the layout (python -m pydoc lobi.triangle)' % exp)
    # a rollout whose book dies at the end leaves final = NaN; aggregate nan-robustly
    finals = np.asarray(finals, float)
    nf = max(int(np.isfinite(finals).sum()), 1)
    out = {
        'n': n,
        'traj_mean': trajs.mean(0), 'traj_se': trajs.std(0) / max(np.sqrt(n), 1),
        'spr_mean': sprs.mean(0),
        'final_mean': float(np.nanmean(finals)), 'final_se': float(np.nanstd(finals) / np.sqrt(nf)),
        'mech_mean': float(np.nanmean(mechs)) if mechs else 0.0,
        'drift_mean': float(np.nanmean(drifts)) if drifts else float(np.nanmean(finals)),
        'min_gap': min_gap if min_gap < 10 ** 9 else None,
    }
    if with_insertions:
        cnt = np.maximum(r_cnt, 1)
        mean = r_sum / cnt
        var = np.maximum(r_sumsq / cnt - mean ** 2, 0.0)
        enough = r_cnt > R_MIN_EVENTS
        r_mean = np.where(enough, mean, np.nan)
        r_se = np.where(enough, np.sqrt(var / cnt), np.nan)
        r_mean[0] = r_se[0] = np.nan
        out['resp_mean'], out['resp_se'] = r_mean, r_se
        out['resp_n_events'] = int(r_cnt[1])
    return out


def load_hist_spread(exp, n=64):
    """Mean spread (ticks) of the real conditioning windows, <exp>/data_cond/*orderbook*.csv,
    evenly strided to at most n files; NaN if there are none."""
    books = sorted(glob.glob(os.path.join(exp, 'data_cond', '*orderbook*.csv')))
    step = max(len(books) // n, 1)
    vals = []
    for bf in books[::step][:n]:
        _, spr = book_mid_spread(read_csv_np(bf))
        vals.append(np.nanmean(spr))
    return float(np.nanmean(vals)) if vals else float('nan')


def arm_spec(grid, controls, stock, model, baseline=False, analog=None,
             noins_sub='noins', blind_sub='invisible'):
    """arm -> (side directory, trade sign, has child orders)."""
    scen = '%s-%s-beta' % (stock, model)
    spec = {
        'visible-buy': (os.path.join(grid, scen, 'buy'), +1, True),
        'visible-sell': (os.path.join(grid, scen, 'sell'), -1, True),
        'noins': (os.path.join(controls, noins_sub, scen, 'buy'), +1, False),
    }
    if not baseline:
        spec['invisible-buy'] = (os.path.join(controls, blind_sub, scen, 'buy'), +1, True)
        spec['invisible-sell'] = (os.path.join(controls, blind_sub, scen, 'sell'), -1, True)
    elif analog:
        ana_scen = '%s-%s-beta' % (stock, analog)
        spec['invisible-buy'] = (os.path.join(grid, ana_scen, 'buy'), +1, True)
        spec['invisible-sell'] = (os.path.join(grid, ana_scen, 'sell'), -1, True)
    return spec


def load_json(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except (IOError, ValueError):
        return None


def vis_blind(numbers):
    """Visible/blind ratio from a numbers.json (path or parsed dict), or None.

    vis  = mean of visible-buy and visible-sell final_mean;
    mech = mean of invisible-buy and invisible-sell mech_mean;
    None when an input is missing, when |vis| < DETECTION_SE * se with
    se = sqrt(se_buy^2 + se_sell^2) / 2 of the visible final means, or when mech == 0."""
    d = load_json(numbers) if isinstance(numbers, (str, os.PathLike)) else numbers
    if not d:
        return None

    def g(arm, field):
        o = d.get(arm) or {}
        v = o.get(field)
        return float(v) if isinstance(v, (int, float)) else None
    vb, vs = g('visible-buy', 'final_mean'), g('visible-sell', 'final_mean')
    mb, ms = g('invisible-buy', 'mech_mean'), g('invisible-sell', 'mech_mean')
    if None in (vb, vs, mb, ms):
        return None
    eb, es = g('visible-buy', 'final_se'), g('visible-sell', 'final_se')
    if None not in (eb, es):
        se = ((eb ** 2 + es ** 2) ** 0.5) / 2.0
        if se > 0 and abs((vb + vs) / 2.0) < DETECTION_SE * se:
            return None
    vis, mech = (vb + vs) / 2.0, (mb + ms) / 2.0
    return float(vis / mech) if mech else None


def plot_trajectories(path, R, model):
    """Mean trajectory per arm with a 2-SE band."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    colours = {'visible': 'C0', 'invisible': 'C1', 'noins': 'C2'}
    names = {'visible': 'visible', 'invisible': 'blind', 'noins': 'no child orders'}
    fig, ax = plt.subplots(figsize=(9.4, 4.9))
    for key in ARMS:
        if key not in R:
            continue
        r = R[key]
        kind, _, side = key.partition('-')
        ls = '--' if side == 'sell' else '-'
        lab = names[kind] + (', ' + side if side else '')
        ax.fill_between(GRID01 * 100, r['traj_mean'] - 2 * r['traj_se'],
                        r['traj_mean'] + 2 * r['traj_se'], color=colours[kind], alpha=0.13, lw=0)
        ax.plot(GRID01 * 100, r['traj_mean'], color=colours[kind], ls=ls, lw=2.0,
                label='%s (n=%d, end %+.1f)' % (lab, r['n'], r['traj_mean'][-1]))
    ax.axhline(0, color='0.6', lw=1.0)
    ax.set_xlim(0, 100)
    ax.set_xlabel('rollout progress, %')
    ax.set_ylabel('mid-price move in trade direction, ticks')
    ax.set_title('%s - mean impact trajectory by arm' % model)
    ax.legend(loc='upper left', fontsize=8.5, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _models(text):
    return [m for m in (text or '').split(',') if m]


def _analogs(text):
    out = {}
    for item in _models(text):
        base, sep, other = item.partition('=')
        if not sep or not base or not other:
            raise SystemExit('--blind_analog expects BASE=OTHER pairs, got %r' % item)
        out[base] = other
    return out


def _ratio_line(model, numbers):
    r = vis_blind(numbers)
    if r is None:
        return '%s: visible/blind ratio not reported (missing arm, visible arm below %d SE, ' \
               'or zero blind mechanics)' % (model, DETECTION_SE)
    return '%s: visible/blind ratio = %.6g' % (model, r)


def build_model(args, model, baseline, analog, out_dir):
    spec = arm_spec(args.grid, args.controls, args.stock, model, baseline, analog,
                    args.noins_sub, args.blind_sub)
    R = {}
    for key, (d, sign, ins) in spec.items():
        exp = discover_exp(d)
        if not exp:
            raise SystemExit('missing arm directory: %s' % d)
        print('loading %s <- %s' % (key, exp))
        R[key] = load_regime(exp, sign, args.n_samples, ins)
        print('  n=%d final=%+.1f +/- %.1f mech=%+.1f drift=%+.1f'
              % (R[key]['n'], R[key]['final_mean'], R[key]['final_se'],
                 R[key]['mech_mean'], R[key]['drift_mean']))
    num = {}
    if args.emp_json:
        with open(args.emp_json) as fh:
            num['empirical'] = json.load(fh)
    num['model'] = model
    num['stock'] = args.stock
    if args.cond_spread:
        hist_spread = load_hist_spread(discover_exp(spec['visible-buy'][0]), args.n_samples)
        print('real conditioning windows (data_cond): mean spread %.2f ticks' % hist_spread)
        num['hist_spread'] = hist_spread
    num['baseline'] = baseline
    num['invisible_analog'] = analog
    for k, r in R.items():
        num[k] = {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv)
                  for kk, vv in r.items()}
        num[k]['spread_first'] = float(np.mean(r['spr_mean'][:SPREAD_EDGE]))
        num[k]['spread_last'] = float(np.mean(r['spr_mean'][-SPREAD_EDGE:]))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'numbers.json')
    with open(path, 'w') as fo:
        json.dump(num, fo, indent=1)
    print('saved -> %s' % path)
    print(_ratio_line(model, path))
    if not args.no_plot:
        try:
            plot_trajectories(os.path.join(out_dir, 'trajectories.png'), R, model)
        except Exception as exc:          # the plot is a diagnostic; numbers are written
            print('plot skipped: %s: %s' % (type(exc).__name__, exc))


def build(args):
    models = _models(args.models)
    baselines = set(_models(args.baselines))
    analogs = _analogs(args.blind_analog)
    stray = sorted(set(analogs) - baselines)
    if stray:
        raise SystemExit('--blind_analog applies to --baselines only: %s' % ','.join(stray))
    root = args.out_dir or os.path.join(LOBI_RESULTS, 'triangle', args.stock)
    for model in models:
        build_model(args, model, model in baselines, analogs.get(model),
                    os.path.join(root, model))


def ratio(args):
    if args.numbers:
        pairs = [(None, path) for path in args.numbers]
    else:
        root = args.out_dir or os.path.join(LOBI_RESULTS, 'triangle', args.stock)
        pairs = [(m, os.path.join(root, m, 'numbers.json')) for m in _models(args.models)]
    for model, path in pairs:
        d = load_json(path)
        if d is None:
            print('%s: cannot read %s' % (model or path, path))
            continue
        print(_ratio_line(model or d.get('model', path), d))


def main(argv=None):
    ap = argparse.ArgumentParser(prog='python -m lobi.triangle',
                                 description=__doc__.split('\n')[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)

    b = sub.add_parser('build', help='read the three arms, write numbers.json per model')
    b.add_argument('--stock', required=True, help='ticker as it appears in the directory names')
    b.add_argument('--models', required=True, help='comma-separated model names')
    b.add_argument('--baselines', default='',
                   help='comma-separated models that cannot condition on the injected child '
                        'orders (no blind arm is read for them)')
    b.add_argument('--blind_analog', default='',
                   help='comma-separated BASE=OTHER: blind arm of baseline BASE read from '
                        "OTHER's grid rollouts (e.g. MyBaseline=Historic)")
    b.add_argument('--grid', default=LOBI_GRID, help='root of the visible (metaorder) rollouts')
    b.add_argument('--controls', default=LOBI_CONTROLS,
                   help='root holding the no-child-order and blind rollouts')
    b.add_argument('--noins_sub', default='noins',
                   help='subdirectory of --controls with the no-child-order arm')
    b.add_argument('--blind_sub', default='invisible',
                   help='subdirectory of --controls with the blind arm')
    b.add_argument('--n_samples', type=int, default=64, help='rollouts read per arm, at most')
    b.add_argument('--emp_json', default=None,
                   help="optional JSON copied into numbers.json as 'empirical'")
    b.add_argument('--cond_spread', action='store_true',
                   help="also store 'hist_spread', the mean spread of the data_cond windows")
    b.add_argument('--out_dir', '--outdir', dest='out_dir', default=None,
                   help='default LOBI_RESULTS/triangle/<STOCK>; one sub-directory per model')
    b.add_argument('--no_plot', action='store_true', help='write numbers.json only')

    r = sub.add_parser('ratio', help='print the visible/blind ratio from numbers.json files')
    r.add_argument('--stock', default=None, help='ticker, to locate the numbers.json files')
    r.add_argument('--models', default='', help='comma-separated model names')
    r.add_argument('--out_dir', '--outdir', dest='out_dir', default=None, help='as for build')
    r.add_argument('--numbers', nargs='+', default=None,
                   help='explicit numbers.json paths instead of --stock/--models')

    args = ap.parse_args(argv)
    if args.cmd == 'build':
        build(args)
    else:
        if not args.numbers and not (args.stock and args.models):
            ap.error('ratio needs --stock and --models, or --numbers')
        ratio(args)


if __name__ == '__main__':
    main()
