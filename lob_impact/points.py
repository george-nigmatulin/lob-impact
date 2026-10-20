"""Collect per-child impact points from metaorder-injection rollouts into a cache.

For every rollout in which a metaorder (a sequence of aggressive child orders)
was injected, one point is recorded per child order k:

    Q_k  cumulative metaorder volume through child k: running sum of the sizes on
         the child-order message rows (shares)
    I_k  signed relative mid-price impact, sign * (mid_k - mid_ref) / mid_ref,
         with sign = +1 for a buy metaorder and -1 for a sell metaorder,
         mid_k   = mid in the orderbook row of child k (the book right after it),
         mid_ref = mid in the row just before the first child (dimensionless)
    day  trading day of the rollout (YYYY-MM-DD)
    kc   child index k, 1-based

Impact stays signed: no I > 0 selection and no outlier trimming, so a downstream
average over points is not biased by selecting on the sign of the response.
Normalisation by daily volume and volatility is applied later (lob_impact.delta), so
one cache serves every normaliser.

Input layout (under --grid, default LOB_IMPACT_GRID):

    <grid>/<STOCK>-<MODEL>-beta/buy/    buy metaorder rollouts  (sign +1)
    <grid>/<STOCK>-<MODEL>-beta/sell/   sell metaorder rollouts (sign -1)

and below each side directory, at any depth (searched recursively), e.g. one
sub-directory per run:

    .../data_gen/*orderbook*gen*.csv    one rollout, LOBSTER orderbook file, e.g.
                                        XYZ_<YYYY-MM-DD>_orderbook_real_id_7_gen_id_7.csv
    .../data_gen/*message*gen*.csv      its LOBSTER message file: the orderbook file
                                        name with 'orderbook' replaced by 'message'
    .../aggressive_indices_<YYYY-MM-DD>.csv
                                        child-order rows for the rollouts of that day

  * The day of a rollout is the first '_YYYY-MM-DD_' in the orderbook file name.
  * CSV files have no header. Orderbook rows: ask_price_1, ask_size_1,
    bid_price_1, bid_size_1, ... (at least 4 columns; level-1 prices are used).
    Message rows: time, event_type, order_id, size, price, direction (at least
    6 columns; event_type and size are used). Row i of the orderbook file is the
    book after message i. Prices in any positive unit (LOBSTER: dollars x 10000);
    a level-1 price >= 2147483647 or <= 0 marks an empty side and leaves that
    row's mid undefined. Unparseable cells are read as NaN.
  * aggressive_indices_<YYYY-MM-DD>.csv: one integer per line, the 0-based row
    index (into the message and orderbook files) of each child order, in
    execution order. All rollouts of that day under the side directory use this
    one schedule. Indices past the end of a file are ignored. Every indexed
    message row must be an execution (event_type 4) with positive size;
    otherwise the rollout is skipped with a warning. Per-day files are required
    because the injection schedule can differ between days; a file without a
    date (aggressive_indices.csv) is ignored. Keep one schedule per day under each
    side directory: with several files for the same day, the last in sorted path
    order is used.
  * A rollout is also skipped when its name has no date, its day has no indices
    file, fewer than two child indices fall inside the file, the first child is
    at row 0, mid_ref is undefined, or the message file is missing. Points with
    an undefined mid at the child row are dropped.

Output: <LOB_IMPACT_RESULTS>/points_<STOCK>.npz (or --out), per model:

    <model>_Q     float64 (n,)   Q_k, shares
    <model>_I     float64 (n,)   I_k, dimensionless
    <model>_days  str     (n,)   day, YYYY-MM-DD
    <model>_kc    int64   (n,)   k, 1-based

Points are ordered buy side first, then sell side; within a side by sorted file
path, then by k. If the output file already exists it is loaded and extended:
models already in it are kept as they are and not re-collected (delete the file
to rebuild them). Models without any usable rollout are reported and left out.

With --daily (format: lob_impact.vol) the cached days of each model are checked
against the daily statistics of --stock; points on uncovered days are dropped
by lob_impact.delta. The check does not change the cache.

Usage:
    python -m lob_impact.points --stock XYZ --models MyModel,Baseline \\
        [--grid DIR] [--out FILE.npz] [--daily daily_stats.csv]
"""
import argparse
import csv
import glob
import os
import re

import numpy as np

from lob_impact.paths import LOB_IMPACT_GRID, LOB_IMPACT_RESULTS
from lob_impact.vol import daily_sigmas

SENTINEL = 2147483647                   # int32 max: placeholder price of an empty book side
DATE_RE = re.compile(r'_(\d{4}-\d{2}-\d{2})_')


def _read(f):
    """Read a header-less numeric CSV into a float array.

    Cells that do not parse as a float (for example two integers glued together by
    a faulty writer) become NaN; the finiteness checks downstream drop them.
    """
    out = []
    with open(f) as fh:
        for r in csv.reader(fh):
            if not r:
                continue
            row = []
            for x in r:
                try:
                    row.append(float(x))
                except ValueError:
                    row.append(np.nan)
            out.append(row)
    return np.array(out, dtype=float)


def _aggr_by_day(side_dir):
    """day -> child-order row indices, from the per-day aggressive_indices_<day>.csv files.

    The per-day files are used, not a single shared aggressive_indices.csv, because
    the injection schedule can differ between days.
    """
    out = {}
    for f in sorted(glob.glob(os.path.join(side_dir, '**', 'aggressive_indices_*.csv'),
                              recursive=True)):
        d = os.path.basename(f)[len('aggressive_indices_'):-len('.csv')]
        if re.fullmatch(r'\d{4}-\d{2}-\d{2}', d):
            out[d] = np.loadtxt(f, dtype=int, ndmin=1)
    return out


def collect(side_dir, sign):
    """Per (rollout, child): (Q_k, I_k, day, k) for one side directory.

    side_dir  <grid>/<STOCK>-<MODEL>-beta/<buy|sell>
    sign      +1 for a buy metaorder, -1 for a sell metaorder
    Returns a list of tuples (Q cumulative volume, I signed impact, day str,
    k 0-based child index). Sigma and V are applied later.
    """
    obs = sorted(glob.glob(os.path.join(side_dir, '**', 'data_gen', '*orderbook*gen*.csv'),
                           recursive=True))
    aggr_by_day = _aggr_by_day(side_dir)
    if not obs or not aggr_by_day:
        return []
    rows = []
    for ob in obs:
        m = DATE_RE.search(os.path.basename(ob))
        if not m:
            continue
        day = m.group(1)
        aggr = aggr_by_day.get(day)
        if aggr is None:
            continue
        a = _read(ob)
        if a.ndim != 2 or a.shape[1] < 4:
            continue
        ask_p, bid_p = a[:, 0].copy(), a[:, 2].copy()
        bad = (ask_p >= SENTINEL) | (bid_p >= SENTINEL) | (ask_p <= 0) | (bid_p <= 0)
        mid = (ask_p + bid_p) / 2.0
        mid[bad] = np.nan
        L = len(mid)
        idx = aggr[aggr < L]
        if len(idx) < 2 or idx[0] < 1:
            continue
        ref = mid[idx[0] - 1]
        if not np.isfinite(ref) or ref <= 0:
            continue
        mf = os.path.join(os.path.dirname(ob),
                          os.path.basename(ob).replace('orderbook', 'message'))
        if not os.path.exists(mf):
            continue
        mm = _read(mf)
        if mm.ndim != 2 or mm.shape[1] < 6:
            continue
        # every child row must be an execution (event 4) with positive size; anything
        # else means the indices do not line up with this rollout
        ev, sz = mm[idx, 1], mm[idx, 3]
        if not np.all(ev == 4) or np.any(sz <= 0):
            print(f'  [WARN] skip {os.path.basename(mf)}: aggressive rows misaligned '
                  f'({np.sum(ev != 4)} non-exec, {np.sum(sz <= 0)} size<=0)')
            continue
        Q = np.cumsum(mm[idx, 3])
        for k, step in enumerate(idx):
            I = sign * (mid[step] - ref) / ref          # signed: negative responses are kept
            if np.isfinite(I) and Q[k] > 0:
                rows.append((Q[k], I, day, k))
    return rows


def collect_model(grid, stock, model):
    """All points of one model, buy then sell. Returns (Q, I, days, kc) arrays or None."""
    exp_dir = os.path.join(grid, f'{stock}-{model}-beta')
    raw = collect(os.path.join(exp_dir, 'buy'), +1) + collect(os.path.join(exp_dir, 'sell'), -1)
    if not raw:
        return None
    Q = np.array([r[0] for r in raw], float); I = np.array([r[1] for r in raw], float)
    days = np.array([r[2] for r in raw]); kc = np.array([r[3] for r in raw], int) + 1
    return Q, I, days, kc


def build_points(grid, stock, models, cache_path):
    """Create or extend the points cache at cache_path for the given models.

    Models already present in an existing cache are kept and not re-collected;
    every other key of the existing cache is carried over unchanged. The file is
    rewritten after each newly collected model.
    """
    pts = None
    if os.path.exists(cache_path):
        with np.load(cache_path, allow_pickle=False) as z:
            pts = dict(z)
        print(f'points cache loaded: {cache_path}', flush=True)
    # start from the loaded cache: the file is rewritten as a whole, so starting
    # empty would drop every cached model that is not collected in this run
    pts_out = dict(pts) if pts is not None else {}
    for model in models:
        if pts is not None and f'{model}_Q' in pts:
            Q = pts[f'{model}_Q']
        else:
            r = collect_model(grid, stock, model)
            if r is None:
                print(f'{model}: no data', flush=True)
                continue
            Q, I, days, kc = r
            pts_out[f'{model}_Q'] = Q; pts_out[f'{model}_I'] = I
            pts_out[f'{model}_days'] = days; pts_out[f'{model}_kc'] = kc
            np.savez_compressed(cache_path, **pts_out)
        print(f'{model}: {len(Q):,} points', flush=True)
    return pts_out


def report_day_coverage(cache_path, stock, models, daily_csv):
    """Print, per cached model, the days without usable daily statistics for the stock."""
    sig = daily_sigmas(daily_csv)
    if not os.path.exists(cache_path):
        print('no cache written: nothing to check')
        return
    with np.load(cache_path, allow_pickle=False) as z:
        cached = {m: z[f'{m}_days'] for m in models if f'{m}_days' in z.files}
    def usable(d):
        v = sig.get((stock, d), {}).get('V', np.nan)
        s = sig.get((stock, d), {}).get('parkinson', np.nan)
        return bool(np.isfinite(v) and v > 0 and np.isfinite(s) and s > 0)

    for model, days in cached.items():
        uniq = sorted(set(days.tolist()))
        bad = [d for d in uniq if not usable(d)]
        n_bad = int(np.isin(days, bad).sum()) if bad else 0
        msg = 'all covered' if not bad else ('%d day(s) without positive daily V and sigma, %d '
                                             'points will be dropped: %s'
                                             % (len(bad), n_bad, ', '.join(bad)))
        print(f'{model}: {len(uniq)} days, {msg}')


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog='python -m lob_impact.points',
        description='Collect per-child (Q, I, day, k) impact points from metaorder-injection '
                    'rollouts into points_<STOCK>.npz. See the module docstring for the layout.')
    ap.add_argument('--stock', required=True,
                    help='ticker; experiment directories are named <STOCK>-<MODEL>-beta')
    ap.add_argument('--models', required=True,
                    help='comma-separated model names, as they appear in the directory names')
    ap.add_argument('--grid', default=LOB_IMPACT_GRID,
                    help='root of the metaorder-injection rollouts (default: LOB_IMPACT_GRID)')
    ap.add_argument('--out', default=None,
                    help='cache file (default: <LOB_IMPACT_RESULTS>/points_<STOCK>.npz); an existing '
                         'file is extended, not replaced')
    ap.add_argument('--daily', default=None,
                    help='optional daily statistics CSV (see lob_impact.vol): report cached days it '
                         'does not cover for --stock')
    args = ap.parse_args(argv)

    models = [m for m in args.models.split(',') if m]
    if not models:
        ap.error('--models is empty')
    if not os.path.isdir(args.grid):
        ap.error(f'grid directory not found: {args.grid}')
    if args.daily and not os.path.isfile(args.daily):
        ap.error(f'daily statistics CSV not found: {args.daily}')
    out = args.out or os.path.join(LOB_IMPACT_RESULTS, f'points_{args.stock}.npz')
    if not out.endswith('.npz'):
        out += '.npz'                   # numpy appends it on save; keep reads and writes on one name
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)

    build_points(args.grid, args.stock, models, out)
    if not os.path.exists(out):
        raise SystemExit(f'no usable rollouts for {args.stock} under {args.grid}: check '
                         '--stock, --models and the layout (python -m pydoc lob_impact.points)')
    print(f'cache -> {out}')
    if args.daily:
        report_day_coverage(out, args.stock, models, args.daily)


if __name__ == '__main__':
    main()
