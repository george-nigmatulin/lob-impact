"""Phase 1 free drift: mean end-to-end mid-price displacement of free-running rollouts, in ticks.

With no metaorder injected, a generator should not trend on its own; the free drift is
the baseline against which any measured response has to be read. For every arm (the
reference arm and each model) and every selected rollout:

  mid   = (best ask + best bid) / 2, rows with an empty side (price >= SENT or <= 0) dropped
  drift = (last finite mid - first finite mid) / TICK

A rollout with fewer than 100 orderbook rows, or fewer than 100 finite mids, gives no
value. Rollouts whose |drift| exceeds 5000 ticks are treated as a collapsed book and
excluded, so that a single broken rollout cannot dominate the mean. The table reports
the mean over the remaining rollouts and its standard error (population std / sqrt(n)).
Buy- and sell-side folders are pooled: nothing is injected, so the side label carries
no information. Files are spread over trading days by flow_balance.stratify (see its
docstring for the selection rule and its limitation when n_files < number of days).

Inputs (headerless orderbook CSVs: ask_price_1, ask_size_1, bid_price_1, bid_size_1, ...;
prices in integer units with TICK units per tick)
  <controls>/<noins_sub>/<STOCK>-<MODEL>-beta/{buy,sell}/**/data_gen/*orderbook*gen*.csv
  The YYYY-MM-DD date in the file name identifies the trading day. <controls> defaults
  to LOBI_CONTROLS, <noins_sub> to 'noins'; the reference arm is read from --ref_sub,
  which defaults to <noins_sub>. Rows whose first four cells do not parse as numbers
  are skipped.

Output
  <LOBI_RESULTS>/drift_<STOCK><tag>.md (or in --outdir)
    | model | drift, ticks | s.e. | n rollouts | days |
  model labels are printed with '_' replaced by '-'; 'days' counts the distinct trading
  days of the selected rollouts over both side folders; a model with no usable rollout
  gets 'n/a'. The tool exits with an error when no model has a usable rollout.

Example
  python -m lobi.drift --stock XYZ --models MyModel,MyBaseline
"""
import argparse
import csv
import glob
import os

import numpy as np

from lobi.flow_balance import days_of, stratify
from lobi.paths import LOBI_CONTROLS, LOBI_RESULTS

TICK = 100.0               # price units per tick
SENT = 2147483647          # price written for an empty book side (int32 max)
MIN_ROWS = 100             # minimum orderbook rows, and finite mids, for a usable rollout
COLLAPSE_TICKS = 5000.0    # |drift| above this is a collapsed book, excluded from the mean
SIDES = ('buy', 'sell')


def read_ob(f):
    """First four columns (best ask/bid price and size) of an orderbook CSV."""
    rows = []
    with open(f) as fh:
        for r in csv.reader(fh):
            if not r:
                continue
            try:
                rows.append([float(x) for x in r[:4]])
            except ValueError:
                continue          # skip a row with a corrupt cell, keep the rest of the file
    return np.array(rows, float) if rows else np.empty((0, 4))


def drift_one(f):
    """End-to-end mid displacement of one rollout in ticks, or NaN if unusable."""
    ob = read_ob(f)
    if ob.ndim != 2 or ob.shape[0] < MIN_ROWS:
        return np.nan
    ask, bid = ob[:, 0], ob[:, 2]
    mid = (ask + bid) / 2.0
    mid[(ask >= SENT) | (bid >= SENT) | (ask <= 0) | (bid <= 0)] = np.nan
    fin = np.flatnonzero(np.isfinite(mid))
    if len(fin) < MIN_ROWS:
        return np.nan
    return (mid[fin[-1]] - mid[fin[0]]) / TICK


def model_drifts(root, stock, model, n_files):
    """Per-rollout drifts (both sides, unfiltered) and the number of distinct trading days."""
    vals, days = [], set()
    for side in SIDES:
        fs = stratify(sorted(glob.glob(os.path.join(
            root, f'{stock}-{model}-beta', side, '**', 'data_gen', '*orderbook*gen*.csv'),
            recursive=True)), n_files)
        days |= days_of(fs)
        for f in fs:
            vals.append(drift_one(f))
    return np.array(vals, float), len(days)


def summarise(model, v, nd):
    """(Markdown row, rollouts used) for one model after dropping NaN and collapsed rollouts."""
    v = v[np.isfinite(v)]
    bad = np.abs(v) > COLLAPSE_TICKS
    if bad.any():
        print(f'  [collapse] {model}: {int(bad.sum())}/{len(v)} rollouts dropped '
              f'(extreme {v[np.argmax(np.abs(v))]:+.0f} ticks)', flush=True)
        v = v[~bad]
    label = model.replace('_', '-')
    if not len(v):
        print(f'{model:11s} no data')
        return f'| {label} | n/a | | 0 | {nd} |', 0
    mu, se = float(np.mean(v)), float(np.std(v) / np.sqrt(len(v)))
    print(f'{model:11s} drift {mu:+8.1f} ticks  (s.e. {se:.1f}, n={len(v)}, days={nd})',
          flush=True)
    return f'| {label} | {mu:+.0f} | {se:.1f} | {len(v)} | {nd} |', len(v)


def model_list(models, ref):
    """Labels from --models; the reference arm goes first unless it is already listed."""
    out = [x for x in models.split(',') if x]
    if ref and ref not in out:
        out = [ref] + out
    return out


def parse_args(argv):
    ap = argparse.ArgumentParser(
        prog='python -m lobi.drift',
        description='Free drift (mean end-to-end mid displacement, ticks) of '
                    'free-running rollouts.')
    ap.add_argument('--stock', required=True, help='ticker used in the folder names')
    ap.add_argument('--models', required=True,
                    help='comma-separated model labels (folders <STOCK>-<MODEL>-beta)')
    ap.add_argument('--ref', default='Historic',
                    help='reference arm label, put first in the table unless already in '
                         "--models; pass '' to leave it out")
    ap.add_argument('--n_files', type=int, default=64, help='per model per side')
    ap.add_argument('--controls', default=LOBI_CONTROLS,
                    help='root of the free-generation rollouts (default: LOBI_CONTROLS)')
    ap.add_argument('--noins_sub', default='noins',
                    help='subdirectory of --controls to read')
    ap.add_argument('--ref_sub', default=None,
                    help='subdirectory holding the reference arm (default: --noins_sub)')
    ap.add_argument('--tag', default='', help='output filename suffix')
    ap.add_argument('--outdir', '--out_dir', dest='outdir', default=None,
                    help='output directory (default: LOBI_RESULTS)')
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    root = os.path.join(args.controls, args.noins_sub)
    ref_root = os.path.join(args.controls, args.ref_sub or args.noins_sub)
    out_lines = [f'# Free drift - {args.stock} (mean end displacement of noins rollouts, ticks)',
                 '', '| model | drift, ticks | s.e. | n rollouts | days |',
                 '|---|---|---|---|---|']
    used = 0
    for m in model_list(args.models, args.ref):
        v, nd = model_drifts(ref_root if m == args.ref else root, args.stock, m, args.n_files)
        row, n = summarise(m, v, nd)
        out_lines.append(row)
        used += n
    if not used:
        raise SystemExit('no usable rollouts for %s under %s: check --stock, --models and the '
                         'layout (python -m pydoc lobi.drift)' % (args.stock, args.controls))

    outdir = args.outdir or LOBI_RESULTS
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, f'drift_{args.stock}{args.tag}.md')
    with open(out, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(out_lines) + '\n')
    print('\nsaved ->', out)
    return 0


if __name__ == '__main__':
    main()
