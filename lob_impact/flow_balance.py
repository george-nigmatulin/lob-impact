"""Phase 1 flow balance: trade-sign memory, price diffusivity and trade response of free rollouts.

Each free-running rollout (no injected metaorder) is reduced to a trade-time series: one
sign eps_t per aggressive order and the mid-price m_t right after it. Pooled over the
rollouts of one source (the reference arm or a model), three statistics are computed:

  C(l)  autocorrelation of trade signs, l = 1..300 trades, rescaled to C(1) = 1 when
        C(1) > 0 (all NaN if the rescaled curve leaves [-1, 1]; left unscaled when
        C(1) <= 0); gamma_flow = minus the slope of log C(l) against log l over
        2 <= l <= 40, using the lags where C(l) > 0.
  V(l)  mean squared mid displacement over l trades (price units squared) on a
        log-spaced lag grid; H = half the slope of log V(l) against log l over
        2 <= l <= 100. Rows with an empty book side are dropped before differencing.
  R(l)  response E[eps_t (m_{t+l} - m_t)] in ticks, l = 1..200, and the drift-corrected
        Rc(l) = E[eps dm] - E[eps] E[dm].

Why these: persistent order flow (C(l) ~ l^-gamma with 0 < gamma < 1) is compatible with
a diffusive price (H close to 1/2) only if the impact of each trade partly decays
(Bouchaud, Gefen, Potters and Wyart, "Fluctuations and response in financial markets:
the subtle nature of 'random' price changes", Quantitative Finance, 2004). Persistent
flow together with H above 1/2 means that compensating resilience is missing. 95%
intervals for gamma_flow and H come from a bootstrap over rollouts (200 resamples,
fixed seed). Rollouts with fewer than 30 execution rows are skipped.

Inputs (headerless LOBSTER-style CSV pairs, one pair per rollout)
  <controls>/<ref_sub>/<STOCK>-<REF>-beta/buy/**/data_gen/*message*gen*.csv    reference arm
  <controls>/<noins_sub>/<STOCK>-<MODEL>-beta/buy/**/data_gen/*message*gen*.csv each model
  The orderbook file of a rollout is the message file with 'message' replaced by
  'orderbook' in the file name. The YYYY-MM-DD date in the file name identifies the
  trading day.
  Message columns: time, event_type (4 = execution of a visible order), order_id, size,
  price, direction (+1 resting buy order, -1 resting sell order). Orderbook columns:
  ask_price_1, ask_size_1, bid_price_1, bid_size_1, ... Prices in integer units with
  TICK units per tick; a side priced >= SENT or <= 0 is empty.
  <controls> defaults to LOB_IMPACT_CONTROLS; <ref_sub> and <noins_sub> default to 'noins'.
  Only the buy-side folder is read: with nothing injected the side label is arbitrary.

Outputs, in <LOB_IMPACT_RESULTS>/flow_balance/ or --outdir (the reference arm is stored as 'Real')
  flow_balance_<STOCK><tag>.npz       per source <name>: <name>_C (300,), <name>_V (21,)
                                      on var_lags(), <name>_R and <name>_Rc (200,) in ticks,
                                      <name>_gamma, <name>_H (scalars),
                                      <name>_H_ci, <name>_gamma_ci (2,) 95% bootstrap CI
  flow_balance_<STOCK><tag>_stats.md  gamma_flow, H, both CIs, trading days, rollouts
  flow_balance_<STOCK><tag>.md        rollouts, trades per rollout, sign pairs at lag 100,
                                      buy share, gamma_flow and H with CIs, and a
                                      diffusivity label: balanced, SUPERDIFFUSIVE,
                                      subdiffusive, or not identified when H is undefined
  flow_balance_<STOCK><tag>.png       diagnostic figure (skipped with --no_plot)

Example
  python -m lob_impact.flow_balance --stock XYZ --models MyModel,MyBaseline --n_files 64
"""
import argparse
import csv
import glob
import os
import re

import numpy as np

from lob_impact.paths import LOB_IMPACT_CONTROLS, LOB_IMPACT_RESULTS

SENT = 2147483647          # price written for an empty book side (int32 max)
TICK = 100                 # price units per tick
DATE_RE = re.compile(r'(\d{4}-\d{2}-\d{2})')     # trading day, read from the file name
LMAX = 300                 # longest lag (trades) of the sign ACF and of the variance grid
FIT_LO, FIT_HI = 2, 40     # lag window of the power-law fit to C(l)
MIN_EXEC = 30              # rollouts with fewer execution rows are skipped
N_BOOT = 200               # bootstrap resamples over rollouts
BOOT_SEED = 7              # bootstrap seed, re-created for every source
H_BALANCE_TOL = 0.07       # |H - 0.5| below this is reported as balanced
H_SUPERDIFFUSIVE = 0.57    # H above this is reported as superdiffusive
REAL = 'Real'              # cache/table key of the reference arm
SIDE = 'buy'               # side folder read for every source

MAX_MSGS = None            # set by --max_msgs: keep only the first N messages of each rollout

PALETTE = ('#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
           '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf')
REF_COLOR = '#111111'


def stratify(files, n):
    """Return n of the (sorted) files spread evenly over trading days.

    Paths sort by date, so the first n sorted paths would cover only the earliest days.
    Take n // n_days files from every day, top up round-robin across days, and return
    the selection sorted. Files whose name carries no date form one extra group. Lists
    no longer than n, and n <= 0, return the input unchanged.

    Known limitation, kept for reproducibility: when n is smaller than the number of
    days, one file per day is taken and the sorted list is cut to n, so only the
    earliest n days are represented.
    """
    if not n or n <= 0 or len(files) <= n:
        return files
    by = {}
    for f in files:
        m = DATE_RE.search(os.path.basename(f))
        by.setdefault(m.group(1) if m else 'na', []).append(f)
    days = sorted(by)
    per = max(1, n // len(days))
    out = [f for d in days for f in by[d][:per]]
    k = per
    while len(out) < n:
        added = False
        for d in days:
            if len(out) >= n:
                break
            if k < len(by[d]):
                out.append(by[d][k]); added = True
        if not added:
            break
        k += 1
    return sorted(out)[:n]


def days_of(files):
    """Distinct trading days in the file names."""
    return {m.group(1) for m in (DATE_RE.search(os.path.basename(f)) for f in files) if m}


def n_days_of(files):
    """Number of distinct trading days in the file names."""
    return len(days_of(files))


def sibling(path, old, new):
    """The same path with old replaced by new in the file name only."""
    return os.path.join(os.path.dirname(path), os.path.basename(path).replace(old, new))


def read_csv(f):
    """Headerless numeric CSV as a float array (raises ValueError on a corrupt cell)."""
    with open(f) as fh:
        return np.array([r for r in csv.reader(fh) if r], dtype=float)


def signs_and_mid(msg_file, ob_file):
    """Trade signs and mid-price at each trade for one rollout, or None if unusable.

    Trades are execution rows (event type 4) with positive size. The direction column
    is the side of the resting order, so the trade sign is -direction. One aggressive
    order that fills several resting orders produces consecutive same-timestamp,
    same-sign rows; each such burst counts as one trade (the literature convention for
    the sign series) and its mid is taken at the last fill of the burst.
    """
    mm = read_csv(msg_file)
    ob = read_csv(ob_file)
    if MAX_MSGS:
        mm, ob = mm[:MAX_MSGS], ob[:MAX_MSGS]
    if mm.ndim != 2 or ob.ndim != 2 or mm.shape[1] < 6 or ob.shape[1] < 4:
        return None
    n = min(len(mm), len(ob))
    mm, ob = mm[:n], ob[:n]
    ask, bid = ob[:, 0], ob[:, 2]
    mid = (ask + bid) / 2.0
    mid[(ask >= SENT) | (bid >= SENT) | (ask <= 0) | (bid <= 0)] = np.nan
    ex = np.flatnonzero((mm[:, 1] == 4) & (mm[:, 3] > 0))
    if len(ex) < MIN_EXEC:
        return None
    eps = -np.sign(mm[ex, 5])
    ts = mm[ex, 0]
    keep = np.ones(len(ex), bool)
    keep[1:] = ~((np.diff(ts) == 0) & (eps[1:] == eps[:-1]))
    # keep[i] marks the first row of a burst; shift it to mark the last row instead
    last = np.ones(len(ex), bool)
    last[:-1] = keep[1:]
    return eps[last], mid[ex][last]


def accum(files_msg):
    """Per-rollout list of (eps, mid_at_trades); corrupt or unusable rollouts are skipped."""
    runs = []
    for mf in files_msg:
        obf = sibling(mf, 'message', 'orderbook')
        if not os.path.exists(obf):
            continue
        try:
            r = signs_and_mid(mf, obf)
        except ValueError as e:      # a corrupt cell must not abort the whole source
            print(f'  SKIP corrupt rollout {os.path.basename(mf)}: {e}', flush=True)
            continue
        if r is not None:
            runs.append(r)
    return runs


def acf_gamma(runs):
    """Pooled sign autocorrelation C(l), l = 1..LMAX, and its power-law exponent gamma."""
    num = np.zeros(LMAX); den = 0.0
    for eps, _ in runs:
        e = eps - eps.mean()
        v = float(np.dot(e, e))
        den += v
        for l in range(1, min(LMAX, len(e) - 1) + 1):
            num[l - 1] += float(np.dot(e[:-l], e[l:]))
    C = num / max(den, 1e-12)          # pooled ACF: |C| <= 1 by Cauchy-Schwarz
    # Rescaling to C(1) = 1 makes shapes comparable, but only while C(1) is clearly above
    # noise: a near-zero C(1) inflates every other lag. A rescaled curve that leaves
    # [-1, 1] is not an autocorrelation, so the ACF is reported as not identified (NaN).
    if C[0] > 0:
        Cn = C / C[0]
        C = Cn if np.nanmax(np.abs(Cn)) <= 1.0 + 1e-9 else np.full_like(C, np.nan)
    ls = np.arange(1, LMAX + 1)
    m = (ls >= FIT_LO) & (ls <= FIT_HI) & (C > 0)
    gamma = np.nan
    if m.sum() > 10:
        gamma = -float(np.polyfit(np.log(ls[m]), np.log(C[m]), 1)[0])
    return ls, C, gamma


def var_lags():
    """Log-spaced integer lag grid (trades) of the variance curve V(l)."""
    return np.unique(np.round(np.logspace(0, np.log10(LMAX), 24)).astype(int))


def var_H(runs):
    """Mean squared mid displacement V(l) over l trades, and H = slope/2 on 2 <= l <= 100."""
    ls = var_lags()
    s = np.zeros(len(ls)); c = np.zeros(len(ls))
    for _, mid in runs:
        ok = np.isfinite(mid)
        m = mid[ok]
        for i, l in enumerate(ls):
            if len(m) > l + 10:
                d = m[l:] - m[:-l]
                s[i] += float(np.nansum(d * d)); c[i] += len(d)
    V = np.where(c > 0, s / np.maximum(c, 1), np.nan)
    m = np.isfinite(V) & (V > 0) & (ls >= 2) & (ls <= 100)
    H = np.nan
    if m.sum() > 6:
        H = float(np.polyfit(np.log(ls[m]), np.log(V[m]), 1)[0]) / 2.0
    return ls, V, H


def resp(runs):
    """Response R(l) = E[eps dm] and drift-corrected Rc(l), l = 1..200, in ticks.

    A source with a free drift and a sign imbalance picks up a term in R that grows
    roughly linearly in l and is pure drift; Rc = E[eps dm] - E[eps] E[dm] removes it.
    """
    ls = np.arange(1, 201)
    s = np.zeros(len(ls)); c = np.zeros(len(ls))
    sm = np.zeros(len(ls)); se = np.zeros(len(ls))
    for eps, mid in runs:
        for i, l in enumerate(ls):
            if len(mid) > l + 5:
                dm = mid[l:] - mid[:-l]; e = eps[:-l]
                d = dm * e
                ok = np.isfinite(d)
                s[i] += float(np.sum(d[ok])); c[i] += int(ok.sum())
                sm[i] += float(np.sum(dm[ok])); se[i] += float(np.sum(e[ok]))
    cc = np.maximum(c, 1)
    R = np.where(c > 0, s / cc / TICK, np.nan)
    Rc = np.where(c > 0, (s / cc - (se / cc) * (sm / cc)) / TICK, np.nan)
    return ls, R, Rc


def bootstrap_ci(runs):
    """95% percentile intervals of gamma_flow and H, resampling rollouts with replacement."""
    bs_rng = np.random.default_rng(BOOT_SEED)
    gs, hs = [], []
    for _ in range(N_BOOT):
        idx = bs_rng.integers(0, len(runs), len(runs))
        sub = [runs[i] for i in idx]
        gs.append(acf_gamma(sub)[2]); hs.append(var_H(sub)[2])
    g_lo, g_hi = np.nanpercentile(gs, [2.5, 97.5])
    h_lo, h_hi = np.nanpercentile(hs, [2.5, 97.5])
    return g_lo, g_hi, h_lo, h_hi


def verdict_of(H):
    """Diffusivity label of a Hurst exponent."""
    if not np.isfinite(H):
        return 'not identified'
    return ('balanced' if abs(H - 0.5) < H_BALANCE_TOL else
            'SUPERDIFFUSIVE (persistence uncompensated)' if H > H_SUPERDIFFUSIVE else
            'subdiffusive')


def message_files(controls, sub, stock, label):
    """Sorted generated-message CSVs of one arm (buy-side folder)."""
    return sorted(glob.glob(os.path.join(
        controls, sub, f'{stock}-{label}-beta', SIDE, '**', 'data_gen',
        '*message*gen*.csv'), recursive=True))


def build_sources(args, models):
    """Ordered {name: selected message files}; the reference arm comes first as 'Real'."""
    pick = stratify if args.stratify_days else (lambda f, n: f[:n])
    sources = {REAL: pick(message_files(args.controls, args.ref_sub, args.stock, args.ref),
                          args.n_files)}
    for m in models:
        sources[m] = pick(message_files(args.controls, args.noins_sub, args.stock, m),
                          args.n_files)
    return sources


def measure(name, files):
    """All statistics of one source, or None when it has no files or no usable rollout."""
    if not files:
        print(f'{name}: no files', flush=True)
        return None
    runs = accum(files)
    if not runs:
        print(f'{name}: no usable runs', flush=True)
        return None
    n_tr = np.array([len(e) for e, _ in runs])
    buy_share = float(np.mean(np.concatenate([e for e, _ in runs]) > 0))
    pairs100 = int(np.sum(np.maximum(n_tr - 100, 0)))
    print(f'{name}: runs={len(runs)} trades/run median={int(np.median(n_tr))} '
          f'pairs@l=100={pairs100:,} buy_share={buy_share:.3f}', flush=True)
    ls, C, gamma = acf_gamma(runs)
    lv, V, H = var_H(runs)
    lr, R, Rc = resp(runs)
    g_lo, g_hi, h_lo, h_hi = bootstrap_ci(runs)
    print(f'{name}: gamma_flow={gamma:.2f} H_price={H:.2f} runs={len(runs)}', flush=True)
    return dict(name=name, n_runs=len(runs), n_days=n_days_of(files),
                trades_med=int(np.median(n_tr)), pairs100=pairs100, buy_share=buy_share,
                ls=ls, C=C, gamma=gamma, lv=lv, V=V, H=H, lr=lr, R=R, Rc=Rc,
                g_lo=g_lo, g_hi=g_hi, h_lo=h_lo, h_hi=h_hi)


def cache_entries(r):
    """npz entries of one source. Sources are written in table order; the scorecard
    lists its rows in the order of first appearance in the cache."""
    name = r['name']
    return {f'{name}_C': r['C'], f'{name}_V': r['V'], f'{name}_R': r['R'],
            f'{name}_Rc': r['Rc'], f'{name}_gamma': r['gamma'], f'{name}_H': r['H'],
            f'{name}_H_ci': np.array([r['h_lo'], r['h_hi']]),
            f'{name}_gamma_ci': np.array([r['g_lo'], r['g_hi']])}


def write_stats(path, results):
    """Markdown table: gamma_flow, H, their bootstrap CIs, trading days and rollouts."""
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write('| model | $\\gamma_{flow}$ | $H$ | 95% CI on $H$ | 95% CI on $\\gamma$ '
                 '| days | runs |\n|---|---|---|---|---|---|---|\n')
        for r in results:
            fh.write(f'| {r["name"]} | {r["gamma"]:.2f} | {r["H"]:.2f} | '
                     f'[{r["h_lo"]:.2f}, {r["h_hi"]:.2f}] | '
                     f'[{r["g_lo"]:.2f}, {r["g_hi"]:.2f}] | {r["n_days"]} | {r["n_runs"]} |\n')


def verdict_lines(results):
    """Markdown table with sample sizes, gamma_flow, H and the diffusivity verdict."""
    lines = ['| source | runs | trades/run | pairs@100 | buy_share | gamma_flow | H_price | verdict |',
             '|---|---|---|---|---|---|---|---|']
    for r in results:
        lines.append(
            f'| {r["name"]} | {r["n_runs"]} | {r["trades_med"]} | {r["pairs100"]:,} | '
            f'{r["buy_share"]:.3f} | {r["gamma"]:.2f} [{r["g_lo"]:.2f},{r["g_hi"]:.2f}] | '
            f'{r["H"]:.2f} [{r["h_lo"]:.2f},{r["h_hi"]:.2f}] | {verdict_of(r["H"])} |')
    return lines


def plot(results, stock, png):
    """Diagnostic figure: C(l), V(l) and R(l) per source. Returns the path or None."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib not available: figure skipped', flush=True)
        return None
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
    for i, r in enumerate(results):
        c = REF_COLOR if r['name'] == REAL else PALETTE[i % len(PALETTE)]
        ok = r['C'] > 0
        axes[0].loglog(r['ls'][ok], r['C'][ok], color=c, lw=1.5, label=r['name'])
        axes[1].loglog(r['lv'], r['V'], color=c, lw=1.5)
        axes[2].plot(r['lr'], r['R'], color=c, lw=1.5)
    axes[0].set_title('trade-sign autocorrelation $C(\\ell)$', fontsize=10)
    axes[1].set_title('mid displacement variance vs lag\n(slope $=2H$; diffusive: $H = 0.5$)',
                      fontsize=10)
    axes[2].set_title('response $R(\\ell) = E[\\varepsilon_t\\,(m_{t+\\ell}-m_t)]$, ticks',
                      fontsize=10)
    for ax in axes:
        ax.set_xlabel(r'lag $\ell$ (trades)')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=min(7, max(4, (len(l) + 1) // 2)),
               frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, -0.03))
    axes[2].axhline(0, color='#cccccc', lw=0.8)
    fig.suptitle(f'{stock}: order-flow persistence vs price diffusivity', y=1.03)
    fig.tight_layout()
    fig.savefig(png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return png


def parse_args(argv):
    ap = argparse.ArgumentParser(
        prog='python -m lob_impact.flow_balance',
        description='Trade-sign ACF / gamma_flow, Hurst H and trade response R, Rc '
                    'of free-running rollouts against the reference arm.')
    ap.add_argument('--stock', required=True, help='ticker used in the folder and file names')
    ap.add_argument('--models', required=True,
                    help='comma-separated model labels (folders <STOCK>-<MODEL>-beta)')
    ap.add_argument('--ref', default='Historic',
                    help="label of the reference arm (real messages replayed through the same "
                         "pipeline); stored under the key 'Real'")
    ap.add_argument('--controls', default=LOB_IMPACT_CONTROLS,
                    help='root of the free-generation rollouts (default: LOB_IMPACT_CONTROLS)')
    ap.add_argument('--noins_sub', default='noins',
                    help='subdirectory of --controls holding the model rollouts')
    ap.add_argument('--ref_sub', default='noins',
                    help='subdirectory of --controls holding the reference rollouts')
    ap.add_argument('--n_files', type=int, default=64, help='rollouts per source')
    ap.add_argument('--max_msgs', type=int, default=None,
                    help='truncate each rollout to its first N messages')
    ap.add_argument('--tag', default='', help='output filename suffix, e.g. _w500')
    ap.add_argument('--stratify_days', dest='stratify_days', action='store_true', default=True,
                    help='sample --n_files evenly across trading days (default; see stratify)')
    ap.add_argument('--no_stratify_days', dest='stratify_days', action='store_false',
                    help='take the first --n_files paths in sorted order instead')
    ap.add_argument('--outdir', '--out_dir', dest='outdir', default=None,
                    help='output directory (default: LOB_IMPACT_RESULTS/flow_balance)')
    ap.add_argument('--no_plot', action='store_true', help='skip the diagnostic figure')
    return ap.parse_args(argv)


def main(argv=None):
    global MAX_MSGS
    args = parse_args(argv)
    MAX_MSGS = args.max_msgs
    outdir = args.outdir or os.path.join(LOB_IMPACT_RESULTS, 'flow_balance')
    models = [x for x in args.models.split(',') if x]

    results = []
    for name, files in build_sources(args, models).items():
        r = measure(name, files)
        if r is not None:
            results.append(r)
    if not results:
        raise SystemExit('no usable rollouts for %s under %s: check --stock, --models, --ref '
                         'and the layout (python -m pydoc lob_impact.flow_balance)'
                         % (args.stock, args.controls))

    os.makedirs(outdir, exist_ok=True)
    stem = os.path.join(outdir, f'flow_balance_{args.stock}{args.tag}')
    cache = {}
    for r in results:
        cache.update(cache_entries(r))
    np.savez_compressed(stem + '.npz', **cache)
    tbl = stem + '_stats.md'
    write_stats(tbl, results)
    print('STATS ->', tbl)
    real = next((r for r in results if r['name'].lower().startswith('real')), None)
    if real:
        print(f'  (reference stream: gamma={real["gamma"]:.2f}, H={real["H"]:.2f})')
    lines = verdict_lines(results)
    with open(stem + '.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines), flush=True)
    png = None if args.no_plot else plot(results, args.stock, stem + '.png')
    print(f'FLOW_BALANCE_DONE -> {png or stem + ".npz"}', flush=True)


if __name__ == '__main__':
    main()
