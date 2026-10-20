"""Mid-price impact trajectory of an injected metaorder, per model (build-up and relaxation).

Every rollout is measured against the mid just before its first child order,

    I_t = sign * (mid_t - mid_ref) / mid_ref * 1e4      [bps]; buy sign = +1, sell sign = -1,

so buy and sell rollouts pool into one sample. Two clocks are kept and must not be mixed:

message clock  I_t at every message row, pooled across rollouts row by row with a robust
               central estimator (trimmed mean by default). Rollouts are padded with NaN to
               the longest one; rows reached by too few rollouts are blanked.
k-clock        I_t sampled at executed-volume boundaries: the rows of the child orders
               k = 1..n_ins and, for relaxation shapes, the ends of n_cool cooling windows
               after the last child (k = 0 is 0 by construction). Every rollout gives the
               same n_ins + n_cool + 1 values, so a plain mean and its standard error are
               taken per k. A rollout whose |I| exceeds COLLAPSE_BPS anywhere on the k-clock
               is dropped first: a displacement that large is a broken book, not a price move.

Subcommands
-----------
build    read the rollouts; write the npz cache and two diagnostic plots.
summary  read a cache; print per model the largest |mean| on the k-clock and the standard
         error at that same child, whether it clears DETECTION_SE standard errors, the
         horizon (last child with a finite mean) and end_over_execution (mean at the last
         finite k divided by the mean at k = n_ins, i.e. when execution ended).

Inputs (build)
--------------
    <grid>/<STOCK>-<MODEL>-<shape>/{buy,sell}/**/data_gen/*orderbook*gen*.csv
    <grid>/<STOCK>-<MODEL>-<shape>/{buy,sell}/**/aggressive_indices_<YYYY-MM-DD>.csv

* --grid defaults to LOB_IMPACT_GRID. `**` is any depth, typically one exp_* directory per
  generation run.
* orderbook CSV: headerless LOBSTER orderbook, one row per message of the rollout,
  column 0 = best ask price, column 2 = best bid price (integer price units). A row whose
  ask or bid is >= 2147483647 or <= 0 has no valid mid. The trading day must appear in the
  file name as _YYYY-MM-DD_, e.g. XYZ_2024-01-02_orderbook_real_id_7_gen_id_0.csv.
* aggressive_indices_<day>.csv: one integer per line, the 0-based row indices of the
  injected child orders in that day's rollouts (one schedule per day; if several run
  directories hold a file for the same day, the last in sorted path order is used).
  Rollouts of a day without this file are skipped, as are rollouts that end before
  their first child, whose first child sits at row 0, or whose mid just before it is
  invalid.
* shape 'beta' is the build-up protocol (100 children, no cooling); any other shape name
  (e.g. 'relaxation') is 10 children followed by 100 cooling windows. The cooling window
  length is the smallest gap between consecutive child indices of the rollout.
* The reference arm is, by convention, the model `Historic`: the real message stream
  replayed through the same pipeline. It is read like any other model.
* Optional reference curves (--daily, --per_day_params, --Y, all three together): the
  square-root law I(k) = Y * sigma_d * sqrt(k * child_d / V_d) * 1e4 bps averaged over days,
  with sigma_d and V_d from lob_impact.vol.daily_sigmas(--daily) (daily statistics CSV, format
  in lob_impact.vol) and the child size child_d and messages between children mb_d from the
  per-day parameter CSV (header row; columns day, child, mb; other columns ignored; if a
  day appears on several rows the last one is used). For relaxation shapes also the
  transient-impact (propagator) curve anchored at the square-root peak.

Outputs (build)
---------------
<stem>.npz, default LOB_IMPACT_RESULTS/mid_impact/mid_trajectory_<STOCK>_<shape>.npz, per model M:
    M_x       int      message-row index, 0 .. longest rollout - 1
    M_mid     float    message-clock central trajectory, bps (NaN on thinly covered rows)
    M_band    float    its standard error, bps (--estimator median: 1.57 * IQR / sqrt(n))
    M_cnt     int      number of rollouts reaching each row
    M_k_mean  float    k-clock mean, bps, length n_ins + n_cool + 1
    M_k_se    float    k-clock standard error, bps
  with --dump_samples also, one row per rollout kept by the collapse filter
    M_K       float32  per-rollout k-clock matrix
    M_K_side  int      +1 buy / -1 sell
    M_K_day   str      trading day
  with the reference curves also
    sqrt_x, sqrt_y  square-root law on the message clock: row k * mean(mb_d), bps
    prop_x, prop_y  propagator curve on the message clock, bps (relaxation shapes only)
<stem>.png, <stem>_k.png: message-clock and k-clock plots (skipped with --no_plot).

Usage
-----
    python -m lob_impact.trajectory build --stock XYZ --models Historic,MyModel --shape beta
    python -m lob_impact.trajectory build --stock XYZ --models Historic,MyModel --shape relaxation
    python -m lob_impact.trajectory summary --cache results/mid_impact/mid_trajectory_XYZ_beta.npz
    (summary infers n_ins from the cache: 100 for the build-up shape, 10 for relaxation)
"""
import argparse
import csv
import glob
import os
import re

import numpy as np

from lob_impact.paths import LOB_IMPACT_GRID, LOB_IMPACT_RESULTS

SENTINEL = 2147483647        # price the simulator writes for an empty book side
DATE_RE = re.compile(r'_(\d{4}-\d{2}-\d{2})_')
COLLAPSE_BPS = 2000.0        # |I| beyond this on the k-clock is a broken book, not a move
MIN_ROLLOUTS = 30            # message-clock rows need at least this many rollouts
DETECTION_SE = 3             # detection gate: |peak| must reach this many standard errors
BOOK_BAD_MAX = 0.5           # share of rows without a valid mid above which the book collapsed
GEOMETRY = {'beta': (100, 0)}        # shape -> (children, cooling windows)
DEFAULT_GEOMETRY = (10, 100)         # every other shape: short execution, long relaxation


def k_clock_geometry(shape):
    """(n_ins, n_cool) of the k-clock for a protocol shape name."""
    return GEOMETRY.get(shape, DEFAULT_GEOMETRY)


def _read(path):
    with open(path) as fh:
        return np.array([r for r in csv.reader(fh) if r], dtype=float)


def _aggr_by_day(side_dir):
    """day -> child-order row indices, from every aggressive_indices_<day>.csv below side_dir."""
    out = {}
    for f in sorted(glob.glob(os.path.join(side_dir, '**', 'aggressive_indices_*.csv'),
                              recursive=True)):
        d = os.path.basename(f)[len('aggressive_indices_'):-len('.csv')]
        if re.fullmatch(r'\d{4}-\d{2}-\d{2}', d):
            out[d] = np.loadtxt(f, dtype=int, ndmin=1)
    return out


def collect_traj(side_dir, sign):
    """Signed relative mid trajectories (bps) of every rollout of one side.

    Returns ([(I, child_rows, gap, day), ...], child rows of the first usable rollout).
    gap = smallest spacing between consecutive children (0 for a single child)."""
    obs = sorted(glob.glob(os.path.join(side_dir, '**', 'data_gen', '*orderbook*gen*.csv'),
                           recursive=True))
    aggr_by_day = _aggr_by_day(side_dir)
    trajs, any_aggr = [], None
    for ob in obs:
        m = DATE_RE.search(os.path.basename(ob))
        if not m:
            continue
        aggr = aggr_by_day.get(m.group(1))
        if aggr is None or len(aggr) < 1 or aggr[0] < 1:
            continue
        a = _read(ob)
        if a.ndim != 2 or a.shape[1] < 4:
            continue
        ask_p, bid_p = a[:, 0].copy(), a[:, 2].copy()
        bad = (ask_p >= SENTINEL) | (bid_p >= SENTINEL) | (ask_p <= 0) | (bid_p <= 0)
        mid = (ask_p + bid_p) / 2.0
        mid[bad] = np.nan
        if aggr[0] > len(mid):                       # the rollout ends before its first child
            continue
        ref = mid[aggr[0] - 1]                       # mid just before the first child
        if not np.isfinite(ref) or ref <= 0:
            continue
        mb = int(np.diff(aggr).min()) if len(aggr) > 1 else 0
        trajs.append((sign * (mid - ref) / ref * 1e4, aggr, mb, m.group(1)))
        if any_aggr is None:
            any_aggr = aggr
    return trajs, any_aggr


def at_boundaries(I, aggr, mb, n_ins, n_cool):
    """Sample one trajectory on the k-clock: child rows k = 1..n_ins, then the ends of
    n_cool cooling windows spaced mb rows after the last child. Returns n_ins + n_cool + 1
    values; index 0 is 0, boundaries past the end of the rollout are NaN."""
    nb = n_ins + n_cool
    vals = np.full(nb + 1, np.nan)
    vals[0] = 0.0
    if len(aggr) < 1:
        return vals
    idx = list(aggr[:n_ins])
    last = aggr[min(n_ins, len(aggr)) - 1]
    if n_cool and mb > 0:
        idx += [int(last + j * mb) for j in range(1, n_cool + 1)]
    for j, ix in enumerate(idx, start=1):
        if ix >= len(I) and mb > 0 and ix - (len(I) - 1) <= mb:
            ix = len(I) - 1          # the final window boundary lands on the last row
        if ix < len(I):
            vals[j] = I[ix]
    return vals


def propagator_curve(peak, T, beta, Lmax):
    """Reference transient-impact shape for a constant-rate metaorder of duration T under a
    power-law kernel G(l) ~ l^(-beta): I(t) ~ t^(1-beta) - max(t-T, 0)^(1-beta), scaled so
    that I(T) = peak. Returns (x, y) on 1200 points from 1 to Lmax."""
    g = 1.0 - beta
    x = np.linspace(1.0, float(Lmax), 1200)
    raw = x**g - np.maximum(x - T, 0.0)**g
    y = peak * raw / (T**g)
    return x, y


def sqrt_law_curve(daily, per_day_csv, stock, n_ins, n_total_blocks, Lmax, Y, method):
    """Square-root-law reference (delta = 1/2) with per-day child size, spacing and volume:
    child k = 1..n_ins lands near message row k * mb_d, so x = k * mean_d(mb_d) and
    y(k) = mean_d[Y * sigma_d * sqrt(k * child_d / V_d)] * 1e4 bps. For relaxation shapes
    (n_total_blocks > n_ins) the curve is held flat at its last value out to Lmax: the law
    gives the peak and says nothing about relaxation. Returns (x, y) or (None, None)."""
    from lob_impact.vol import daily_sigmas
    sig = daily_sigmas(daily)
    with open(per_day_csv) as fh:
        pd_rows = {r['day']: r for r in csv.DictReader(fh)}
    days = []
    for day, r in pd_rows.items():
        s = sig.get((stock, day), {})
        sd, V = s.get(method, np.nan), s.get('V', np.nan)
        try:
            child, mb = float(r['child']), float(r['mb'])
        except (KeyError, ValueError):
            continue
        if np.isfinite(sd) and np.isfinite(V) and V > 0 and child > 0 and mb > 0:
            days.append((sd, V, child, mb))
    if not days or n_ins < 1:
        return None, None
    ks = np.arange(1, n_ins + 1)
    per_day_I = np.array([Y * sd * np.sqrt(ks * child / V) for (sd, V, child, mb) in days])
    y = np.nanmean(per_day_I, axis=0) * 1e4
    mb_mean = float(np.mean([mb for (_, _, _, mb) in days]))
    x = ks * mb_mean
    if n_total_blocks > n_ins and Lmax:
        x = np.append(x, float(Lmax))
        y = np.append(y, y[-1])
    return x, y


def reference_curves(args, ins_steps, Lmax, n_ins_k, n_cool_k):
    """Optional square-root-law and propagator references. Returns (cache entries, info)
    or None when no day has complete inputs."""
    n_ins = len(ins_steps) if ins_steps is not None else 0
    xs, ys = sqrt_law_curve(args.daily, args.per_day_params, args.stock, n_ins,
                            n_ins_k + n_cool_k, Lmax, args.Y, args.sigma_method)
    if xs is None:
        print('square-root law: no day with complete inputs, reference skipped')
        return None
    entries = {'sqrt_x': xs, 'sqrt_y': ys}
    info = dict(n_ins=n_ins, xs=xs, ys=ys, Y=args.Y, beta_prop=args.beta_prop)
    keep = xs <= Lmax if Lmax else np.ones_like(xs, bool)
    print('square-root law: end=%.3f bps (per-day child, sigma=%s, Y=%g)'
          % (ys[keep][-1], args.sigma_method, args.Y))
    if args.shape != 'beta' and n_ins >= 1:
        peak = float(ys[n_ins - 1])
        T = float(xs[n_ins - 1])
        px, py = propagator_curve(peak, T, args.beta_prop, Lmax)
        entries['prop_x'] = px
        entries['prop_y'] = py
        info.update(px=px, py=py, peak=peak)
        print('propagator: peak=%.3f at row %.0f, end=%.3f bps (beta=%g)'
              % (peak, T, py[-1], args.beta_prop))
    return entries, info


def _central(M, cnt, estimator, trim):
    """Per-row central value and its standard error on the message clock."""
    if estimator == 'tmean' and trim > 0:
        # trimmed mean: drop `trim` from each tail per row, average the rest
        lo = np.nanpercentile(M, trim * 100, axis=0)
        hi = np.nanpercentile(M, (1 - trim) * 100, axis=0)
        Mk = np.where((M >= lo) & (M <= hi), M, np.nan)
        cntk = np.sum(np.isfinite(Mk), axis=0)
        mid = np.nanmean(Mk, axis=0)
        band = np.nanstd(Mk, axis=0) / np.sqrt(np.maximum(1, cntk))
    elif estimator == 'median':
        # robust, but staircases on tick-discrete prices; band = box-plot notch half-width
        q1, mid, q3 = (np.nanpercentile(M, q, axis=0) for q in (25, 50, 75))
        band = 1.57 * (q3 - q1) / np.sqrt(np.maximum(1, cnt))
    else:
        mid = np.nanmean(M, axis=0)
        band = np.nanstd(M, axis=0) / np.sqrt(np.maximum(1, cnt))
    return mid, band


def model_curves(grid, stock, model, shape, n_ins_k, n_cool_k, estimator='tmean',
                 trim=0.10, min_frac=0.5, smooth=0, dump_samples=False):
    """Both clocks for one model. Returns (cache entries, plot info), or None if no data."""
    exp = '%s-%s-%s' % (stock, model, shape)
    tb, ab = collect_traj(os.path.join(grid, exp, 'buy'), +1)
    ts, asl = collect_traj(os.path.join(grid, exp, 'sell'), -1)
    items = tb + ts
    if not items:
        print('%s: no data' % exp)
        return None
    nb = n_ins_k + n_cool_k
    trajs = [x[0] for x in items]
    K = np.vstack([at_boundaries(I, aggr, mb, n_ins_k, n_cool_k)
                   for I, aggr, mb, _day in items])
    smp_side = np.array([+1] * len(tb) + [-1] * len(ts))
    smp_day = np.array([x[3] for x in items])

    # message clock: pad to the longest rollout, central value per row
    Lm = max(len(t) for t in trajs)
    M = np.full((len(trajs), Lm), np.nan)
    for i, t in enumerate(trajs):
        M[i, :len(t)] = t
    cnt = np.sum(np.isfinite(M), axis=0)
    mid, band = _central(M, cnt, estimator, trim)
    keep = cnt >= max(min_frac * len(trajs), MIN_ROLLOUTS)   # same coverage rule for every model
    Lkeep = int(np.max(np.where(keep)[0])) + 1 if keep.any() else Lm
    mid[~keep] = np.nan
    band[~keep] = np.nan
    if smooth > 1:                                           # optional cosmetic smoothing
        k = np.ones(smooth) / smooth
        valid = np.isfinite(mid)
        mid[valid] = np.convolve(mid[valid], k, mode='same')
    x = np.arange(Lm)
    end = mid[keep][-1] if keep.any() else np.nan
    entries = {model + '_x': x, model + '_mid': mid, model + '_band': band, model + '_cnt': cnt}
    ends = np.array([t[np.isfinite(t)][-1] if np.isfinite(t).any() else np.nan for t in trajs])
    p = np.nanpercentile(ends, [1, 50, 99])
    print('%s: n=%d, rows kept=%d, end=%.2f bps | per-rollout end p1/p50/p99 = %.1f/%.1f/%.1f'
          % (exp, len(trajs), Lkeep, end, p[0], p[1], p[2]))

    # k-clock: drop collapsed books, then plain mean and SE per boundary
    bad = np.nanmax(np.abs(K), axis=1) > COLLAPSE_BPS
    if bad.any():
        print('  [collapse] %s: dropped %d/%d rollouts (|I| > %.0f bps; worst %.0f bps)'
              % (model, int(bad.sum()), len(K), COLLAPSE_BPS, np.nanmax(np.abs(K[bad]))),
              flush=True)
        K = K[~bad]
        smp_side, smp_day = smp_side[~bad], smp_day[~bad]
    kcnt = np.sum(np.isfinite(K), axis=0)
    kmean = np.nanmean(K, axis=0)
    kse = np.nanstd(K, axis=0) / np.sqrt(np.maximum(1, kcnt))
    entries[model + '_k_mean'] = kmean
    entries[model + '_k_se'] = kse
    if dump_samples:
        entries[model + '_K'] = K.astype(np.float32)
        entries[model + '_K_side'] = smp_side
        entries[model + '_K_day'] = smp_day
    print('  k-clock: end=%.2f +/- %.2f bps (k=%d)' % (kmean[-1], kse[-1], nb))
    info = dict(model=model, x=x, mid=mid, band=band, end=end, n=len(trajs), Lkeep=Lkeep,
                ins=ab if ab is not None else asl, kmean=kmean, kse=kse, n_items=len(items))
    return entries, info


def plot_curves(stem, stock, shape, curves, n_ins_k, n_cool_k, ins_steps, Lmax, ref=None):
    """Diagnostic plots: <stem>.png (message clock) and <stem>_k.png (k-clock)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    nb = n_ins_k + n_cool_k
    cmap = plt.get_cmap('tab10')
    fig, ax = plt.subplots(figsize=(11, 5.8))
    figk, axk = plt.subplots(figsize=(11, 5.8))
    for i, c in enumerate(curves):
        col = cmap(i % 10)
        ax.plot(c['x'], c['mid'], '-', color=col, lw=1.6,
                label='%s (end: %.1f bps, n=%d)' % (c['model'], c['end'], c['n']))
        ax.fill_between(c['x'], c['mid'] - 1.96 * c['band'], c['mid'] + 1.96 * c['band'],
                        color=col, alpha=0.10)
        ks = np.arange(nb + 1)
        axk.plot(ks, c['kmean'], '-', color=col, lw=1.7,
                 label='%s (end: %.1f bps, n=%d)' % (c['model'], c['kmean'][-1], c['n_items']))
        axk.fill_between(ks, c['kmean'] - 1.96 * c['kse'], c['kmean'] + 1.96 * c['kse'],
                         color=col, alpha=0.10)
    if ins_steps is not None and Lmax:
        for s in ins_steps[ins_steps < Lmax]:
            ax.axvline(s, color='k', lw=0.3, alpha=0.12)
    if ref is not None:
        _plot_reference(ax, axk, ref, shape, n_ins_k, nb, Lmax)
    if Lmax:
        ax.set_xlim(0, Lmax)
    ax.axhline(0, color='k', lw=0.6)
    ax.set_xlabel('message row (generation time)')
    ax.set_ylabel('mean signed mid-price change (bps), buy + (-1) * sell')
    ax.set_title('%s - mid-price trajectory by model (thin ticks: child orders)' % stock)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(stem + '.png', dpi=150)
    plt.close(fig)

    if shape != 'beta':
        axk.axvline(n_ins_k, color='k', lw=0.8, ls=':', alpha=0.6)
    axk.axhline(0, color='k', lw=0.6)
    axk.set_xlim(0, nb)
    axk.set_xlabel('k: children executed, then cooling windows' if shape != 'beta'
                   else 'k: children executed')
    axk.set_ylabel('mean signed mid-price change (bps), buy + (-1) * sell')
    axk.set_title('%s - mid-price impact on the executed-volume clock%s'
                  % (stock, ' (dotted: execution end)' if shape != 'beta' else ''))
    axk.legend(loc='upper left', fontsize=9)
    axk.grid(True, alpha=0.3)
    figk.tight_layout()
    figk.savefig(stem + '_k.png', dpi=150)
    plt.close(figk)
    print('saved -> %s.png, %s_k.png' % (stem, stem))


def _plot_reference(ax, axk, ref, shape, n_ins_k, nb, Lmax):
    """Draw the square-root-law (dashed) and propagator (dotted) references on both clocks."""
    xs, ys = ref['xs'], ref['ys']
    keep = xs <= Lmax if Lmax else np.ones_like(xs, bool)
    lbl = 'square-root law' + (' (peak held)' if shape != 'beta' else ' (Y=%g)' % ref['Y'])
    ax.plot(xs[keep], ys[keep], '--', color='k', lw=2.0,
            label='%s, end %.2f bps' % (lbl, ys[keep][-1]))
    xk = np.arange(1, n_ins_k + 1, dtype=float)
    yk = ys[:n_ins_k].copy()
    if shape != 'beta':
        xk = np.append(xk, float(nb))
        yk = np.append(yk, yk[-1])
    axk.plot(xk, yk, '--', color='k', lw=2.0, label='%s, end %.2f bps' % (lbl, yk[-1]))
    if 'px' in ref:
        ax.plot(ref['px'], ref['py'], ':', color='tab:purple', lw=2.2,
                label='propagator beta=%g (peak %.2f -> %.2f bps)'
                      % (ref['beta_prop'], ref['peak'], ref['py'][-1]))
        pxk, pyk = propagator_curve(float(yk[n_ins_k - 1]), float(n_ins_k),
                                    ref['beta_prop'], nb)
        axk.plot(pxk, pyk, ':', color='tab:purple', lw=2.2,
                 label='propagator beta=%g (peak %.2f -> %.2f bps)'
                       % (ref['beta_prop'], yk[n_ins_k - 1], pyk[-1]))


def _models(text):
    return [m for m in (text or '').split(',') if m]


def build(args):
    models = _models(args.models)
    n_ins_k, n_cool_k = k_clock_geometry(args.shape)
    cache, curves = {}, []
    ins_steps, Lmax = None, 0
    for model in models:
        res = model_curves(args.grid, args.stock, model, args.shape, n_ins_k, n_cool_k,
                           estimator=args.estimator, trim=args.trim, min_frac=args.min_frac,
                           smooth=args.smooth, dump_samples=args.dump_samples)
        if res is None:
            continue
        entries, info = res
        cache.update(entries)
        if ins_steps is None:
            ins_steps = info['ins']
        Lmax = max(Lmax, info['Lkeep'])
        curves.append(info)
    if not curves:
        raise SystemExit('no usable rollouts for %s, shape %s, under %s: check --stock, '
                         '--models, --shape and the layout (python -m pydoc lob_impact.trajectory)'
                         % (args.stock, args.shape, args.grid))
    ref = None
    if args.daily and args.per_day_params:
        res = reference_curves(args, ins_steps, Lmax, n_ins_k, n_cool_k)
        if res is not None:
            cache.update(res[0])
            ref = res[1]
    stem = (os.path.splitext(args.out)[0] if args.out else
            os.path.join(LOB_IMPACT_RESULTS, 'mid_impact',
                         'mid_trajectory_%s_%s' % (args.stock, args.shape)))
    os.makedirs(os.path.dirname(os.path.abspath(stem)), exist_ok=True)
    np.savez(stem + '.npz', **cache)
    print('saved -> %s.npz' % stem)
    if not args.no_plot and curves:
        try:
            plot_curves(stem, args.stock, args.shape, curves, n_ins_k, n_cool_k, ins_steps, Lmax,
                        ref)
        except Exception as exc:          # plots are diagnostics; the cache is already written
            print('plot skipped: %s: %s' % (type(exc).__name__, exc))


# --------------------------------------------------------------------------------------
# Reading a cache back: detection gate, horizon, relaxation ratio, book health
# --------------------------------------------------------------------------------------

def cache_arr(cache, key):
    """cache[key] as float array, or None. cache: np.load result or dict."""
    keys = cache.files if hasattr(cache, 'files') else cache
    if cache is None or key not in keys:
        return None
    return np.asarray(cache[key], dtype=float)


def peak_and_se(cache, model):
    """Largest |k-clock mean| (signed value) and the standard error at that same child.
    Both come from the per-child curve (<model>_k_mean / <model>_k_se), never from the
    message-clock arrays."""
    km, kse = cache_arr(cache, model + '_k_mean'), cache_arr(cache, model + '_k_se')
    if km is None or kse is None or km.size == 0:
        return None, None
    ok = np.isfinite(km) & np.isfinite(kse)
    if not ok.any():
        return None, None
    i = int(np.nanargmax(np.where(ok, np.abs(km), np.nan)))
    return float(km[i]), float(kse[i])


def is_detected(peak, se):
    """Detection gate: |peak| >= DETECTION_SE * se (a zero or missing se does not fail it).
    None when there is no peak."""
    if peak is None:
        return None
    return not (se not in (None, 0) and abs(peak) < DETECTION_SE * se)


def horizon(cache, model):
    """Last k-clock index with a finite mean: the horizon the model was measured over."""
    km = cache_arr(cache, model + '_k_mean')
    if km is None:
        return None
    f = np.where(np.isfinite(km))[0]
    return int(f[-1]) if f.size else None


def end_over_execution(k_mean, n_ins):
    """k_mean[last finite index] / k_mean[n_ins]: the displacement at the end of the window
    relative to the displacement when execution ended. On the k-clock index 0 is the origin
    and index n_ins the last child, followed by the cooling windows. None if undefined."""
    km = np.asarray(k_mean, dtype=float)
    f = np.where(np.isfinite(km))[0]
    if not f.size or n_ins < 0 or n_ins >= km.size:
        return None
    return float(km[f[-1]] / km[n_ins])


def book_health(grid, stock, model, shape='beta', n_files=6):
    """Share of orderbook rows without a valid mid (empty side or sentinel price), averaged
    over the first n_files rollout files in path order (buy side first; first 40000 rows of
    each). A share above BOOK_BAD_MAX means the book collapsed and cannot carry impact."""
    pat = os.path.join(grid, '%s-%s-%s' % (stock, model, shape), '*', '**',
                       'data_gen', '*orderbook*gen*.csv')
    fs = sorted(glob.glob(pat, recursive=True))[:n_files]
    if not fs:
        return None
    shares = []
    for f in fs:
        try:
            a = np.genfromtxt(f, delimiter=',', max_rows=40000)
        except Exception as exc:          # unreadable file: skip it, as the gate always has
            print('book_health: skipped %s (%s: %s)' % (f, type(exc).__name__, exc))
            continue
        if a.ndim != 2 or a.shape[1] < 4:
            continue
        ask, bid = a[:, 0], a[:, 2]
        bad = (ask >= SENTINEL) | (bid >= SENTINEL) | (ask <= 0) | (bid <= 0)
        shares.append(float(bad.mean()))
    return float(np.mean(shares)) if shares else None


def _fmt(v):
    if v is None:
        return '-'
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return '%.6g' % v


def infer_n_ins(cache, models):
    """Children of the protocol shape whose k-clock length matches the cache, or None."""
    by_length = {}
    for shape in ('beta', 'relaxation'):
        n_ins, n_cool = k_clock_geometry(shape)
        by_length[n_ins + n_cool + 1] = n_ins
    for m in models:
        km = cache_arr(cache, m + '_k_mean')
        if km is not None:
            return by_length.get(km.size)
    return None


def summary(args):
    cache = np.load(args.cache, allow_pickle=False)
    models = _models(args.models) or sorted(k[:-len('_k_mean')] for k in cache.files
                                            if k.endswith('_k_mean'))
    inferred = infer_n_ins(cache, models)
    n_ins = args.n_ins if args.n_ins is not None else inferred
    if n_ins is None:
        raise SystemExit('cannot infer the number of children from %s; pass --n_ins' % args.cache)
    if inferred is not None and n_ins != inferred:
        raise SystemExit('--n_ins %d does not match the cache, whose k-clock length implies %d '
                         'children' % (n_ins, inferred))
    cols = ['model', 'peak_bps', 'se_at_peak', 'detected', 'horizon', 'end_over_execution']
    if args.book_health:
        cols += ['book_bad_share', 'book_collapse']
    rows = []
    for m in models:
        peak, se = peak_and_se(cache, m)
        km = cache_arr(cache, m + '_k_mean')
        row = [m, peak, se, is_detected(peak, se), horizon(cache, m),
               end_over_execution(km, n_ins) if km is not None else None]
        if args.book_health:
            share = book_health(args.grid, args.stock, m, args.shape)
            row += [share, None if share is None else share > BOOK_BAD_MAX]
        rows.append([_fmt(v) if i else v for i, v in enumerate(row)])
    widths = [max(len(c), *(len(r[i]) for r in rows)) if rows else len(c)
              for i, c in enumerate(cols)]
    print('cache: %s   n_ins=%d   gate: |peak| >= %d SE' % (args.cache, n_ins, DETECTION_SE))
    print('  '.join(c.ljust(w) for c, w in zip(cols, widths)))
    for r in rows:
        print('  '.join(v.ljust(w) for v, w in zip(r, widths)))


def main(argv=None):
    ap = argparse.ArgumentParser(prog='python -m lob_impact.trajectory',
                                 description=__doc__.split('\n')[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)

    b = sub.add_parser('build', help='read rollouts, write the trajectory cache')
    b.add_argument('--stock', required=True,
                   help='ticker as it appears in the grid directory names')
    b.add_argument('--models', required=True,
                   help='comma-separated model names, e.g. Historic,MyModel')
    b.add_argument('--shape', default='beta',
                   help="protocol suffix of the grid directories: 'beta' (100 children) or a "
                        "relaxation shape such as 'relaxation' (10 children + 100 cooling "
                        "windows)")
    b.add_argument('--grid', default=LOB_IMPACT_GRID, help='root of the metaorder rollouts')
    b.add_argument('--estimator', default='tmean', choices=['tmean', 'median', 'mean'],
                   help='message-clock central estimator')
    b.add_argument('--trim', type=float, default=0.10,
                   help='trimmed mean: fraction dropped from EACH tail per row')
    b.add_argument('--min_frac', type=float, default=0.5,
                   help='keep message rows reached by >= this fraction of rollouts '
                        '(and by at least %d)' % MIN_ROLLOUTS)
    b.add_argument('--smooth', type=int, default=0,
                   help='rolling-mean window (rows) on the message-clock line; 0 = off')
    b.add_argument('--dump_samples', action='store_true',
                   help='also store the per-rollout k-clock matrix <model>_K (+ _K_side, _K_day)')
    b.add_argument('--out', default=None,
                   help='output stem (extension ignored); default '
                        'LOB_IMPACT_RESULTS/mid_impact/mid_trajectory_<STOCK>_<shape>')
    b.add_argument('--no_plot', action='store_true', help='write the cache only')
    b.add_argument('--daily', default=None,
                   help='daily statistics CSV (format: lob_impact.vol); with --per_day_params and --Y '
                        'adds the square-root-law reference')
    b.add_argument('--per_day_params', default=None,
                   help='per-day protocol CSV with columns day, child, mb')
    b.add_argument('--Y', type=float, default=None,
                   help='square-root-law prefactor, I = Y * sigma * sqrt(Q/V), in the units of '
                        '--sigma_method')
    b.add_argument('--sigma_method', default='parkinson',
                   choices=['parkinson', 'garman_klass', 'rogers_satchell', 'close_to_close',
                            'yang_zhang', 'range_raw'],
                   help='daily sigma from lob_impact.vol for the square-root-law reference')
    b.add_argument('--beta_prop', type=float, default=0.5,
                   help='propagator kernel exponent of the relaxation reference')

    s = sub.add_parser('summary', help='print gate numbers from a trajectory cache')
    s.add_argument('--cache', required=True, help='npz written by `build`')
    s.add_argument('--n_ins', type=int, default=None,
                   help='children in the protocol (100 for beta, 10 for relaxation shapes); '
                        'default: inferred from the cache')
    s.add_argument('--models', default=None, help='comma-separated; default: all in the cache')
    s.add_argument('--book_health', action='store_true',
                   help='also read raw orderbooks (needs --stock, --shape, --grid)')
    s.add_argument('--stock', default=None, help='ticker, for --book_health')
    s.add_argument('--shape', default='beta', help='protocol shape, for --book_health')
    s.add_argument('--grid', default=LOB_IMPACT_GRID,
                   help='root of the metaorder rollouts, for --book_health (default: LOB_IMPACT_GRID)')

    args = ap.parse_args(argv)
    if args.cmd == 'build':
        if bool(args.daily) != bool(args.per_day_params):
            ap.error('--daily and --per_day_params go together')
        if args.daily and args.Y is None:
            ap.error('the square-root-law reference needs --Y')
        if args.Y is not None and not args.daily:
            ap.error('--Y is used only by the square-root-law reference: add --daily and '
                     '--per_day_params')
        build(args)
    else:
        if args.book_health and not args.stock:
            ap.error('--book_health needs --stock')
        summary(args)


if __name__ == '__main__':
    main()
