"""Daily volatility and daily volume from a per-day statistics CSV.

The build-up fit normalises impact by a daily volatility sigma and the metaorder
volume by the daily traded volume V. Both come from a daily statistics CSV that
you provide (building it from raw market data is not part of this package).

Input: daily statistics CSV, comma-separated, with a header row and one row per
ticker and trading day. Extra columns are ignored.

    ticker          symbol, matched exactly against --stock
    day             trading day as YYYY-MM-DD; must be the same string as the date
                    in the rollout file names (see lob_impact.points)
    open_price      first execution price of the day           (optional column)
    highest_price   highest execution price of the day         (required)
    lowest_price    lowest execution price of the day          (required)
    close_price     last execution price of the day            (optional column)
    execution_sum   total executed volume of the day, in the same unit as the
                    child-order sizes in the rollouts (shares)  (required)

Reference construction, from the day's historical LOBSTER message file: take the
execution messages (event_type 4) with a positive price; open and close are the
first and last of their prices, high and low the maximum and minimum, and
execution_sum the sum of their sizes. Execution prices rather than quotes keep
far-away resting orders from setting the range. Prices can be in any unit (for
example LOBSTER integer prices, dollars x 10000): every estimator works on
log-price differences, so the unit cancels.

Row handling:
  * a row is skipped when it is shorter than the header, when highest_price,
    lowest_price or execution_sum is missing or cannot be parsed as a number,
    when highest_price or lowest_price is <= 0, or when highest_price <
    lowest_price;
  * if the open_price or close_price COLUMN is absent, the estimators that need
    it are NaN; a cell that is present but empty or non-numeric skips the row;
  * a non-positive open or close keeps the row, with the open/close-based
    estimators NaN for that day;
  * rows are sorted by the day string per ticker; keep one row per (ticker, day)
    (with duplicates the last row wins for the per-day values, and every row
    enters the per-ticker estimators).

Estimators (h, l, o, c are natural-log prices of the day; negative variance
terms are clipped at zero):

    parkinson        sqrt((h - l)^2 / (4 ln 2))                       per day
    garman_klass     sqrt(0.5 (h - l)^2 - (2 ln 2 - 1) (c - o)^2)     per day
    rogers_satchell  sqrt((h - c)(h - o) + (l - c)(l - o))            per day
    close_to_close   std (ddof=1) of successive differences of c      per ticker
    yang_zhang       sqrt(var_overnight + k var_open_to_close
                          + (1 - k) mean(rogers_satchell^2)),
                     k = 0.34 / (1.34 + (n + 1) / (n - 1))            per ticker
    range_raw        h - l, the plain daily log range (not in METHODS) per day

Variances use ddof=1. In yang_zhang, overnight returns are o_t - c_{t-1} between
consecutive rows, open-to-close returns are c_t - o_t, and n is the smaller of
their finite counts. The per-ticker estimators use every row of that ticker in
the file as one window (close_to_close needs at least 3 finite closes,
yang_zhang at least 4 days with finite open and close) and are broadcast to all
of its days: a constant rescaling of y per ticker, which leaves a log-log slope
unchanged.

Output: daily_sigmas() returns {(ticker, day): {method: sigma, ..., 'V':
execution_sum}}; sigma is a daily log-return scale (dimensionless), V is in the
unit of execution_sum. The command line prints the table for one ticker.

Usage:
    python -m lob_impact.vol --daily daily_stats.csv --stock XYZ
"""
import argparse
import csv
from collections import defaultdict

import numpy as np

METHODS = ['parkinson', 'garman_klass', 'rogers_satchell', 'close_to_close', 'yang_zhang']
_LN2 = np.log(2.0)


def _rows(daily_csv):
    """Parse the daily CSV into {ticker: [row dict, ...]} sorted by day."""
    by_ticker = defaultdict(list)
    with open(daily_csv) as fh:
        for r in csv.DictReader(fh):
            try:
                O = float(r.get('open_price', 0)); H = float(r['highest_price'])
                L = float(r['lowest_price']); C = float(r.get('close_price', 0))
                V = float(r['execution_sum'])
            except (KeyError, ValueError, TypeError):   # missing, unparseable or short row
                continue
            if min(O, H, L, C) <= 0 or H < L:
                # open/close may be missing: the row is still usable if high/low are valid
                if H <= 0 or L <= 0 or H < L:
                    continue
            by_ticker[r['ticker']].append(dict(day=r['day'], O=O, H=H, L=L, C=C, V=V))
    for t in by_ticker:
        by_ticker[t].sort(key=lambda x: x['day'])
    return by_ticker


def daily_sigmas(daily_csv):
    """(ticker, day) -> dict(method -> sigma, 'range_raw' -> h - l, 'V' -> execution_sum)."""
    by_ticker = _rows(daily_csv)
    out = {}
    for ticker, days in by_ticker.items():
        O = np.array([d['O'] for d in days]); H = np.array([d['H'] for d in days])
        L = np.array([d['L'] for d in days]); C = np.array([d['C'] for d in days])
        o, h, l, c = (np.log(np.where(x > 0, x, np.nan)) for x in (O, H, L, C))

        park = np.sqrt(np.clip((h - l) ** 2 / (4 * _LN2), 0, None))
        gk = np.sqrt(np.clip(0.5 * (h - l) ** 2 - (2 * _LN2 - 1) * (c - o) ** 2, 0, None))
        rs = np.sqrt(np.clip((h - c) * (h - o) + (l - c) * (l - o), 0, None))

        # close-to-close: std of successive log-close changes, one value per ticker
        if np.isfinite(c).sum() >= 3:
            dc = np.diff(c[np.isfinite(c)])
            c2c_val = float(np.std(dc, ddof=1)) if dc.size > 1 else np.nan
        else:
            c2c_val = np.nan

        # Yang-Zhang over the whole window: overnight variance + k * open-to-close
        # variance + (1 - k) * mean Rogers-Satchell variance
        yz_val = np.nan
        fin = np.isfinite(o) & np.isfinite(c)
        if fin.sum() >= 4:
            oc_overnight = o[1:] - c[:-1]          # ln(O_t / C_{t-1})
            co_open = c - o                         # ln(C_t / O_t)
            on = oc_overnight[np.isfinite(oc_overnight)]
            op = co_open[np.isfinite(co_open)]
            rs2 = rs[np.isfinite(rs)] ** 2
            if on.size > 1 and op.size > 1 and rs2.size > 0:
                n = min(on.size, op.size)
                sig_o2 = np.var(on, ddof=1)
                sig_c2 = np.var(op, ddof=1)
                sig_rs2 = np.mean(rs2)
                k = 0.34 / (1.34 + (n + 1) / (n - 1))
                yz_val = float(np.sqrt(max(sig_o2 + k * sig_c2 + (1 - k) * sig_rs2, 0)))

        for i, d in enumerate(days):
            out[(ticker, d['day'])] = dict(
                range_raw=float(h[i] - l[i]) if np.isfinite(h[i] - l[i]) else np.nan,
                parkinson=float(park[i]) if np.isfinite(park[i]) else np.nan,
                garman_klass=float(gk[i]) if np.isfinite(gk[i]) else np.nan,
                rogers_satchell=float(rs[i]) if np.isfinite(rs[i]) else np.nan,
                close_to_close=c2c_val,
                yang_zhang=yz_val,
                V=d['V'],
            )
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog='python -m lob_impact.vol',
        description='Print the daily sigma estimators and daily volume of one ticker.',
        epilog='See the module docstring (python -c "import lob_impact.vol; help(lob_impact.vol)") '
               'for the CSV format.')
    ap.add_argument('--daily', required=True, help='daily statistics CSV (format: see module docstring)')
    ap.add_argument('--stock', required=True, help='ticker, matched exactly against the ticker column')
    args = ap.parse_args(argv)

    sig = daily_sigmas(args.daily)
    days = sorted(k for k in sig if k[0] == args.stock)
    if not days:
        raise SystemExit('%s: no usable rows in %s' % (args.stock, args.daily))
    print('%-10s ' % 'day' + ' '.join('%15s' % m for m in METHODS) + ' %14s' % 'V')
    for key in days:
        s = sig[key]
        print('%-10s ' % key[1] + ' '.join('%15.6f' % s[m] for m in METHODS) + ' %14.0f' % s['V'])
    print('%d days' % len(days))


if __name__ == '__main__':
    main()
