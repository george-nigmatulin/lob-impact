"""Synthetic rollouts in the layout the tools read: try the pipeline without market data.

Writes a small, deterministic data set for one stock (XYZ), three trading days and two
models: `Historic`, standing in for the reference arm (the real message stream replayed
through the same pipeline), and `ToyModel`. Nothing here imitates a real market; the
order flow is a toy built so that every tool has something finite to measure:

  * trade signs come in runs of Pareto-distributed length, mixed with herding and noise,
    so the sign autocorrelation decays slowly; a trade moves the price one tick in its
    direction with a fixed probability;
  * a metaorder adds an impact path amp * sqrt(k) ticks after its k-th child (a square
    root in executed volume), which after the last child relaxes towards a permanent
    fraction with an exponential time constant;
  * the visible arm carries the full impact path, the blind arm a small mechanical one.

Layout written under --out (see README for the contract):

    controls/noins/XYZ-<MODEL>-beta/{buy,sell}/exp_1/data_gen/*.csv       free runs
    controls/invisible/XYZ-ToyModel-beta/{buy,sell}/exp_1/...             blind arm
    grid/XYZ-<MODEL>-<shape>/{buy,sell}/exp_1/data_gen/*.csv              visible arm
    grid/.../exp_1/aggressive_indices_<YYYY-MM-DD>.csv                    child rows
    daily_stats.csv                                                       lobi.vol format

Usage:
    python -m lobi.synthetic --out /tmp/lobi_demo
"""
import argparse
import os

import numpy as np

STOCK = 'XYZ'
DAYS = ('2024-01-02', '2024-01-03', '2024-01-04')
TICK = 100                  # price units per tick (LOBSTER: dollars x 10000)
BID0 = 20_000               # starting best bid, in ticks
CHILD = 100                 # shares per child order
GAP = 11                    # rows from one child to the next (10 messages in between)
FIRST = 10                  # row of the first child; the row before it is the reference
ROLLOUTS = 5                # per day and side
N_FREE = 2000               # rows of a free rollout
SHAPES = {'beta': 100, 'relaxation': 10}    # shape -> children
N_COOL = 100                # cooling windows after the last child (relaxation shape)

FLOW = {                    # free-run order flow per model
    'Historic': dict(herd=0.30, noise=0.40, alpha=1.6, p_exec=0.30, p_move=0.30),
    'ToyModel': dict(herd=0.40, noise=0.25, alpha=1.4, p_exec=0.30, p_move=0.40),
}
GRID_FLOW = dict(herd=0.20, noise=0.60, alpha=1.6, p_exec=0.30, p_move=0.06)
IMPACT = {                  # (amplitude in ticks, permanent fraction, relaxation rows)
    ('Historic', 'visible'): (2.0, 1.0, 300.0),
    ('ToyModel', 'visible'): (3.0, 0.5, 300.0),
    ('ToyModel', 'blind'): (0.8, 1.0, 300.0),
}
DAILY = dict(open_price=2_000_000, highest_price=2_040_000, lowest_price=1_960_000,
             close_price=2_010_000, execution_sum=5_000_000)


def _trade_signs(rng, n, p):
    """n trade signs: herding, noise, or the current run of Pareto-distributed length."""
    u, coin = rng.random(n), rng.choice((-1, 1), n)
    out = np.empty(n, dtype=int)
    run_sign, run_left, last = int(coin[0]), 0, int(coin[0])
    for i in range(n):
        if u[i] < p['herd']:
            s = last
        elif u[i] < p['herd'] + p['noise']:
            s = int(coin[i])
        else:
            if run_left <= 0:
                run_sign = int(rng.choice((-1, 1)))
                run_left = int(np.ceil(rng.pareto(p['alpha'] - 1.0) + 1.0))
            s, run_left = run_sign, run_left - 1
        out[i], last = s, s
    return out


def _impact_path(n, children, amp, perm, tau):
    """amp * sqrt(children executed so far), relaxing after the last child (ticks)."""
    if not len(children):
        return np.zeros(n)
    t = np.arange(n)
    k = np.searchsorted(children, t, side='right')
    path = amp * np.sqrt(k)
    after = t > children[-1]
    peak = amp * np.sqrt(len(children))
    path[after] = peak * (perm + (1.0 - perm) * np.exp(-(t[after] - children[-1]) / tau))
    return path


def rollout(rng, n, flow, side=+1, children=(), impact=(0.0, 1.0, 1.0)):
    """One rollout: (message rows, orderbook rows) as integer/float arrays."""
    children = np.asarray(children, dtype=int)
    is_child = np.zeros(n, bool)
    is_child[children] = True
    is_exec = (rng.random(n) < flow['p_exec']) & ~is_child
    signs = np.zeros(n, dtype=int)
    signs[is_exec] = _trade_signs(rng, int(is_exec.sum()), flow)
    moves = signs * (rng.random(n) < flow['p_move'])
    bid = BID0 + np.cumsum(moves) + np.rint(side * _impact_path(n, children, *impact))
    bid = bid.astype(np.int64) * TICK
    ask = bid + TICK

    time = 34_200.0 + np.cumsum(rng.exponential(0.05, n) + 1e-6)
    event = np.where(is_exec | is_child, 4, rng.choice((1, 3), n))
    size = np.where(is_child, CHILD, rng.integers(1, 200, n))
    trade_sign = np.where(is_child, side, signs)
    direction = np.where(event == 4, -trade_sign, rng.choice((-1, 1), n))
    price = np.where(direction < 0, ask, bid)
    msg = np.column_stack([time, event, np.arange(1, n + 1), size, price, direction])
    book = np.column_stack([ask, rng.integers(50, 500, n), bid, rng.integers(50, 500, n)])
    return msg, book


def _write(exp_dir, day, rid, msg, book):
    gen = os.path.join(exp_dir, 'data_gen')
    os.makedirs(gen, exist_ok=True)
    stem = os.path.join(gen, f'{STOCK}_{day}_%s_real_id_{rid}_gen_id_{rid}.csv')
    np.savetxt(stem % 'message', msg, fmt=['%.9f', '%d', '%d', '%d', '%d', '%d'], delimiter=',')
    np.savetxt(stem % 'orderbook', book, fmt='%d', delimiter=',')


def _arm(rng, side_dir, flow, side, n, children=(), impact=(0.0, 1.0, 1.0)):
    """ROLLOUTS rollouts per day under side_dir/exp_1, plus the per-day child rows."""
    exp_dir = os.path.join(side_dir, 'exp_1')
    for day in DAYS:
        for r in range(ROLLOUTS):
            msg, book = rollout(rng, n, flow, side, children, impact)
            _write(exp_dir, day, r, msg, book)
        if len(children):
            np.savetxt(os.path.join(exp_dir, f'aggressive_indices_{day}.csv'),
                       np.asarray(children), fmt='%d')


def generate(out, seed=0):
    """Write the whole data set under out; returns the roots it created."""
    rng = np.random.default_rng(seed)
    controls, grid = os.path.join(out, 'controls'), os.path.join(out, 'grid')
    for model, flow in FLOW.items():
        for side, sign in (('buy', +1), ('sell', -1)):
            _arm(rng, os.path.join(controls, 'noins', f'{STOCK}-{model}-beta', side),
                 flow, sign, N_FREE)
    for shape, n_children in SHAPES.items():
        children = FIRST + GAP * np.arange(n_children)
        n = int(children[-1]) + GAP * (N_COOL if shape != 'beta' else 1) + 5
        for (model, arm), impact in IMPACT.items():
            root = grid if arm == 'visible' else os.path.join(controls, 'invisible')
            if arm == 'blind' and shape != 'beta':
                continue                        # the blind arm is read on the build-up shape
            for side, sign in (('buy', +1), ('sell', -1)):
                _arm(rng, os.path.join(root, f'{STOCK}-{model}-{shape}', side),
                     GRID_FLOW, sign, n, children, impact)
    daily = os.path.join(out, 'daily_stats.csv')
    with open(daily, 'w') as fh:
        fh.write('ticker,day,' + ','.join(DAILY) + '\n')
        for day in DAYS:
            fh.write(f'{STOCK},{day},' + ','.join(str(v) for v in DAILY.values()) + '\n')
    return dict(controls=controls, grid=grid, daily=daily)


def main(argv=None):
    ap = argparse.ArgumentParser(prog='python -m lobi.synthetic',
                                 description='Write a small synthetic data set in the input layout.')
    ap.add_argument('--out', required=True, help='directory to create the data set in')
    ap.add_argument('--seed', type=int, default=0, help='random seed (default 0)')
    args = ap.parse_args(argv)
    roots = generate(args.out, args.seed)
    print(f'synthetic data -> {args.out}  (stock {STOCK}, models Historic and ToyModel)')
    print(f'  export LOBI_CONTROLS={roots["controls"]}')
    print(f'  export LOBI_GRID={roots["grid"]}')
    print(f'  daily statistics: {roots["daily"]}')


if __name__ == '__main__':
    main()
