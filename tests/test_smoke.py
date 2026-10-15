"""End to end on the synthetic data set: every tool runs and writes finite numbers.

The synthetic flow has a square-root impact path built in, so the build-up exponent must
come out near 1/2, and the toy model relaxes while the reference arm does not. The data
live under a directory whose name contains 'message' and 'orderbook', which the tools must
not confuse with the file names they pair.
"""
import json
import os

import numpy as np
import pytest

from lobi import (decay_shape, delta, drift, flow_balance, master_curve, points, scorecard,
                  synthetic, trajectory, triangle)

STOCK = synthetic.STOCK
MODELS = 'Historic,ToyModel'


@pytest.fixture(scope='session')
def data(tmp_path_factory):
    root = tmp_path_factory.mktemp('message_and_orderbook_data')
    roots = synthetic.generate(str(root))
    roots['out'] = str(root / 'results')
    return roots


def _out(data, *parts):
    return os.path.join(data['out'], *parts)


def test_phase1_flow_balance_and_scorecard(data, capsys):
    outdir = _out(data, 'flow_balance')
    flow_balance.main(['--stock', STOCK, '--models', 'ToyModel', '--controls', data['controls'],
                       '--outdir', outdir, '--no_plot'])
    z = np.load(os.path.join(outdir, f'flow_balance_{STOCK}.npz'))
    for name in ('Real', 'ToyModel'):
        assert np.isfinite(z[f'{name}_gamma']) and np.isfinite(z[f'{name}_H'])
        assert np.all(np.isfinite(z[f'{name}_H_ci']))
    scorecard.main(['--cache', os.path.join(outdir, f'flow_balance_{STOCK}.npz')])
    printed = capsys.readouterr().out
    assert 'ToyModel' in printed and 'distance to "Real"' in printed


def test_phase1_drift(data):
    drift.main(['--stock', STOCK, '--models', 'ToyModel', '--controls', data['controls'],
                '--outdir', data['out']])
    table = open(_out(data, f'drift_{STOCK}.md')).read()
    assert '| Historic |' in table and '| ToyModel |' in table and 'n/a' not in table
    # the days column counts distinct trading days, not days per side folder
    assert all(row.rstrip(' |').endswith(str(len(synthetic.DAYS)))
               for row in table.splitlines() if row.startswith('| Historic'))


@pytest.mark.parametrize('shape, n_ins', [('beta', 100), ('relaxation', 10)])
def test_trajectory_and_master_curve(data, shape, n_ins, capsys):
    stem = _out(data, 'mid_impact', f'mid_trajectory_{STOCK}_{shape}')
    trajectory.main(['build', '--stock', STOCK, '--models', MODELS, '--shape', shape,
                     '--grid', data['grid'], '--out', stem, '--no_plot'])
    cache = np.load(stem + '.npz')
    for m in MODELS.split(','):
        peak, se = trajectory.peak_and_se(cache, m)
        assert peak > 0 and trajectory.is_detected(peak, se)
        assert trajectory.horizon(cache, m) == len(cache[m + '_k_mean']) - 1
    assert trajectory.infer_n_ins(cache, MODELS.split(',')) == n_ins
    trajectory.main(['summary', '--cache', stem + '.npz'])
    assert 'n_ins=%d' % n_ins in capsys.readouterr().out

    mc = _out(data, 'master_curve', f'master_curve_{STOCK}_{shape}')
    master_curve.main(['--stock', STOCK, '--models', MODELS, '--shape', shape,
                       '--grid', data['grid'], '--out', mc, '--no_plot'])
    z = np.load(mc + '.npz')
    assert bool(z['ToyModel_sig']) and z['ToyModel_master'][n_ins] == pytest.approx(1.0)
    if shape == 'relaxation':
        ep = trajectory.end_over_execution(cache['ToyModel_k_mean'], n_ins)
        assert 0 < ep < 0.8                 # the toy model relaxes
        ref = trajectory.end_over_execution(cache['Historic_k_mean'], n_ins)
        assert ref > 0.8                    # the reference arm does not


def test_decay_shape(data):
    mc = _out(data, 'master_curve', f'master_curve_{STOCK}_relaxation')
    if not os.path.exists(mc + '.npz'):
        master_curve.main(['--stock', STOCK, '--models', MODELS, '--shape', 'relaxation',
                           '--grid', data['grid'], '--out', mc, '--no_plot'])
    decay_shape.main(['--stock', STOCK, '--master_curve', mc + '.npz',
                      '--out_dir', _out(data, 'master_curve')])
    z = np.load(_out(data, 'master_curve', f'decay_shape_{STOCK}.npz'))
    assert str(z['ToyModel_verdict']) in ('down', 'exp', 'pl')
    assert z['ToyModel_drop'] > z['Historic_drop']


def test_triangle(data):
    out = _out(data, 'triangle')
    triangle.main(['build', '--stock', STOCK, '--models', 'ToyModel', '--grid', data['grid'],
                   '--controls', data['controls'], '--out_dir', out, '--no_plot'])
    numbers = json.load(open(os.path.join(out, 'ToyModel', 'numbers.json')))
    assert set(triangle.ARMS) <= set(numbers)
    ratio = triangle.vis_blind(numbers)
    assert ratio is not None and ratio > 1     # the visible arm moves the price more


def test_points_and_delta(data):
    cache = _out(data, f'points_{STOCK}.npz')
    points.main(['--stock', STOCK, '--models', MODELS, '--grid', data['grid'], '--out', cache,
                 '--daily', data['daily']])
    delta.main(['--stock', STOCK, '--models', MODELS, '--points', cache,
                '--daily', data['daily'], '--out_dir', data['out']])
    res = json.load(open(_out(data, f'delta_{STOCK}.json')))
    for m in MODELS.split(','):
        e = res[f'{STOCK}:{m}']
        assert 0.35 < e['delta'] < 0.65         # a square root was built in
        assert e['lo'] <= e['hi'] and e['n_day'] == len(synthetic.DAYS)


def test_unknown_stock_is_an_error(data, tmp_path):
    with pytest.raises(SystemExit):
        flow_balance.main(['--stock', 'NOPE', '--models', 'ToyModel',
                           '--controls', data['controls'], '--outdir', str(tmp_path), '--no_plot'])
    with pytest.raises(SystemExit):
        trajectory.main(['build', '--stock', 'NOPE', '--models', 'ToyModel',
                         '--grid', data['grid'], '--out', str(tmp_path / 't'), '--no_plot'])
    with pytest.raises(SystemExit):
        points.main(['--stock', 'NOPE', '--models', 'ToyModel', '--grid', data['grid'],
                     '--out', str(tmp_path / 'p.npz')])


def test_rollout_shorter_than_its_first_child_is_skipped(tmp_path):
    exp = tmp_path / 'buy' / 'exp_1'
    (exp / 'data_gen').mkdir(parents=True)
    np.savetxt(exp / 'aggressive_indices_2024-01-02.csv', [50, 61, 72], fmt='%d')
    book = np.tile([200100, 10, 200000, 10], (20, 1))         # 20 rows, first child at 50
    np.savetxt(exp / 'data_gen' / 'XYZ_2024-01-02_orderbook_real_id_0_gen_id_0.csv', book,
               fmt='%d', delimiter=',')
    trajs, _ = trajectory.collect_traj(str(tmp_path / 'buy'), +1)
    assert trajs == []
    assert master_curve.collect_paths(str(tmp_path / 'buy'), +1, 3, 0) == []
