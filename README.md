# LOB-Impact Benchmark

**Does your generative limit-order-book model reproduce market impact?**

Distributional metrics answer whether generated order flow *looks* like the real stream.
They cannot answer whether the model *reacts* correctly when someone trades against it,
and a model used as a market simulator is used for exactly that. These tools measure the
reaction.

You generate rollouts with your own model, following the protocol below. The tools turn
them into the numbers that show whether the free run keeps the real stream's balance,
whether a metaorder moves the price, whether the model (rather than the book mechanics)
caused the move, how impact grows with executed volume, and whether the price relaxes
once execution stops.

## Install

```bash
git clone https://github.com/george-nigmatulin/lob-impact.git
cd lob-impact
python -m pip install -U pip       # editable installs need pip 21.3 or newer
pip install -e ".[test]"
pytest -q                          # end-to-end smoke test on synthetic data, about ten seconds
```

Python 3.9 or newer; NumPy, pandas, Matplotlib and PyYAML. `python -m lobi` lists the
tools. Each one is a module with its own help, `python -m lobi.<tool> --help`, and its
full documentation (inputs, outputs, every cached key) is in
`python -m pydoc lobi.<tool>`.

## Try it on synthetic data

```bash
python -m lobi.synthetic --out /tmp/lobi_demo
export LOBI_CONTROLS=/tmp/lobi_demo/controls LOBI_GRID=/tmp/lobi_demo/grid \
       LOBI_RESULTS=/tmp/lobi_demo/results
python -m lobi.flow_balance --stock XYZ --models ToyModel
python -m lobi.scorecard --stock XYZ
python -m lobi.trajectory build --stock XYZ --models Historic,ToyModel --shape beta
python -m lobi.points --stock XYZ --models Historic,ToyModel
python -m lobi.delta --stock XYZ --models Historic,ToyModel --daily /tmp/lobi_demo/daily_stats.csv
```

The toy data has a square-root impact path built in, so the build-up exponent comes out
near 0.5. It imitates no real market; it exists so that every tool has something to
measure.

## The protocol

Every rollout starts from a window of real messages that conditions the model. The model
then generates messages, which are replayed into an order-book simulator so that every
message comes with the book state after it.

**Phase 1: free run.** No intervention. Does the generated stream keep the balance of the
real one: long memory in trade signs, a diffusive price, no drift of its own?

**Phase 2: build-up.** A metaorder is injected: child market orders of a fixed size, all
in one direction, spaced by a fixed number of generated messages. The same generation is
run in two arms:

- **visible**: the child orders execute in the simulator, and the model conditions on them;
- **blind**: the child orders execute in the simulator, but the model never sees them.

The blind arm isolates the book mechanics; the difference between the arms is the part of
the response the model produced. Shape `beta` has 100 children and no cooling.

**Phase 3: relaxation.** Shape `relaxation` has 10 children followed by 100 cooling windows
of the same length without children. Does the price come back once execution stops?

Children are market orders (LOBSTER event type 4). Their size, the spacing between them
and the length of the conditioning window are yours to choose; keep them the same for every
model you compare, and for the reference arm. Run buy and sell metaorders (they are pooled
with the sign flipped) on many trading days: the exponent's confidence interval resamples
days.

## Input layout

```
$LOBI_CONTROLS/
    noins/<STOCK>-<MODEL>-beta/{buy,sell}/exp_<run>/          free runs    (Phase 1)
    invisible/<STOCK>-<MODEL>-beta/{buy,sell}/exp_<run>/      blind arm    (Phase 2)
$LOBI_GRID/
    <STOCK>-<MODEL>-beta/{buy,sell}/exp_<run>/                visible arm, build-up
    <STOCK>-<MODEL>-relaxation/{buy,sell}/exp_<run>/          visible arm, relaxation
```

Every `exp_<run>` directory (one or several per side; `<run>` is any name, and a single
run may also put its `data_gen/` straight into the side directory) holds

- `data_gen/<STOCK>_<YYYY-MM-DD>_message_real_id_<r>_gen_id_<g>.csv` and the matching
  `..._orderbook_...` file: one pair per rollout. The date in the file name is the trading
  day.
- in the arms with child orders, `aggressive_indices_<YYYY-MM-DD>.csv`: one integer per
  line, the 0-based row of each child order. All rollouts of a day share this schedule, so
  if several runs hold a file for the same day, the files must be identical.

The reference arm is called `Historic`: the real message stream of the same days, replayed
through the same pipeline (with the child orders inserted, in the metaorder arms). It is
read exactly like a model. A few names are historical and kept for compatibility: `beta`
is the build-up shape (the exponent itself is called delta), `invisible` holds the blind
arm, and `flow_balance` stores the reference arm under the key `Real`.

CSV files are headerless and follow LOBSTER:

- **message**: `time, event_type, order_id, size, price, direction`. Time in seconds after
  midnight; event type 4 is the execution of a visible order; direction is the side of the
  resting order (+1 buy, -1 sell), so a buyer-initiated trade has direction -1.
- **orderbook**: `ask_price_1, ask_size_1, bid_price_1, bid_size_1, ...`; row *i* is the
  book after message *i*. Prices are integers on a 100-unit tick (LOBSTER: dollars times
  10 000, one-cent tick). An empty side is written as 2147483647, or a price of 0 or less.

The build-up exponent also needs **daily statistics of the real stock**: a CSV with the
header `ticker,day,open_price,highest_price,lowest_price,close_price,execution_sum`, one row
per trading day, holding the first, highest, lowest and last execution price and the total
executed volume in shares (format details: `python -c "import lobi.vol; help(lobi.vol)"`).

Point the tools at the data with the environment variables `LOBI_CONTROLS`, `LOBI_GRID`
and `LOBI_RESULTS`, with `config/paths.yaml` (copy `config/paths.example.yaml`), or with
each tool's path flags. `python -m lobi.paths` prints what was resolved.

## The tools

| phase | what it measures | command |
|---|---|---|
| 1 | trade-sign autocorrelation and its exponent, Hurst exponent of the mid, response to a trade | `python -m lobi.flow_balance --stock S --models M` |
| 1 | free drift | `python -m lobi.drift --stock S --models M` |
| 1 | response ratio R(100)/R(1), H on two lag bands, distances to the reference | `python -m lobi.scorecard --stock S` |
| 2, 3 | mean impact per message and per child | `python -m lobi.trajectory build --stock S --models Historic,M --shape beta` (and `--shape relaxation`) |
| 2, 3 | detection against three standard errors, horizon, end over execution | `python -m lobi.trajectory summary --cache <npz>` |
| 2 | visible arm against blind arm | `python -m lobi.triangle build --stock S --models M` |
| 2 | impact against executed volume, one point per child | `python -m lobi.points --stock S --models Historic,M` |
| 2 | build-up exponent with a day-clustered 95% interval | `python -m lobi.delta --stock S --models Historic,M --daily daily.csv` |
| 2, 3 | master curve, impact normalised at the end of execution | `python -m lobi.master_curve --stock S --models Historic,M --shape relaxation` |
| 3 | shape of the relaxation: rise, flat, decline, and its law where identifiable | `python -m lobi.decay_shape --stock S` |

Outputs land under `LOBI_RESULTS`: caches as `.npz`, tables as Markdown. A tool whose own
gate fails reports that the quantity is not identified rather than printing a number: a
response that does not clear three standard errors, a fit without enough positive bins, a
bootstrap whose resamples are all undefined.

## Two things that will bite you

**Sample the days, not the first N files.** Paths sort by date, so the first N files are
the earliest days. For a model with thousands of rollouts that can be a single trading day,
measured against a reference that spans the whole period. `flow_balance` and `drift` spread
their sample over trading days by default, and their tables carry the day and rollout
counts. Check those columns.

**Match the horizon to the reference.** The estimators have fit windows: the sign
autocorrelation is fitted over lags of 2 to 40 trades, the price variance up to lag 100. A
rollout much shorter than about ten times the longest lag starves them. Measure the model
and the reference at the same length (`--max_msgs` truncates), or the difference you see is
the truncation, not the model.

## What is not here

- Training and generation. Bring your rollouts.
- Market data. LOBSTER data ([lobsterdata.com](https://lobsterdata.com)) is licensed
  separately, and nothing here ships it, raw or derived.
- The order-book simulator that turns generated messages into book states, for example
  [JAX-LOB](https://github.com/KangOxford/jax-lob).
- Distributional metrics: see [LOB-Bench](https://github.com/peernagy/lob_bench).

## Licence and citation

Apache License 2.0, see `LICENSE` and `NOTICE`. To cite, see `CITATION.cff`.
