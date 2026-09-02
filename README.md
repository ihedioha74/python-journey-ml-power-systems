# python-journey-ml-power-systems

Machine Learning module (Phase 4.5) of the *Python for Power Engineers* program — split
out from [python-journey](https://github.com/ihedioha74/python-journey) into its own
repository once the module started.

20 sessions (~4 weeks), classical/interpretable ML first (regression, trees, boosting)
before anything neural, built entirely on `load_data_2025.csv` — a synthetic year of
15-minute feeder load data with designed-in daily/weekly/seasonal structure and injected
data-quality problems (missing values, sentinel fault codes, case-inconsistent labels).

## House rules

- Classical, interpretable ML first — deep learning earns its place in Week 4, compared
  honestly against boosted trees.
- Baseline before sophistication — every model has to beat "predict the mean."
- Never test on data trained on.
- Metrics are physical quantities: MW, not abstract scores.

## Data

`load_data_2025.csv` is not committed (see `.gitignore`) — it's fully reproducible via:

```bash
python generate_load_data.py
```

(fixed seed, so this always produces the identical file used across every session).

## Sessions

| # | Topic | Script |
|---|-------|--------|
| ML-1 | What ML actually is — pure-pandas baseline | `ml01_baseline.py` |
| ML-2 | Train/test discipline & metrics — chronological split, held-out baseline (27.18 MW MAE) | `ml02_train_test.py` |

Session PDFs (concept writeup + full committed script as an appendix) are produced
alongside this repo as the module progresses.
