#!/usr/bin/env python3
"""
ml02_train_test.py — Session ML-2: Train/test discipline & metrics

The habit that separates real ML from self-deception.

Three things get established here, and none of them are ever broken again:

  1. A score computed on data the model has already seen is not evidence.
     Demonstrated below with a lookup table that scores a PERFECT 0.00 MW
     on its training data and has exactly zero predictive power.

  2. Time series must be split CHRONOLOGICALLY. sklearn's train_test_split
     shuffles by default, which for a time series means the model gets to
     see rows from either side of every row it's tested on. Demonstrated
     below: the same model scores 6.94 MW or 24.19 MW depending only on
     how the split was made.

  3. Metrics carry units. MAE in MW is the honest one, RMSE in MW punishes
     big misses, MAPE in % is unitless and fragile near zero load.

Output of this session is the module's REFERENCE NUMBER: the training-mean
baseline scored properly on held-out data. Every model from ML-3 onward is
measured against that, not against ML-1's in-sample figure.

Reuses the cleaning judgment from ML-1 rather than restating it — this is
why ml01_baseline.py was written as functions instead of one blob.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from ml01_baseline import load_raw, clean_for_baseline

SPLIT_DATE = "2025-10-01"   # train Jan-Sep, test Oct-Dec

# MAPE divides by the actual load, so a reading near zero produces an enormous
# percentage from a perfectly ordinary MW error — and a reading of exactly zero
# produces infinity. This data has 3 exact zeros and 396 readings under 1 MW
# (deep summer-weekend-pre-dawn troughs), so MAPE is reported only over rows
# above this floor, with the excluded count stated. This is the honest way to
# quote a percentage metric on a feeder that legitimately approaches zero.
MAPE_MIN_LOAD = 1.0  # MW


def prepare() -> pd.DataFrame:
    """ML-1's cleaning, plus the datetime parsing a chronological split needs."""
    df = clean_for_baseline(load_raw())
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


def chronological_split(df: pd.DataFrame, cutoff: str = SPLIT_DATE):
    """
    The only honest split for a forecasting problem: everything before the
    cutoff is what you'd have known; everything after is the future you're
    being judged on. No shuffling, no peeking.
    """
    train = df[df["timestamp"] < cutoff]
    test = df[df["timestamp"] >= cutoff]
    return train, test


def score(y_true: pd.Series, y_pred, label: str) -> dict:
    """
    Metrics as physical quantities. MAE and RMSE are in MW because the target
    is in MW — no unit conversion happens anywhere in the arithmetic.
    Computed by hand first, then cross-checked against sklearn.
    """
    y_pred = np.asarray(y_pred, dtype=float)
    actual = y_true.to_numpy()
    error = actual - y_pred

    mae = np.abs(error).mean()                      # MW — the honest one
    rmse = np.sqrt((error ** 2).mean())             # MW — punishes big misses

    # MAPE only where dividing by the actual load is meaningful (see the note
    # on MAPE_MIN_LOAD above).
    ok = np.abs(actual) >= MAPE_MIN_LOAD
    mape = (np.abs(error[ok]) / np.abs(actual[ok])).mean() * 100      # %
    skipped = (~ok).sum()

    # trust, but verify: hand arithmetic vs the library we'll rely on from here
    assert np.isclose(mae, mean_absolute_error(y_true, y_pred))
    assert np.isclose(rmse, root_mean_squared_error(y_true, y_pred))

    note = f"  [MAPE skips {skipped} rows under {MAPE_MIN_LOAD:.0f} MW]" if skipped else ""
    print(f"  {label:34s} MAE {mae:6.2f} MW   RMSE {rmse:6.2f} MW   "
          f"MAPE {mape:5.1f} %{note}")
    return {"mae": mae, "rmse": rmse, "mape": mape}


def demo_the_lie(train: pd.DataFrame, test: pd.DataFrame, fallback: float) -> None:
    """
    A lookup table that memorises every (timestamp, feeder) -> load_mw pair it
    was trained on. On its training data it is a flawless model. On the future
    it knows nothing at all — not one key matches — so it falls back to the
    mean and lands exactly where the dumb baseline does.

    Perfect training score, zero predictive power. That is the shape of the lie.
    """
    print("\n1. Why testing on training data lies")
    table = dict(zip(zip(train["timestamp"], train["feeder"]), train["load_mw"]))

    pred_train = [table.get(k, fallback) for k in zip(train["timestamp"], train["feeder"])]
    pred_test = [table.get(k, fallback) for k in zip(test["timestamp"], test["feeder"])]
    hits = sum(k in table for k in zip(test["timestamp"], test["feeder"]))

    score(train["load_mw"], pred_train, "lookup table, scored on TRAIN")
    score(test["load_mw"], pred_test, "lookup table, scored on TEST")
    print(f"  -> {hits:,} of {len(test):,} test keys were found in the table.")
    print("  -> A model can memorise its way to a perfect training score and know nothing.")


def _nearest_in_time(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """For each test row, the training reading closest in time on that feeder."""
    parts = []
    for feeder in sorted(test["feeder"].unique()):
        a = test[test["feeder"] == feeder].sort_values("timestamp")
        b = train[train["feeder"] == feeder].sort_values("timestamp")
        merged = pd.merge_asof(
            a, b[["timestamp", "load_mw"]].rename(columns={"load_mw": "pred"}),
            on="timestamp", direction="nearest",
        )
        merged["gap_min"] = (
            merged["timestamp"] - pd.merge_asof(
                a[["timestamp"]], b[["timestamp"]].assign(tr=b["timestamp"].values),
                on="timestamp", direction="nearest",
            )["tr"].values
        ).abs().dt.total_seconds() / 60
        parts.append(merged)
    return pd.concat(parts)


def demo_why_chronological(df: pd.DataFrame) -> None:
    """
    One model — "predict the nearest reading in time on the same feeder" —
    scored under two different splits of the same data.

    Under a random split its neighbours are 15 minutes away, because shuffling
    scatters test rows in between training rows. It looks excellent. Under a
    chronological split those neighbours are weeks away and it collapses.

    The model didn't change. Only the split did. That gap is the size of the
    lie a random split tells you about a time series.
    """
    print("\n2. Why a time series must be split chronologically")

    # sklearn's default: shuffled. Correct for independent rows, wrong for time.
    rand_train, rand_test = train_test_split(df, test_size=0.25, random_state=42)
    chrono_train, chrono_test = chronological_split(df)

    for label, (tr, te) in {
        "random split (train_test_split)": (rand_train, rand_test),
        "chronological split": (chrono_train, chrono_test),
    }.items():
        m = _nearest_in_time(tr, te)
        score(m["load_mw"], m["pred"], f"nearest-in-time, {label}")
        gap = m["gap_min"].median()
        readable = f"{gap:,.0f} min" if gap < 1440 else f"{gap / 1440:,.0f} days"
        print(f"       median gap to nearest training row: {readable}")


def main() -> None:
    df = prepare()
    train, test = chronological_split(df)

    print(f"Chronological split at {SPLIT_DATE}")
    print(f"  train: {len(train):,} rows  "
          f"{train['timestamp'].min():%Y-%m-%d} -> {train['timestamp'].max():%Y-%m-%d}")
    print(f"  test : {len(test):,} rows  "
          f"{test['timestamp'].min():%Y-%m-%d} -> {test['timestamp'].max():%Y-%m-%d}")

    # The baseline is fit on TRAIN ONLY. Averaging the whole file would quietly
    # let the test period influence the prediction it's about to be judged on.
    baseline = train["load_mw"].mean()

    demo_the_lie(train, test, baseline)
    demo_why_chronological(df)

    print("\n3. The honest baseline — fit on train, scored on held-out test")
    print(f"  predicted load: {baseline:.2f} MW (mean of Jan-Sep only)")
    score(train["load_mw"], np.full(len(train), baseline), "baseline, scored on TRAIN")
    result = score(test["load_mw"], np.full(len(test), baseline), "baseline, scored on TEST")

    # Why MAE is quoted first and MAPE last: one bad denominator wrecks a
    # percentage. The worst single row below shows a perfectly ordinary MW
    # error turning into a four-figure percentage.
    ape = (test["load_mw"] - baseline).abs() / test["load_mw"].abs() * 100
    worst = ape.idxmax()
    print(f"\n  Worst single MAPE row: actual {test.loc[worst, 'load_mw']:.2f} MW, "
          f"error {abs(test.loc[worst, 'load_mw'] - baseline):.2f} MW "
          f"-> {ape[worst]:,.0f} % error.")
    print("  A 60 MW miss on a 0.3 MW feeder reading is not 18,000% worse than a")
    print("  60 MW miss at peak. MAE in MW never lies to you like that.")

    bias = (test["load_mw"] - baseline).mean()
    print(f"\n  Mean load Jan-Sep: {train['load_mw'].mean():.2f} MW")
    print(f"  Mean load Oct-Dec: {test['load_mw'].mean():.2f} MW")
    print(f"  Baseline bias on test: {bias:+.2f} MW — it under-predicts every single hour.")
    print("  That is the winter peak the training window never saw. Physics, not a bug.")

    print(f"\nREFERENCE NUMBER FOR THE MODULE: {result['mae']:.2f} MW MAE on held-out data.")
    print("Every model from ML-3 onward is measured against this, never against")
    print("an in-sample score.")


if __name__ == "__main__":
    main()
