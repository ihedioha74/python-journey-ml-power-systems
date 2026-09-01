#!/usr/bin/env python3
"""
ml01_baseline.py — Session ML-1: What ML actually is

Vocabulary, defined against our own data — no sklearn yet.

  Supervised learning : learn a mapping from features X to a target y.
                         On this data: X = (hour, day_of_week, month, ...),
                         y = load_mw.
  Regression           : predicting a number (MW).      <- today
  Classification       : predicting a category (fault/no-fault).  <- ML-13
  Features & targets   : plain pandas column selection, nothing new.

Today's actual point: the baseline by hand. Before any model touches this
data, we compute the dumbest possible prediction — the training mean — and
measure exactly how wrong it is, in MW. Every real model built in this
module has to beat this number, or it isn't earning its complexity.

No modeling libraries used here. Just pandas.
"""
import pandas as pd

DATA_PATH = "load_data_2025.csv"


def load_raw(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the CSV exactly as it comes off disk — no cleaning yet."""
    return pd.read_csv(path)


def audit_data_quality(df: pd.DataFrame) -> None:
    """
    Quantify the messiness before deciding what to do about it.
    This dataset (Session 25) has three DIFFERENT kinds of imperfection,
    and they are not the same kind of problem:

      1. Missing values (NaN)        — unknown truth, can't score against it
      2. Sentinel fault codes (9999) — a known "this reading is bad" flag,
                                        not a real MW value. These are the
                                        rows ML-13 will learn to CLASSIFY —
                                        they don't belong in a load number.
      3. Case-inconsistent labels    — 'a' instead of 'A'. Cosmetic. The MW
                                        reading itself is still legitimate.

    Note what's absent from this list: negative load_mw values. Those are
    NOT faults — they fall out naturally from the generator's daily/weekly/
    seasonal layers (a low-base feeder at its weekend, summer, pre-dawn
    trough can legitimately dip below zero in this synthetic signal). We
    keep them. Filtering "surprising" numbers just because they're negative
    would be exactly the kind of physically-blind cleaning this module
    warns against.
    """
    n = len(df)
    n_missing = df["load_mw"].isna().sum()
    n_fault = (df["load_mw"] == 9999).sum()
    n_case = df["feeder"].str.islower().sum()
    n_negative = (df["load_mw"] < 0).sum()

    print("Data quality audit")
    print(f"  total rows            : {n:,}")
    print(f"  missing load_mw (NaN) : {n_missing:,}  -> excluded from baseline")
    print(f"  9999 sentinel faults  : {n_fault:,}  -> excluded from baseline")
    print(f"  case-inconsistent lbl : {n_case:,}  -> label only, MW value kept")
    print(f"  negative load_mw      : {n_negative:,}  -> kept (legitimate trough, not a fault)")
    print()


def clean_for_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineering-judgment filter for THIS baseline only:
      - drop rows with unknown load (NaN) — nothing to learn or score against
      - drop rows flagged with the 9999 sentinel — not a physical MW reading
    Everything else, including negative dips and messy-cased feeder labels,
    stays. This is a documented decision, not a silent one.
    """
    clean = df[df["load_mw"].notna() & (df["load_mw"] != 9999)].copy()
    clean["feeder"] = clean["feeder"].str.upper()
    return clean


def baseline_predict_and_score(df: pd.DataFrame) -> tuple[float, float]:
    """
    The baseline by hand: predict the training mean for every row, then
    measure the average absolute error in MW. No sklearn — just arithmetic.
    """
    y = df["load_mw"]
    baseline_prediction = y.mean()               # the "model": one number
    absolute_errors = (y - baseline_prediction).abs()
    mean_absolute_error_mw = absolute_errors.mean()
    return baseline_prediction, mean_absolute_error_mw


def main() -> None:
    raw = load_raw()
    print(f"Loaded {DATA_PATH}: {raw.shape[0]:,} rows x {raw.shape[1]} cols\n")

    audit_data_quality(raw)

    clean = clean_for_baseline(raw)
    dropped = len(raw) - len(clean)
    print(f"Rows usable for the baseline: {len(clean):,} "
          f"({dropped:,} dropped: NaN + 9999 sentinel)\n")

    baseline_mw, mae_mw = baseline_predict_and_score(clean)

    print("Baseline model: predict the training mean, always")
    print(f"  predicted load        : {baseline_mw:.2f} MW  (constant, for every row)")
    print(f"  mean absolute error    : {mae_mw:.2f} MW")
    print()
    print(f"This is the bar. Any model built in ML-2 onward that can't beat "
          f"{mae_mw:.2f} MW isn't worth shipping.")


if __name__ == "__main__":
    main()
