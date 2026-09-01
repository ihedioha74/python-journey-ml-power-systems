#!/usr/bin/env python3
"""
Created on Sat Jul 25 20:41:07 2026

@author: emmanuel_uchenna_ihedioha
"""

"""
generate_load_data.py — Generate a realistic year of synthetic feeder load data.

Real load is a SUM OF LAYERS: a base level, a daily cycle, a weekly cycle, a
seasonal cycle, random noise, and some realistic messiness. Building each layer
explicitly means the structure is known — so when analysis recovers it, both
the generator and the analysis are validated against ground truth.

Tuned for a WINTER-PEAKING German grid: demand peaks in January, troughs in July.

Output: load_data_2025.csv  (140,160 rows = 35,040 fifteen-minute intervals x 4 feeders)
Usage:  python generate_load_data.py
"""
import numpy as np
import pandas as pd

SEED = 42  # fixed seed -> reproducible data
YEAR = 2025
BASE = {"A": 60, "B": 75, "C": 55, "D": 70}  # per-feeder floor (MW)


def generate():
    rng = np.random.default_rng(SEED)

    # --- time backbone: every feeder reports at every 15-min timestamp ---
    timestamps = pd.date_range(
        f"{YEAR}-01-01 00:00", f"{YEAR}-12-31 23:45", freq="15min"
    )
    index = pd.MultiIndex.from_product(
        [timestamps, list(BASE)], names=["timestamp", "feeder"]
    )
    df = index.to_frame(index=False)

    # helper time components
    hour = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60
    dayofweek = df["timestamp"].dt.dayofweek  # 0=Mon .. 6=Sun
    dayofyear = df["timestamp"].dt.dayofyear

    # --- the layers ---
    base = df["feeder"].map(BASE)
    daily = 35 * np.cos((hour - 19) / 24 * 2 * np.pi)  # peak 19:00, trough ~07:00
    weekly = np.where(dayofweek >= 5, -12, +5)  # weekends lower
    seasonal = 20 * np.cos((dayofyear - 1) / 365 * 2 * np.pi)  # winter peak (day 1)
    noise = rng.normal(0, 6, len(df))  # +/-6 MW wobble

    df["load_mw"] = (base + daily + weekly + seasonal + noise).round(2)
    df["feeder"] = df["feeder"].astype(str)
    data = df[["timestamp", "load_mw", "feeder"]].copy()

    # --- realistic messiness, so cleaning tools have work to do ---
    n = len(data)
    miss = rng.choice(n, int(0.02 * n), replace=False)  # 2% missing
    data.loc[miss, "load_mw"] = np.nan
    fault = rng.choice(n, 20, replace=False)  # sentinel fault codes
    data.loc[fault, "load_mw"] = 9999
    case = rng.choice(n, int(0.01 * n), replace=False)  # case-inconsistent labels
    data.loc[case, "feeder"] = data.loc[case, "feeder"].str.lower()

    return data


def main():
    data = generate()
    out = "load_data_2025.csv"
    data.to_csv(out, index=False)
    print(f"Saved {out} — {len(data):,} rows")
    print(f"  Missing:   {data['load_mw'].isna().sum():,}")
    print(f"  Faults:    {(data['load_mw'] == 9999).sum()}")
    print(f"  Feeders:   {sorted(data['feeder'].unique())}")


if __name__ == "__main__":
    main()
