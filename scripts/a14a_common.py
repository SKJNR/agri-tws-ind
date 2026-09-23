"""Shared data loading / field building for agent 14-a (generator forensics).

Cell key = (lat, lon). Calendar index t_abs = year*12 + (month-1).
Train span: 2002-05..2015-08 (138 months present of 160), 15,715 cells.
"""
import numpy as np
import pandas as pd

DATA = "/home/z/my-project/data"
TRAIN = f"{DATA}/Train (1).csv"
TEST = f"{DATA}/Test (2).csv"
COVS = ["SPEI_01_t", "SPEI_03_t", "SPEI_06_t", "SPEI_12_t", "SOIL_MOISTURE_t"]

# calendar month range of train
T0 = 2002 * 12 + (5 - 1)          # 2002-05 => 24016
T1 = 2015 * 12 + (8 - 1)          # 2015-08


def load_train(cols=("time", "lat", "lon", "TWS_t", "target") + tuple(COVS)):
    df = pd.read_csv(TRAIN, usecols=list(cols))
    tm = pd.to_datetime(df["time"])
    df["year"] = tm.dt.year
    df["month"] = tm.dt.month
    df["t_abs"] = df["year"] * 12 + (df["month"] - 1)
    return df


def load_test(cols=("ID", "time", "lat", "lon", "TWS_t", "TWS_t_masked") + tuple(COVS)):
    df = pd.read_csv(TEST, usecols=list(cols))
    tm = pd.to_datetime(df["time"])
    df["year"] = tm.dt.year
    df["month"] = tm.dt.month
    df["t_abs"] = df["year"] * 12 + (df["month"] - 1)
    return df


def build_grid(df):
    """Return (cells: DataFrame lat,lon sorted; cell_idx: Series->int, months: sorted unique t_abs)."""
    cells = df[["lat", "lon"]].drop_duplicates().sort_values(["lat", "lon"]).reset_index(drop=True)
    cells["cid"] = np.arange(len(cells))
    months = np.sort(df["t_abs"].unique())
    return cells, months


def field_matrix(df, value_col, cells, months):
    """(T, C) float64 matrix with NaN for missing (cell, month)."""
    cmap = {(r.lat, r.lon): r.cid for r in cells.itertuples()}
    cidx = pd.Series([cmap[(a, b)] for a, b in zip(df["lat"], df["lon"])])
    midx = pd.Series(np.searchsorted(months, df["t_abs"].values))
    F = np.full((len(months), len(cells)), np.nan)
    F[midx.values, cidx.values] = df[value_col].values
    return F


def month_of_tabs(t_abs):
    return (t_abs % 12) + 1
