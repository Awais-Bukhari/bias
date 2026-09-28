import pandas as pd
import numpy as np
from src.methods import METHODS
from src.metrics import compute_metrics

def remove_statistical_outliers(df, factor=3.0):
    temp = df.copy()
    temp["residual"] = temp["ERA5_mm"] - temp["OBS_mm"]
    Q1, Q3 = temp["residual"].quantile(0.25), temp["residual"].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - factor * IQR, Q3 + factor * IQR
    clean = temp[(temp["residual"] >= lower) & (temp["residual"] <= upper)].copy()
    return clean.drop(columns=["residual"])

def temporal_cv(pooled, train_start, train_end, methods=METHODS):
    years = range(train_start, train_end + 1)
    by_year = {y: pooled[pooled["DATE"].dt.year == y] for y in years if y in pooled["DATE"].dt.year.unique()}
    rows = []
    for held in by_year.keys():
        tr = pd.concat([by_year[y] for y in by_year if y != held], ignore_index=True)
        te = by_year[held]
        if len(te) < 5 or len(tr) < 5:
            continue
        m = compute_metrics(te["OBS_mm"], te["ERA5_mm"]); m.update({"Method": "Raw_ERA5", "Held": held}); rows.append(m)
        for mn, mf in methods.items():
            try:
                _, c = mf(tr, te)
                m = compute_metrics(te["OBS_mm"], c); m.update({"Method": mn, "Held": held}); rows.append(m)
            except Exception:
                continue
    return pd.DataFrame(rows)

def run_pipeline(pooled_train: pd.DataFrame, train_start: int, train_end: int, outlier_factor: float = 3.0):
    """pooled_train must have columns: DATE (datetime), OBS_mm, ERA5_mm"""
    pooled_train = remove_statistical_outliers(pooled_train, factor=outlier_factor)

    cv_results = temporal_cv(pooled_train, train_start, train_end)
    summary = cv_results.groupby("Method")[["RMSE_mm", "R2", "NSE"]].mean().round(3)

    winner = summary.drop(index="Raw_ERA5").sort_values(["NSE", "R2"], ascending=False).index[0]
    winner_params, corrected = METHODS[winner](pooled_train, pooled_train)

    pooled_train_c = pooled_train.copy()
    pooled_train_c["CORR_mm"] = corrected

    return {
        "pooled_train_clean": pooled_train,
        "cv_summary": summary,
        "winner": winner,
        "winner_params": winner_params,
        "pooled_train_corrected": pooled_train_c,
    }
from src.fetch_era5 import fetch_era5_daily_precip

def build_pooled_dataset(obs_df: pd.DataFrame) -> pd.DataFrame:
    """
    obs_df must have columns: STATION, LAT, LON, DATE, OBS_mm
    Fetches ERA5 for each unique station automatically and merges.
    """
    pieces = []
    stations = obs_df[["STATION", "LAT", "LON"]].drop_duplicates()

    for _, s in stations.iterrows():
        station_obs = obs_df[obs_df["STATION"] == s["STATION"]].copy()
        start = station_obs["DATE"].min().strftime("%Y-%m-%d")
        end = station_obs["DATE"].max().strftime("%Y-%m-%d")

        era5 = fetch_era5_daily_precip(s["LAT"], s["LON"], start, end)
        merged = pd.merge(station_obs, era5, on="DATE", how="inner")
        merged["STATION"] = s["STATION"]
        pieces.append(merged)

    if not pieces:
        return pd.DataFrame(columns=["DATE", "OBS_mm", "ERA5_mm", "STATION"])

    return pd.concat(pieces, ignore_index=True)