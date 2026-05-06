"""
data_loader.py
Loads and preprocesses ACN-Data sessions for the EV charging simulator.
Supports both live API download and local JSON/CSV files.
"""

import json
import math
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from sklearn.ensemble import IsolationForest


# ── Constants ────────────────────────────────────────────────────────────────
STEP_MINUTES   = 5                        # simulation time-step in minutes
MAX_RATE_KW    = 6.6                      # standard Level-2 EVSE max rate (kW)
DEFAULT_DEMAND = 17.5                     # default kWh if not provided
DEFAULT_HOURS  = 4.0                      # default session hours if not provided


# ── Public entry point ───────────────────────────────────────────────────────
def load_sessions(source, site="caltech", start_date=None, end_date=None,
                  remove_outliers=True, verbose=True):
    """
    Load and preprocess ACN-Data sessions.

    Parameters
    ----------
    source : str
        Path to a local JSON file exported from ACN portal,
        OR 'api' to attempt a live download (requires acnportal token).
    site   : str   - 'caltech' or 'jpl'
    start_date, end_date : str 'YYYY-MM-DD' or None
    remove_outliers : bool - apply Isolation Forest

    Returns
    -------
    pd.DataFrame with columns:
        session_id, arrival_step, deadline_step, energy_kwh,
        max_rate_kw, user_id, claimed
    int : total number of simulation steps
    """
    if source == "api":
        raw = _download_from_api(site, start_date, end_date)
    else:
        raw = _load_local(source)

    df = _parse(raw, start_date, end_date)

    if verbose:
        print(f"[data_loader] Raw sessions loaded : {len(df)}")

    df = _filter_invalid(df)

    if verbose:
        print(f"[data_loader] After invalid filter: {len(df)}")

    if remove_outliers:
        df = _remove_outliers(df)
        if verbose:
            print(f"[data_loader] After outlier removal: {len(df)}")

    df = _discretise(df)
    df = _user_history(df)
    df = _cyclic_features(df)

    n_steps = int(df["deadline_step"].max()) + 1

    if verbose:
        print(f"[data_loader] Simulation-ready sessions: {len(df)}")
        print(f"[data_loader] Total time steps: {n_steps}")

    return df.reset_index(drop=True), n_steps


# ── Step 1 : load raw data ───────────────────────────────────────────────────
def _load_local(path):
    """Load sessions from a local JSON or CSV file."""
    if path.endswith(".csv"):
        return pd.read_csv(path)
    with open(path) as f:
        data = json.load(f)
    # ACN API returns {'_items': [...]}
    if isinstance(data, dict) and "_items" in data:
        return pd.json_normalize(data["_items"])
    if isinstance(data, list):
        return pd.json_normalize(data)
    raise ValueError("Unrecognised JSON structure.")


def _download_from_api(site, start_date, end_date):
    """Download sessions from the ACN-Data REST API."""
    try:
        import requests
        token = input("Enter your ACN-Data API token: ").strip()
        url   = f"https://ev.caltech.edu/api/v1/sessions/{site}"
        params = {"pretty": "false"}
        if start_date:
            params["connectionTime[gte]"] = start_date
        if end_date:
            params["connectionTime[lte]"] = end_date
        resp = requests.get(url, params=params,
                            auth=(token, ""), timeout=60)
        resp.raise_for_status()
        return pd.json_normalize(resp.json()["_items"])
    except Exception as e:
        raise RuntimeError(f"API download failed: {e}\n"
                           "Use a local JSON file instead.")


# ── Step 2 : parse timestamps ────────────────────────────────────────────────
def _parse(df, start_date, end_date):
    """Convert timestamp strings to UTC-aware datetimes."""

    def to_utc(col):
        return pd.to_datetime(df[col], utc=True, errors="coerce")

    df["connectionTime"]   = to_utc("connectionTime")
    df["disconnectTime"]   = to_utc("disconnectTime")
    df["doneChargingTime"] = to_utc("doneChargingTime")

    # Optional user-input fields
    if "userInputs.requestedDeparture" in df.columns:
        df["requestedDeparture"] = to_utc("userInputs.requestedDeparture")
    else:
        df["requestedDeparture"] = pd.NaT

    if "userInputs.kWhRequested" in df.columns:
        df["kWhRequested"] = pd.to_numeric(
            df["userInputs.kWhRequested"], errors="coerce")
    else:
        df["kWhRequested"] = np.nan

    if "userInputs.userID" in df.columns:
        df["userID"] = df["userInputs.userID"].fillna("")
    elif "userID" in df.columns:
        df["userID"] = df["userID"].fillna("")
    else:
        df["userID"] = ""

    df["kWhDelivered"] = pd.to_numeric(
        df.get("kWhDelivered", np.nan), errors="coerce")

    # Date range filter
    if start_date:
        df = df[df["connectionTime"] >= pd.Timestamp(start_date, tz="UTC")]
    if end_date:
        df = df[df["connectionTime"] <= pd.Timestamp(end_date, tz="UTC")]

    return df.sort_values("connectionTime").reset_index(drop=True)


# ── Step 3 : filter invalid sessions ────────────────────────────────────────
def _filter_invalid(df):
    mask = (
        df["connectionTime"].notna()
        & df["disconnectTime"].notna()
        & (df["kWhDelivered"] > 0)
        & (df["disconnectTime"] > df["connectionTime"])
    )
    return df[mask].copy()


# ── Step 4 : outlier removal ─────────────────────────────────────────────────
def _remove_outliers(df, contamination=0.04):
    """Isolation Forest on (duration_min, kWh_delivered)."""
    duration = (df["disconnectTime"] - df["connectionTime"]).dt.total_seconds() / 60
    X = np.column_stack([duration.values, df["kWhDelivered"].values])
    clf  = IsolationForest(contamination=contamination, random_state=42)
    pred = clf.fit_predict(X)
    return df[pred == 1].copy()


# ── Step 5 : discretise to 5-min steps ───────────────────────────────────────
def _discretise(df):
    """
    Map absolute timestamps to integer 5-minute step indices.
    Step 0 = the earliest connectionTime in the dataset.
    """
    t0 = df["connectionTime"].min()

    def to_step(ts):
        if pd.isna(ts):
            return np.nan
        delta = (ts - t0).total_seconds() / 60
        return int(delta // STEP_MINUTES)

    df["arrival_step"]   = df["connectionTime"].apply(to_step).astype(int)

    # Deadline: prefer requestedDeparture, fall back to disconnectTime
    deadline_ts = df["requestedDeparture"].where(
        df["requestedDeparture"].notna(), df["disconnectTime"])
    df["deadline_step"]  = deadline_ts.apply(to_step)
    df["deadline_step"]  = np.maximum(df["deadline_step"],
                                      df["arrival_step"] + 1).astype(int)

    # Energy demand: prefer kWhRequested, fall back to kWhDelivered
    df["energy_kwh"] = df["kWhRequested"].where(
        df["kWhRequested"].notna() & (df["kWhRequested"] > 0),
        df["kWhDelivered"])

    df["max_rate_kw"] = MAX_RATE_KW
    df["claimed"]     = (df["userID"] != "").astype(int)

    # Minimum steps to fully charge
    min_steps = np.ceil(df["energy_kwh"] / (MAX_RATE_KW * STEP_MINUTES / 60))
    df["min_steps_needed"] = min_steps.astype(int)

    df["session_id"] = df.get("sessionID",
                               pd.Series(range(len(df)))).astype(str)

    return df[[
        "session_id", "arrival_step", "deadline_step",
        "energy_kwh", "max_rate_kw", "userID", "claimed",
        "min_steps_needed", "connectionTime"
    ]].copy()


# ── Step 6 : user history features ───────────────────────────────────────────
def _user_history(df):
    """
    Compute rolling per-user averages for claimed sessions:
    mean_dur, mean_energy, mean_departure_hour.
    """
    df = df.sort_values("arrival_step").copy()
    df["duration_steps"] = df["deadline_step"] - df["arrival_step"]
    df["arrival_hour"]   = (df["arrival_step"] * STEP_MINUTES / 60) % 24

    # Shift by 1 so we only use *past* sessions (no data leakage)
    user_stats = (
        df[df["claimed"] == 1]
        .groupby("userID")[["duration_steps", "energy_kwh", "arrival_hour"]]
        .expanding()
        .mean()
        .shift(1)
        .reset_index(level=0, drop=True)
        .rename(columns={
            "duration_steps": "hist_mean_dur",
            "energy_kwh":     "hist_mean_energy",
            "arrival_hour":   "hist_mean_dep_hour",
        })
    )
    df = df.join(user_stats)
    df[["hist_mean_dur", "hist_mean_energy", "hist_mean_dep_hour"]] = \
        df[["hist_mean_dur", "hist_mean_energy", "hist_mean_dep_hour"]].fillna(-1)
    return df


# ── Step 7 : cyclic temporal encoding ────────────────────────────────────────
def _cyclic_features(df):
    """sin/cos encoding for hour-of-day and day-of-week."""
    hour = df["connectionTime"].dt.hour + df["connectionTime"].dt.minute / 60
    dow  = df["connectionTime"].dt.dayofweek          # 0=Monday

    df["hour_sin"] = np.sin(2 * math.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * math.pi * hour / 24)
    df["dow_sin"]  = np.sin(2 * math.pi * dow  / 7)
    df["dow_cos"]  = np.cos(2 * math.pi * dow  / 7)
    return df


# ── Demo / test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Quick test with synthetic data
    from simulator import make_synthetic_sessions
    sessions, n_steps = make_synthetic_sessions(n=200, seed=42)
    print(sessions.head())
    print(f"Steps: {n_steps}")
