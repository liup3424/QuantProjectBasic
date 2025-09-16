"""
Factor calculation module for quantitative trading strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional

from .alpha101_factors import Alpha101Calculator


def compute_shadow_features(
    df: pd.DataFrame,
    code_col: str = "code",
    date_col: str = "date",
    short_win: int = 5,
    long_win: int = 20,
) -> pd.DataFrame:
    """
    Compute candlestick shadow and window-normalized features per code.

    Parameters
    ----------
    df : pd.DataFrame
        Input data with required columns: code, date, money, volume, high, low, open, close
    code_col : str
        Name of the security identifier column
    date_col : str
        Name of the date column
    short_win : int
        Short window for rolling calculations
    long_win : int
        Long window for rolling calculations

    Returns
    -------
    pd.DataFrame
        DataFrame with added columns:
        - vwap: Volume-weighted average price
        - upshadow/downshadow: Raw shadow lengths
        - up/down: Shadow lengths normalized by short MA
        - up_mean/up_std/down_mean/down_std: Rolling stats over long window
        - wm_up/wm_down: Wick measurements from close
        - WM_* variants: Normalized wick measurements and their stats
        - Alpha factors from Alpha101

    Notes
    -----
    - Rolling stats use min_periods = window (strict) to avoid early-window bias
    - Divisions by zero are handled gracefully (result -> NaN)
    - Data is sorted by code and date before calculations
    """
    req_cols = {code_col, date_col, "money", "volume", "high", "low", "open", "close"}
    missing = req_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # work on a copy to avoid mutating caller's df
    df = df.copy()

    # ensure date is datetime for proper sorting (does not change original types otherwise)
    if not np.issubdtype(df[date_col].dtype, np.datetime64):
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    # sort & reset
    df = df.sort_values([code_col, date_col], ascending=True).reset_index(drop=True)

    # helper: safe division (avoid /0)
    def sdiv(a, b):
        return a / b.replace(0, np.nan)

    g = df.groupby(code_col, sort=False)

    # Base features
    df["vwap"] = sdiv(df["money"], df["volume"])

    df["upshadow"] = df["high"] - df[["close", "open"]].max(axis=1)
    df["downshadow"] = df[["close", "open"]].min(axis=1) - df["low"]

    upshadow_ma_s = g["upshadow"].transform(lambda x: x.rolling(short_win, min_periods=short_win).mean())
    downshadow_ma_s = g["downshadow"].transform(lambda x: x.rolling(short_win, min_periods=short_win).mean())

    df["up"] = sdiv(df["upshadow"], upshadow_ma_s)
    df["down"] = sdiv(df["downshadow"], downshadow_ma_s)

    df["up_mean"] = g["up"].transform(lambda x: x.rolling(long_win, min_periods=long_win).mean())
    df["up_std"]  = g["up"].transform(lambda x: x.rolling(long_win, min_periods=long_win).std())

    df["down_mean"] = g["down"].transform(lambda x: x.rolling(long_win, min_periods=long_win).mean())
    df["down_std"]  = g["down"].transform(lambda x: x.rolling(long_win, min_periods=long_win).std())

    # Wick (upper/lower from close)
    df["wm_up"] = df["high"] - df["close"]
    df["wm_down"] = df["close"] - df["low"]

    wm_up_ma_s = g["wm_up"].transform(lambda x: x.rolling(short_win, min_periods=short_win).mean())
    wm_down_ma_s = g["wm_down"].transform(lambda x: x.rolling(short_win, min_periods=short_win).mean())

    df["WM_up"] = sdiv(df["wm_up"], wm_up_ma_s)
    df["WM_down"] = sdiv(df["wm_down"], wm_down_ma_s)

    df["WM_up_mean"]  = g["WM_up"].transform(lambda x: x.rolling(long_win, min_periods=long_win).mean())
    df["WM_up_std"]   = g["WM_up"].transform(lambda x: x.rolling(long_win, min_periods=long_win).std())
    df["WM_down_mean"]= g["WM_down"].transform(lambda x: x.rolling(long_win, min_periods=long_win).mean())
    df["WM_down_std"] = g["WM_down"].transform(lambda x: x.rolling(long_win, min_periods=long_win).std())

    # Calculate alpha factors
    # First ensure we have the required columns
    df['returns'] = g['close'].pct_change()  # Required for alpha calculations
    alpha_calculator = Alpha101Calculator(df)
    alpha_factors = alpha_calculator.calculate_all()
    
    # Convert alpha factors dictionary to DataFrame
    alpha_df = pd.DataFrame(alpha_factors)
    
    # Combine features
    result = pd.concat([df, alpha_df], axis=1)
    
    # Clean up infinities from any divisions and keep NaN where undefined
    result.replace([np.inf, -np.inf], np.nan, inplace=True)

    return result
