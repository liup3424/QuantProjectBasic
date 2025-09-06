import os
import glob
import pandas as pd
from pathlib import Path
from datetime import datetime, date


def get_file_date(path) -> date:
    """
    Extract date from filename like 'YYYY-MM-DD.csv' or 'YYYY-MM-DD.parquet'.
    """
    p = Path(path)
    try:
        return datetime.strptime(p.stem, "%Y-%m-%d").date()
    except ValueError:
        raise ValueError(f"Filename {p.name} does not contain a valid YYYY-MM-DD date")


def load_data(basePath, date_col='date', code_col='code', startDate=None, endDate=None):
    base = Path(basePath)

    dfs = []
    for f in base.glob("*"):
        if f.suffix.lower() not in [".csv", ".parquet"]:
            continue

        file_date = get_file_date(f)

        if startDate and file_date < pd.to_datetime(startDate).date():
            continue
        if endDate and file_date > pd.to_datetime(endDate).date():
            continue

        if f.suffix.lower() == ".csv":
            df = pd.read_csv(f)
        else:
            df = pd.read_parquet(f)
        dfs.append(df)

    if not dfs:
        raise ValueError("No files found within given date range.")

    frame = pd.concat(dfs, axis=0, ignore_index=True)

    missing = [c for c in [date_col, code_col] if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}. "
                         f"Available columns: {list(frame.columns)}")

    # Parse date column
    frame[date_col] = pd.to_datetime(frame[date_col])
    return frame.sort_values([date_col, code_col]).reset_index(drop=True)

