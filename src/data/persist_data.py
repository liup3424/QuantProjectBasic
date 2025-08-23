from __future__ import annotations
import argparse
from pathlib import Path
import sys
import pandas_market_calendars as mcal

import pandas as pd

from data_store import DataStore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, required=True)
    ap.add_argument("--start", type=str, required=True)
    ap.add_argument("--end", type=str, required=True)
    ap.add_argument("--symbols", type=str, default=None, help="Comma-separated tickers")
    ap.add_argument("--columns", type=str, default=None, help="Comma-separated column list")
    ap.add_argument(
        "--override",
        action="store_true",
        default=False,
        help="If set, re-download and overwrite existing day files (default: False)",
    )
    args = ap.parse_args()

    # Parse lists
    symbols = None
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    columns = None
    if args.columns:
        columns = [c.strip().lower() for c in args.columns.split(",") if c.strip()]

    # Instantiate your (renamed) data class
    ds = DataStore(
        data_dir=Path(args.data_dir),
        symbols=symbols,     # None -> use UNIVERSE
        columns=columns,     # None -> use DEFAULT_COLUMNS
        is_override=args.override,  # default False unless --override is passed
    )

    nyse = mcal.get_calendar("XNYS")
    sched = nyse.schedule(start_date=args.start, end_date=args.end)
    bizdays = mcal.date_range(sched, frequency="1D")

    # Loop business days and persist
    for day in bizdays:
        ds.set_date(day)

        # If a cache exists and we're NOT overriding, we can short-circuit the log message.
        out_path = ds._daily_path()  # relies on set_date having been called
        had_cache = out_path.exists()

        ds.pull_day()  # loads cache if present & not overriding; else downloads

        # If nothing to save (empty or None), skip
        # NOTE: save_day() in your version returns None; we just use existence to log.
        if ds._day_df is None or ds._day_df.empty:
            print(f"[skip] {day.date()}: no data")
            continue

        # Try to save (respects is_override)
        ds.save_day()

        if had_cache and not args.override:
            print(f"[cache] {day.date()}: kept existing {out_path.name}")
        else:
            # If we overrode or wrote a new file, it should exist now
            if out_path.exists():
                print(f"[ok] {day.date()}: wrote {out_path}")
            else:
                print(f"[skip] {day.date()}: nothing written")


if __name__ == "__main__":
    sys.argv[1:] = [
        '--data-dir', r'/Users/fionaliu/Desktop/git/data',
        '--start', '20240101',
        '--end', '20240110'
    ]
    sys.exit(main())