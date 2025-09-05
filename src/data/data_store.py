from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence, Dict, Set, List
import time, random, requests

import pandas as pd
import yfinance as yf

from src.data.const import UNIVERSE, DEFAULT_COLUMNS
from src.data.utils import _norm_cols, _biz_days

# Tune these if needed
BATCH_SIZE   = 50         # number of tickers per request
MAX_RETRIES  = 5
BASE_SLEEP_S = 2.0        # base backoff seconds

def _chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i+n]

def _make_session():
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome Safari"
    })
    return s


@dataclass
class DataStore:
    """
    Daily equities ETL (one day at a time).
    - Files: {data_dir}/raw/YYYY-MM-DD.parquet
    - Index in file: symbol
    - symbols=None -> use UNIVERSE; columns=None -> DEFAULT_COLUMNS
    - is_override=False -> if file exists, load cache & skip re-download/save
    """
    data_dir: str | Path
    symbols: Sequence[str] | None = None
    columns: Sequence[str] | None = None
    auto_adjust: bool = True
    is_override: bool = False

    # internal state
    _date: Optional[pd.Timestamp] = field(default=None, init=False, repr=False)
    _raw_dir: Path = field(init=False, repr=False)
    _day_df: Optional[pd.DataFrame] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self.data_dir = Path(self.data_dir)
        self._raw_dir = self.data_dir / "raw"
        self._raw_dir.mkdir(parents=True, exist_ok=True)

        self.symbols = [s.strip().upper() for s in (self.symbols or UNIVERSE)]
        self.columns = list(self.columns) if self.columns is not None else list(DEFAULT_COLUMNS)

    # ---------- config ----------
    def set_date(self, day: str | pd.Timestamp) -> "DataStore":
        """Set single trading date for pull/save operations."""
        self._date = pd.Timestamp(day).normalize()
        self._day_df = None
        return self

    # ---------- paths ----------
    def _daily_path(self) -> Path:
        if self._date is None:
            raise ValueError("Date not set. Call set_date('YYYY-MM-DD') first.")
        return self._raw_dir / f"{self._date.date().isoformat()}.parquet"

    def pull_day(self) -> None:
        """
        Populate self._day_df for the set date.
        - Uses cache if exists and is_override=False
        - Otherwise, downloads symbols in batches (BATCH_SIZE) with retries/backoff
        - Result: self._day_df indexed by 'symbol' with your selected columns
        """
        if self._date is None:
            raise ValueError("Date not set. Call set_date(...) first.")

        out_path = self._daily_path()
        if out_path.exists() and not self.is_override:
            # use cache
            try:
                self._day_df = pd.read_parquet(out_path)  # index = symbol
                return
            except Exception as e:
                print(f"[warn] cache read failed for {out_path.name}: {e}; re-downloading")

        # Ensure we have a shared session
        if not hasattr(self, "session") or self.session is None:
            self.session = _make_session()

        day = self._date.normalize()
        start = day.date().isoformat()
        end = (day + pd.Timedelta(days=1)).date().isoformat()

        rows = []
        for batch in _chunks(self.symbols, BATCH_SIZE):
            # Retry the whole batch if rate-limited
            last_err = None
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    # threads=False to avoid many parallel requests
                    df = yf.download(
                        tickers=batch,
                        start=start, end=end,
                        interval="1d",
                        auto_adjust=self.auto_adjust,
                        group_by="ticker",
                        progress=False,
                        session=self.session,
                        threads=False,
                        timeout=3000,
                    )
                    if df.empty:
                        # nothing for this batch on this day (holiday etc.)
                        break

                    # Normalize into index=symbol, cols = requested columns
                    if isinstance(df.columns, pd.MultiIndex):
                        # df columns like: (AAPL, Open), (AAPL, Close), ...
                        for s in batch:
                            if s in df.columns.get_level_values(0):
                                sub = df[s].copy()
                                if sub.empty:
                                    continue
                                sub.columns = [c.lower().replace("adj close", "adj_close") for c in sub.columns]
                                # keep only requested columns if present
                                keep = [c for c in self.columns if c in sub.columns]
                                if not keep:
                                    continue
                                # there should be a single row for the day
                                sub = sub[keep].reset_index(drop=True)
                                sub.insert(0, "symbol", s)
                                rows.append(sub)
                    else:
                        # Single-ticker frame
                        sub = df.copy()
                        sub.columns = [c.lower().replace("adj close", "adj_close") for c in sub.columns]
                        keep = [c for c in self.columns if c in sub.columns]
                        if keep and not sub.empty:
                            sub = sub[keep].reset_index(drop=True)
                            # figure out which ticker this is
                            s = batch[0]
                            sub.insert(0, "symbol", s)
                            rows.append(sub)

                    # polite tiny pause between batches to avoid burst
                    time.sleep(5)
                    break  # success, leave retry loop

                except yf.shared._exceptions.YFRateLimitError as e:  # type: ignore[attr-defined]
                    last_err = e
                    sleep_s = BASE_SLEEP_S * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
                    print(
                        f"[rate-limit] batch {batch[0]}..{batch[-1]} attempt {attempt}/{MAX_RETRIES}; sleeping {sleep_s:.2f}s")
                    time.sleep(sleep_s)
                except Exception as e:
                    last_err = e
                    sleep_s = BASE_SLEEP_S * (2 ** (attempt - 1)) + random.uniform(0, 0.3)
                    print(
                        f"[warn] batch {batch[0]}..{batch[-1]} attempt {attempt}/{MAX_RETRIES} failed: {e}; sleep {sleep_s:.2f}s")
                    time.sleep(sleep_s)

            else:
                # exhausted retries for this batch
                print(f"[warn] batch {batch[0]}..{batch[-1]} failed after retries: {last_err}")

        if not rows:
            self._day_df = pd.DataFrame(index=pd.Index([], name="symbol"))
            return

        day_df = pd.concat(rows, ignore_index=True).set_index("symbol").sort_index()
        self._day_df = day_df
        return


    def save_day(self) -> None:
        """
        Save the in-memory day dataframe to parquet.
        - Respects is_override: if file exists and not overriding, does nothing.
        Returns the path written (or None if skipped/empty).
        """
        if self._date is None:
            raise ValueError("Date not set. Call set_date(...) first.")
        if self._day_df is None or self._day_df.empty:
            # nothing to write
            return

        out_path = self._daily_path()
        if out_path.exists() and not self.is_override:
            # keep cached; don't overwrite
            return

        out_path.parent.mkdir(parents=True, exist_ok=True)
        self._day_df.to_parquet(out_path)
        return