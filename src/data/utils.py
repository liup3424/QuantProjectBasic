import pandas as pd
from typing import Iterable, Optional, Sequence, Dict, Set, List


def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize yfinance columns to lowercase OHLCV (+ adj_close)."""
    if df.empty:
        return df
    df = df.rename(columns=str.lower).rename(columns={"adj close": "adj_close"})
    selectedCol = [c for c in ("open", "high", "low", "close", "adj_close", "volume") if c in df.columns]
    return df[selectedCol] if selectedCol else df


def _biz_days(start: str, end: str) -> List[str]:
    rng = pd.date_range(start, end, freq="B")
    return [d.date().isoformat() for d in rng]

def parse_boolean_argument(input):
    input = str(input).lower()
    if input in ['yes', 'true', '1']:
        return True
    if input in ['no', 'false', '0']:
        return False
    raise ValueError('Unsupported type for boolean argument')