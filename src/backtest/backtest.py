from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt


class BacktestAnalyzer:
    """
    Concatenate daily factor/return/untradable (UD) snapshots, build decile groups,
    compute PnL/profitability per group, and summarize IC/ICIR & performance.

    Assumptions
    ----------
    - Files under factor_path are named like YYYY-MM-DD.csv (one per day).
    - ret_path and ud_path contain files for the *next* trading day (alignment).
    - 'code' is the security key in the CSVs.
    - Return column is `target_col` (default: '1vwap_pct').
    - UD file contains columns ['zt', 'dt', 'paused'] where 0 means tradable.

    Notes
    -----
    - We form quantile groups with `n_bins` (default: 10 = deciles).
    - We compute two per-group series:
        * pnl: mean of the raw target (% return)
        * pro: mean of the demeaned target (neutralized by cross-sectional mean)
    """

    def __init__(
        self,
        start_date: str,
        end_date: str,
        trade_date_pickle: str,
        factor_path: str,
        ret_path: str,
        ud_path: str,
        n_bins: int = 10,
        target_col: str = "1vwap_pct",
        code_col: str = "code",
        verbose: bool = True,
    ):
        self.start_date = str(start_date)
        self.end_date = str(end_date)
        self.trade_date_pickle = trade_date_pickle
        self.factor_path = Path(factor_path)
        self.ret_path = Path(ret_path)
        self.ud_path = Path(ud_path)
        self.n_bins = int(n_bins)
        self.target_col = target_col
        self.code_col = code_col
        self.verbose = verbose

        # Outputs
        self.pnl: Dict[str, pd.DataFrame] = {}   # factor -> dataframe (date x group)
        self.pro: Dict[str, pd.DataFrame] = {}   # factor -> dataframe (date x group)
        self.ic: Dict[str, List[float]] = {}     # factor -> list of daily IC values
        self.summary: pd.DataFrame = pd.DataFrame()

    # ---------------------------
    # Loading & preparation
    # ---------------------------

    def _load_trading_calendar(self) -> List[str]:
        """Load the trading calendar (list-like) from a pickle."""
        cal = pickle.load(open(self.trade_date_pickle, "rb"))
        # normalize to list of strings YYYY-MM-DD
        cal = [str(x) for x in cal]
        return cal

    def _daily_files(self) -> List[str]:
        """List factor files (YYYY-MM-DD.csv) within [start_date, end_date]."""
        files = sorted([p.name for p in self.factor_path.glob("*.csv")])
        # keep only date-stamped files in range
        return [
            f for f in files
            if (len(f) >= 14 and f.endswith(".csv") and self.start_date <= f[:-4] <= self.end_date)
        ]

    def prepare_data(self) -> None:
        """
        Main loop:
        - For each factor date t, align ret/ud at next trading date t+1.
        - Merge by code, drop untradables, sanitize infs, and compute per-factor metrics.
        """
        trade_cal = self._load_trading_calendar()
        factor_files = self._daily_files()

        iterator = tqdm(factor_files, desc="Processing") if self.verbose else factor_files

        for fname in iterator:
            date_t = fname[:-4]  # 'YYYY-MM-DD'
            # find next trading date
            try:
                i = trade_cal.index(date_t)
                next_date = trade_cal[i + 1]
            except (ValueError, IndexError):
                # skip if date not in calendar OR last day has no next_date
                continue

            # make sure corresponding files exist for next_date
            ret_file = self.ret_path / f"{next_date}.csv"
            ud_file = self.ud_path / f"{next_date}.csv"
            factor_file = self.factor_path / fname

            if not (ret_file.exists() and ud_file.exists()):
                # quietly skip missing pairs
                continue

            # load
            factor_df = pd.read_csv(factor_file, index_col=0)         # index: code?
            ret_df    = pd.read_csv(ret_file, index_col=0)
            ud_df     = pd.read_csv(ud_file, index_col=0)

            # ensure code is index for UD
            if self.code_col in ud_df.columns:
                ud_df = ud_df.set_index(self.code_col)

            # sanity: required columns
            for required in ["zt", "dt", "paused"]:
                if required not in ud_df.columns:
                    raise KeyError(f"Column '{required}' missing in UD: {ud_file}")

            if self.target_col not in ret_df.columns:
                raise KeyError(f"Target column '{self.target_col}' missing in returns: {ret_file}")

            # align indexes
            common = factor_df.index.intersection(ret_df.index)
            if len(common) == 0:
                # nothing to do for this day
                continue

            merged = pd.concat(
                [
                    factor_df.loc[common],
                    ret_df.loc[common, [self.target_col]],
                    ud_df.loc[common, ["zt", "dt", "paused"]],
                ],
                axis=1,
                join="inner",
            )

            # tradable filter & clean infs
            tradable = (merged["zt"] == 0) & (merged["dt"] == 0) & (merged["paused"] == 0)
            final = merged.loc[tradable].replace([np.inf, -np.inf], np.nan).dropna(subset=[self.target_col])

            # compute per-factor daily metrics
            if not final.empty:
                self._calculate_metrics_for_day(final, date_t, factor_df.columns.tolist())

    # ---------------------------
    # Per-day factor metrics
    # ---------------------------

    def _calculate_metrics_for_day(self, df: pd.DataFrame, date_t: str, factor_cols: List[str]) -> None:
        """
        For each factor column:
          - form n_bins quantile groups (labels 1..n_bins)
          - compute mean of {pnl=target, pro=target - mean(target)}
          - append to self.pnl[self.pro][factor] with index=date_t
          - compute daily IC = corr(factor, target)
        """
        # demeaned (cross-sectional) target for "pro"
        df["_pro"] = df[self.target_col] - df[self.target_col].mean()

        for col in factor_cols:
            # init containers
            if col not in self.pnl:
                self.pnl[col] = pd.DataFrame()
                self.pro[col] = pd.DataFrame()
                self.ic[col] = []

            # need enough unique values to form quantile bins
            if df[col].dropna().nunique() < self.n_bins:
                continue

            # build quantile groups (1..n_bins)
            try:
                df[f"{col}__group"] = pd.qcut(
                    df[col],
                    q=self.n_bins,
                    labels=False,
                    duplicates="drop"
                ) + 1
            except ValueError:
                # qcut may fail if not enough unique cut points
                continue

            # aggregate by group
            grouped = df.groupby(f"{col}__group", observed=True).agg(
                pnl=(self.target_col, "mean"),
                pro=("_pro", "mean"),
            )  # index is 1..n_bins

            # to wide (1 row, n_bins cols)
            pnl_row = grouped["pnl"].to_frame().T
            pnl_row.index = [date_t]
            pnl_row.index.name = None

            pro_row = grouped["pro"].to_frame().T
            pro_row.index = [date_t]
            pro_row.index.name = None

            # append
            self.pnl[col] = pd.concat([self.pnl[col], pnl_row], axis=0, sort=False)
            self.pro[col] = pd.concat([self.pro[col], pro_row], axis=0, sort=False)

            # daily IC
            ic_val = df[col].corr(df[self.target_col])
            if pd.notna(ic_val):
                self.ic[col].append(float(ic_val))

    # ---------------------------
    # Effectiveness & plotting
    # ---------------------------

    def calculate_effectiveness(self, plot: bool = True, xtick_step: int = 30) -> pd.DataFrame:
        """
        For each factor:
          - If mean(IC) > 0, reverse group order so Group 1 is always the "long" side.
          - Plot cumulative returns per group for PRO and PNL (optional).
          - Compute performance stats for Group 1 (long) and Hedge (Group1 - GroupN).
          - Return a summary dataframe.
        """
        summaries = []

        for factor, ic_series in self.ic.items():
            if len(ic_series) == 0:
                continue

            ic_mean = float(np.mean(ic_series))
            ic_std = float(np.std(ic_series)) if len(ic_series) > 1 else np.nan
            icir = (ic_mean / ic_std) if (ic_std and ic_std > 0) else np.nan

            # Ensure consistent group order:
            # If IC mean > 0, higher factor predicts higher return -> reverse columns
            if ic_mean > 0:
                self.pro[factor] = self._reverse_group_cols(self.pro[factor])
                self.pnl[factor] = self._reverse_group_cols(self.pnl[factor])

            # --- PRO cumulative plots
            pro_df = self.pro[factor].copy()
            pro_df = pro_df.sort_index()
            pro_cum = (1.0 + pro_df.fillna(0.0)).cumprod()

            if plot and not pro_cum.empty:
                self._plot_cumcurve(pro_cum, title=f"{factor} — PRO cumulative", ylabel="PRO", xtick_step=xtick_step)

            # --- PNL cumulative plots (+ hedge)
            pnl_df = self.pnl[factor].copy().sort_index()
            # ensure group labels are ints starting at 1
            group_cols = self._group_cols(pnl_df)
            if len(group_cols) == 0:
                continue

            first_group = group_cols[0]
            last_group = group_cols[-1]
            pnl_df["hedge"] = pnl_df[first_group] - pnl_df[last_group]

            pnl_cum = (1.0 + pnl_df.fillna(0.0)).cumprod()

            if plot and not pnl_cum.empty:
                self._plot_cumcurve(pnl_cum, title=f"{factor} — PNL cumulative", ylabel="PNL", xtick_step=xtick_step)

            # --- Stats
            def ann_mean(x: pd.Series) -> float:
                return float(x.mean() * 252)

            def ann_vol(x: pd.Series) -> float:
                return float(x.std(ddof=1) * np.sqrt(252))

            def sharpe(x: pd.Series) -> float:
                vol = ann_vol(x)
                return float(ann_mean(x) / vol) if vol > 0 else np.nan

            def max_drawdown(cum: pd.Series) -> float:
                dd = (cum.cummax() - cum) / cum.cummax()
                return float(dd.max() * 100.0) if len(dd) else np.nan

            long_stats = {
                "ann_return": ann_mean(pnl_df[first_group]),
                "ann_vol": ann_vol(pnl_df[first_group]),
                "sharpe": sharpe(pnl_df[first_group]),
                "max_dd_%": max_drawdown(pnl_cum[first_group]),
            }

            hedge_stats = {
                "ann_return": ann_mean(pnl_df["hedge"]),
                "ann_vol": ann_vol(pnl_df["hedge"]),
                "sharpe": sharpe(pnl_df["hedge"]),
                "max_dd_%": max_drawdown(pnl_cum["hedge"]),
            }

            summaries.append(
                {
                    "factor": factor,
                    "IC_mean": round(ic_mean, 4),
                    "ICIR": round(icir, 4) if pd.notna(icir) else np.nan,
                    "long_ann_return": round(long_stats["ann_return"], 4),
                    "long_ann_vol": round(long_stats["ann_vol"], 4),
                    "long_sharpe": round(long_stats["sharpe"], 4) if pd.notna(long_stats["sharpe"]) else np.nan,
                    "long_max_dd_%": round(long_stats["max_dd_%"], 4),
                    "hedge_ann_return": round(hedge_stats["ann_return"], 4),
                    "hedge_ann_vol": round(hedge_stats["ann_vol"], 4),
                    "hedge_sharpe": round(hedge_stats["sharpe"], 4) if pd.notna(hedge_stats["sharpe"]) else np.nan,
                    "hedge_max_dd_%": round(hedge_stats["max_dd_%"], 4),
                }
            )

        self.summary = pd.DataFrame(summaries).sort_values(by=["IC_mean"], ascending=False).reset_index(drop=True)
        return self.summary

    # ---------------------------
    # Utilities
    # ---------------------------

    @staticmethod
    def _reverse_group_cols(df: pd.DataFrame) -> pd.DataFrame:
        """Reverse only the group-number columns (leave any extras like 'hedge')."""
        cols = [c for c in df.columns if BacktestAnalyzer._is_int_like(c)]
        others = [c for c in df.columns if c not in cols]
        cols_sorted = sorted(cols, key=lambda x: int(x))
        return df[cols_sorted[::-1] + others]

    @staticmethod
    def _group_cols(df: pd.DataFrame) -> List[int]:
        """Return sorted integer-like group columns (e.g., 1..n)."""
        cols = [c for c in df.columns if BacktestAnalyzer._is_int_like(c)]
        return sorted(cols, key=lambda x: int(x))

    @staticmethod
    def _is_int_like(x) -> bool:
        try:
            int(x)
            return True
        except Exception:
            return False

    @staticmethod
    def _plot_cumcurve(cum_df: pd.DataFrame, title: str, ylabel: str, xtick_step: int = 30) -> None:
        dates = cum_df.index
        plt.figure(figsize=(12, 6))
        for col in cum_df.columns:
            plt.plot(dates, cum_df[col], label=str(col))
        plt.title(title)
        plt.xlabel("date")
        plt.ylabel(ylabel)
        plt.grid(True)
        if len(dates) > 0:
            plt.xticks(dates[::max(1, xtick_step)], rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.show()



if __name__ == "__main__":
    start_date = "2020-01-01"
    end_date = "2023-11-25"
    trade_date_pickle = "./data/date.pkl"
    factor_path = "./new_factor_neutralization/upshadow"
    ret_path = "./data/data_ret"
    ud_path = "./data/data_ud_new"

    # Instantiate analyzer (new class uses n_bins instead of group list)
    analyzer = BacktestAnalyzer(
        start_date=start_date,
        end_date=end_date,
        trade_date_pickle=trade_date_pickle,
        factor_path=factor_path,
        ret_path=ret_path,
        ud_path=ud_path,
        n_bins=10,
        target_col="1vwap_pct",
        code_col="code",
        verbose=True,
    )

    # Run pipeline
    analyzer.prepare_data()
    summary = analyzer.calculate_effectiveness(plot=True)

    # Show results
    print(summary)