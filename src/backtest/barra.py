from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Union, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from tqdm import tqdm


class NeutralizationProcessor:
    """
    Per-date cross-sectional neutralization of factor columns against Barra controls.

    For each date/file shared by factor_path & barra_path:
      1) Optionally winsorize each factor at mean ± 3*std (per cross section).
      2) Optionally normalize each factor (mode: 1=min-max, 2=z-score, 3=magnitude).
      3) Regress y=factor on controls (barra columns) and take residuals as neutralized factor.

    Saves a CSV per date with columns '<factor>_neu' in output_folder.
    """

    def __init__(
        self,
        barra_path: str,
        factor_path: str,
        output_folder: str,
        mode: int = 0,
        barra_cols: Union[str, Iterable[str]] = "size",
        winsorize: bool = True,
        use_tqdm: bool = True,
    ):
        """
        Parameters
        ----------
        barra_path : str
            Folder with Barra exposures (one CSV per date).
        factor_path : str
            Folder with raw factors (one CSV per date).
        output_folder : str
            Destination folder for neutralized factors.
        mode : int
            Normalization mode for factors before regression:
             0 = none
             1 = min-max in [0,1] (per cross section)
             2 = z-score (mean 0, std 1)
             3 = magnitude scaling by 10**ceil(log10(max_abs))
        barra_cols : str | Iterable[str]
            Column(s) in Barra file used as controls.
        winsorize : bool
            If True, clip each factor at mean ± 3*std per cross section.
        use_tqdm : bool
            If True, show a progress bar.
        """
        self.barra_path = Path(barra_path)
        self.factor_path = Path(factor_path)
        self.output_folder = Path(output_folder)
        self.mode = int(mode)
        self.winsorize = bool(winsorize)
        self.use_tqdm = bool(use_tqdm)

        if isinstance(barra_cols, str):
            self.barra_cols: List[str] = [barra_cols]
        else:
            self.barra_cols = list(barra_cols)

        self.output_folder.mkdir(parents=True, exist_ok=True)

    # ------------------------- helpers -------------------------

    @staticmethod
    def _winsorize_series(s: pd.Series, k: float = 3.0) -> pd.Series:
        mu = s.mean()
        sd = s.std()
        if not np.isfinite(sd) or sd == 0:
            return s  # nothing to clip
        lo, hi = mu - k * sd, mu + k * sd
        return s.clip(lo, hi)

    def _normalize_series(self, s: pd.Series) -> pd.Series:
        if self.mode == 0:
            return s
        if self.mode == 1:
            s_min, s_max = s.min(), s.max()
            rng = s_max - s_min
            return (s - s_min) / rng if np.isfinite(rng) and rng != 0 else s * 0.0
        if self.mode == 2:
            mu, sd = s.mean(), s.std()
            return (s - mu) / sd if np.isfinite(sd) and sd != 0 else s * 0.0
        if self.mode == 3:
            max_abs = np.nanmax(np.abs(s.values)) if len(s) else 0.0
            if not np.isfinite(max_abs) or max_abs == 0:
                return s  # nothing to scale
            scale_pow = int(np.ceil(np.log10(max_abs)))
            return s / (10 ** scale_pow)
        raise ValueError(f"Unknown normalization mode: {self.mode}")

    def _neutralize_one(self, y: pd.Series, X: pd.DataFrame) -> pd.Series:
        """
        Cross-sectional regression residuals of y on X (with intercept).
        Aligns index, drops rows with NaN in y or X.
        """
        df = pd.concat([y, X], axis=1)
        df = df.replace([np.inf, -np.inf], np.nan).dropna(how="any")
        if df.empty:
            # Return NaN residuals aligned to original index
            return pd.Series(index=y.index, dtype=float)

        y_clean = df.iloc[:, 0]
        X_clean = sm.add_constant(df.iloc[:, 1:], has_constant="add")
        model = sm.OLS(y_clean, X_clean)
        res = model.fit()
        resid = y_clean - res.fittedvalues
        # Reindex to original y's index
        return resid.reindex(y.index)

    # ------------------------- main API -------------------------

    def neutralize_frame(self, factor_df: pd.DataFrame, barra_df: pd.DataFrame) -> pd.DataFrame:
        """
        Neutralize ALL factor columns in factor_df using barra_df[self.barra_cols].
        Returns a DataFrame of residuals with columns '<factor>_neu'.
        """
        # Validate barra controls
        missing = [c for c in self.barra_cols if c not in barra_df.columns]
        if missing:
            raise KeyError(f"Barra control column(s) not found: {missing}")

        X = barra_df[self.barra_cols].copy()

        # Ensure numeric dtypes (coerce non-numeric to NaN)
        X = X.apply(pd.to_numeric, errors="coerce")

        out = pd.DataFrame(index=factor_df.index)

        for col in factor_df.columns:
            y = pd.to_numeric(factor_df[col], errors="coerce")

            if self.winsorize:
                y = self._winsorize_series(y)

            y = self._normalize_series(y)

            resid = self._neutralize_one(y, X)
            out[f"{col}_neu"] = resid

        return out

    def process_data(self) -> None:
        """
        Iterate over dates (filenames shared by both folders),
        load factor & barra frames, align on common index, neutralize, and write CSV.
        """
        factor_files = {p.name for p in self.factor_path.glob("*.csv")}
        barra_files = {p.name for p in self.barra_path.glob("*.csv")}
        common_files = sorted(factor_files & barra_files)

        iterator = tqdm(common_files, desc="Neutralizing") if self.use_tqdm else common_files

        for fname in iterator:
            # Load
            factor = pd.read_csv(self.factor_path / fname, index_col=0)
            barra = pd.read_csv(self.barra_path / fname, index_col=0)

            # Align on common universe
            common_idx = factor.index.intersection(barra.index)
            if common_idx.empty:
                # Nothing to do for this date
                continue

            factor_cs = factor.loc[common_idx].copy()
            barra_cs = barra.loc[common_idx].copy()

            # Clean obvious bad values
            factor_cs = factor_cs.replace([np.inf, -np.inf], np.nan)
            barra_cs = barra_cs.replace([np.inf, -np.inf], np.nan)

            # Neutralize cross-section
            neu = self.neutralize_frame(factor_cs, barra_cs)

            # Save
            (self.output_folder / fname).parent.mkdir(parents=True, exist_ok=True)
            neu.to_csv(self.output_folder / fname)
