"""
Implementation of alpha factors from Alpha101.
These factors are based on common market data indicators and technical analysis.
"""

from typing import Dict, Union
import numpy as np
import pandas as pd


class Alpha101Calculator:
    def __init__(self, df: pd.DataFrame):
        """
        Initialize Alpha101Calculator with market data.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing required market data columns:
            - close: Closing prices
            - open: Opening prices
            - high: High prices
            - low: Low prices
            - volume: Trading volumes
            - vwap: Volume-weighted average price
            - returns: Daily returns
        """
        self.df = df.copy()
        self._validate_data()
        
    def _validate_data(self):
        """Validate required columns exist in the data."""
        required = {'close', 'open', 'high', 'low', 'volume', 'vwap', 'returns'}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
    def _rank(self, x: pd.Series) -> pd.Series:
        """Cross-sectional percentile rank."""
        return x.rank(pct=True)
    
    def _correlation(self, x: pd.Series, y: pd.Series, window: int) -> pd.Series:
        """Rolling correlation."""
        return x.rolling(window).corr(y)
    
    def _covariance(self, x: pd.Series, y: pd.Series, window: int) -> pd.Series:
        """Rolling covariance."""
        return x.rolling(window).cov(y)
    
    def _ts_rank(self, x: pd.Series, window: int) -> pd.Series:
        """Time-series rank."""
        return x.rolling(window).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])

    def _decay_linear(self, x: pd.Series, window: int) -> pd.Series:
        """Linear decay weighted sum."""
        weights = np.arange(1, window + 1)
        return x.rolling(window).apply(lambda x: np.dot(x, weights) / weights.sum())

    def alpha001(self) -> pd.Series:
        """
        Alpha#1: (rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 5)) - 0.5)
        """
        inner = np.where(self.df['returns'] < 0,
                        self.df['returns'].rolling(20).std(),
                        self.df['close'])
        return self._rank(self.df['close'].rolling(5).apply(np.argmax)) - 0.5
    
    def alpha002(self) -> pd.Series:
        """
        Alpha#2: (-1 * correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6))
        """
        df = self.df
        volume_delta = np.log(df['volume']).diff(2)
        price_change = (df['close'] - df['open']) / df['open']
        return -1 * self._correlation(self._rank(volume_delta),
                                   self._rank(price_change), 6)

    def alpha003(self) -> pd.Series:
        """
        Alpha#3: (-1 * correlation(rank(open), rank(volume), 10))
        """
        return -1 * self._correlation(self._rank(self.df['open']),
                                   self._rank(self.df['volume']), 10)

    def calculate_all(self) -> Dict[str, pd.Series]:
        """
        Calculate all implemented alpha factors.
        
        Returns
        -------
        Dict[str, pd.Series]
            Dictionary mapping factor names to their values
        """
        factors = {}
        
        # Get all methods starting with 'alpha'
        alpha_methods = [method for method in dir(self) 
                        if method.startswith('alpha') and callable(getattr(self, method))]
        
        for method in alpha_methods:
            try:
                factors[method] = getattr(self, method)()
            except Exception as e:
                print(f"Warning: Failed to calculate {method}: {str(e)}")
                
        return factors