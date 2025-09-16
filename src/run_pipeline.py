"""
Main pipeline script that orchestrates the entire process:
1. Data loading
2. Feature calculation
3. Barra risk factor neutralization
4. Backtesting
"""

import os
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional

from data.data_store import DataStore
from data.load_data import load_data
from features.calculate_factor import compute_shadow_features
from backtest.barra import NeutralizationProcessor
from backtest.backtest import BacktestAnalyzer


class QuantPipeline:
    def __init__(
        self,
        data_dir: str,
        output_dir: str,
        start_date: str,
        end_date: str,
        universe: Optional[list[str]] = None,
        barra_factors: Optional[list[str]] = None
    ):
        """
        Initialize the quantitative analysis pipeline.

        Parameters
        ----------
        data_dir : str
            Base directory for raw data
        output_dir : str
            Directory for output files
        start_date : str
            Start date in YYYY-MM-DD format
        end_date : str
            End date in YYYY-MM-DD format
        universe : Optional[list[str]]
            List of stock symbols to analyze. If None, uses default universe
        barra_factors : Optional[list[str]]
            List of Barra risk factors to use for neutralization
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.start_date = start_date
        self.end_date = end_date
        self.universe = universe
        self.barra_factors = barra_factors or ['size', 'beta', 'momentum', 'volatility']
        
        # Create output directories
        self.features_dir = self.output_dir / 'features'
        self.neutral_dir = self.output_dir / 'neutralized'
        self.backtest_dir = self.output_dir / 'backtest'
        
        for d in [self.features_dir, self.neutral_dir, self.backtest_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def load_market_data(self) -> None:
        """
        Load and prepare market data using DataStore.
        """
        print("Loading market data...")
        self.data_store = DataStore(
            data_dir=self.data_dir,
            symbols=self.universe,
            is_override=False
        )
        
        # Download/load data for each day in range
        current_date = pd.Timestamp(self.start_date)
        end_date = pd.Timestamp(self.end_date)
        
        while current_date <= end_date:
            print(f"Processing {current_date.date()}")
            self.data_store.set_date(current_date)
            self.data_store.pull_day()
            self.data_store.save_day()
            current_date += pd.Timedelta(days=1)

    def calculate_features(self) -> None:
        """
        Calculate alpha factors and technical features.
        """
        print("Calculating features...")
        raw_data_files = sorted(self.data_dir.glob("raw/*.parquet"))
        
        for file in raw_data_files:
            date = file.stem  # YYYY-MM-DD
            if not (self.start_date <= date <= self.end_date):
                continue
                
            # Load daily data
            df = pd.read_parquet(file)
            
            # Calculate features
            features = compute_shadow_features(df)
            
            # Save features
            out_path = self.features_dir / f"{date}.parquet"
            features.to_parquet(out_path)

    def neutralize_factors(self) -> None:
        """
        Apply Barra risk factor neutralization.
        """
        print("Neutralizing factors...")
        processor = NeutralizationProcessor(
            barra_path=str(self.data_dir / "barra"),
            factor_path=str(self.features_dir),
            output_folder=str(self.neutral_dir),
            barra_cols=self.barra_factors,
            winsorize=True,
            use_tqdm=True
        )
        processor.process_data()

    def run_backtest(self) -> pd.DataFrame:
        """
        Run backtest analysis on neutralized factors.
        
        Returns
        -------
        pd.DataFrame
            Summary of backtest results
        """
        print("Running backtest analysis...")
        analyzer = BacktestAnalyzer(
            start_date=self.start_date,
            end_date=self.end_date,
            trade_date_pickle=str(self.data_dir / "trade_dates.pkl"),
            factor_path=str(self.neutral_dir),
            ret_path=str(self.data_dir / "returns"),
            ud_path=str(self.data_dir / "untradeable"),
            n_bins=10,
            target_col="return_1d",
            code_col="code",
            verbose=True
        )
        
        # Run analysis pipeline
        analyzer.prepare_data()
        summary = analyzer.calculate_effectiveness(plot=True)
        
        # Save results
        summary.to_csv(self.backtest_dir / "backtest_summary.csv", index=False)
        return summary

    def run_pipeline(self) -> pd.DataFrame:
        """
        Execute the complete analysis pipeline.
        
        Returns
        -------
        pd.DataFrame
            Backtest results summary
        """
        self.load_market_data()
        self.calculate_features()
        self.neutralize_factors()
        return self.run_backtest()


if __name__ == "__main__":
    # Example usage
    pipeline = QuantPipeline(
        data_dir="/path/to/data",
        output_dir="/path/to/output",
        start_date="2023-01-01",
        end_date="2023-12-31",
        universe=["AAPL", "MSFT", "GOOG", "AMZN"],  # Example universe
        barra_factors=["size", "beta", "momentum", "volatility"]
    )
    
    results = pipeline.run_pipeline()
    print("\nBacktest Results Summary:")
    print(results)