"""
Script to run the quantitative analysis pipeline with configuration.
"""

import sys
import logging
from pathlib import Path

from config import (
    DATA_DIR, OUTPUT_DIR, START_DATE, END_DATE,
    USE_DEFAULT_UNIVERSE, CUSTOM_UNIVERSE,
    BARRA_FACTORS, BACKTEST_CONFIG
)
from run_pipeline import QuantPipeline


def setup_logging():
    """Configure logging for the pipeline."""
    log_dir = Path(OUTPUT_DIR) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "pipeline.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )


def main():
    """Run the quantitative analysis pipeline."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Starting quantitative analysis pipeline")
        
        # Initialize pipeline
        pipeline = QuantPipeline(
            data_dir=DATA_DIR,
            output_dir=OUTPUT_DIR,
            start_date=START_DATE,
            end_date=END_DATE,
            universe=None if USE_DEFAULT_UNIVERSE else CUSTOM_UNIVERSE,
            barra_factors=BARRA_FACTORS
        )
        
        # Run pipeline and get results
        results = pipeline.run_pipeline()
        
        # Log summary statistics
        logger.info("\nBacktest Results Summary:")
        logger.info("\nTop 5 factors by IC:")
        logger.info(results.nlargest(5, "IC_mean")[
            ["factor", "IC_mean", "ICIR", "hedge_sharpe"]
        ])
        
        logger.info("\nPipeline completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())