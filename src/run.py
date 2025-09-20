"""
Script to run the quantitative analysis pipeline with configuration.
"""

import sys
import logging
from pathlib import Path

from src.config_loader import load_config
from run_pipeline import QuantPipeline


def setup_logging(output_dir: Path, logging_cfg: dict | None = None):
    """Configure logging for the pipeline using config values."""
    log_cfg = logging_cfg or {}
    level = getattr(logging, str(log_cfg.get("level", "INFO")).upper(), logging.INFO)
    log_to_file = bool(log_cfg.get("log_to_file", True))
    log_dir_name = log_cfg.get("log_dir", "logs")

    log_dir = output_dir / log_dir_name
    log_dir.mkdir(parents=True, exist_ok=True)

    handlers = [logging.StreamHandler(sys.stdout)]
    if log_to_file:
        handlers.insert(0, logging.FileHandler(log_dir / "pipeline.log"))

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers,
    )


def main():
    """Run the quantitative analysis pipeline."""
    cfg = load_config()
    setup_logging(Path(cfg.output_dir), cfg.logging)
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Starting quantitative analysis pipeline")
        
        # Initialize pipeline from config
        pipeline = QuantPipeline.from_config(cfg)
        
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