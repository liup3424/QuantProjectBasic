"""
YAML configuration loader for the QuantProjectBasic pipeline.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import yaml


@dataclass
class PipelineConfig:
    data_dir: Path
    output_dir: Path
    start_date: str
    end_date: str
    use_default_universe: bool
    custom_universe: List[str]
    barra_factors: List[str]
    backtest: Dict[str, Any]
    logging: Dict[str, Any]


def load_config(path: str | Path = None) -> PipelineConfig:
    """Load and validate YAML config file.

    Parameters
    ----------
    path : str | Path
        Path to YAML config. Defaults to `src/config.yaml` next to this loader.
    """
    if path is None:
        path = Path(__file__).parent / "config.yaml"
    else:
        path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    # Basic validation and extraction
    paths = cfg.get("paths", {})
    dates = cfg.get("dates", {})
    universe = cfg.get("universe", {})
    barra = cfg.get("barra", {})
    backtest = cfg.get("backtest", {})
    logging_cfg = cfg.get("logging", {})

    data_dir = Path(paths.get("data_dir", "./data"))
    output_dir = Path(paths.get("output_dir", "./output"))

    start_date = dates.get("start_date")
    end_date = dates.get("end_date")
    if not start_date or not end_date:
        raise ValueError("start_date and end_date must be set in the config.")

    use_default_universe = bool(universe.get("use_default_universe", True))
    custom_universe = universe.get("custom_universe") or []

    barra_factors = barra.get("factors") or []

    return PipelineConfig(
        data_dir=data_dir,
        output_dir=output_dir,
        start_date=start_date,
        end_date=end_date,
        use_default_universe=use_default_universe,
        custom_universe=custom_universe,
        barra_factors=barra_factors,
        backtest=backtest,
        logging=logging_cfg,
    )
