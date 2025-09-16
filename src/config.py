# Pipeline configuration

# Data paths
DATA_DIR = "/Users/fionaliu/Desktop/git/data"
OUTPUT_DIR = "/Users/fionaliu/Desktop/git/output"

# Date range
START_DATE = "20240101"
END_DATE = "20240331"

# Universe settings
USE_DEFAULT_UNIVERSE = True  # If True, uses universe from const.py
CUSTOM_UNIVERSE = [
    "AAPL", "MSFT", "GOOG", "AMZN"
]  # Used only if USE_DEFAULT_UNIVERSE = False

# Barra risk factors for neutralization
BARRA_FACTORS = [
    "size",        # Market capitalization
    "beta",        # Market sensitivity
    "momentum",    # Price momentum
    "volatility",  # Return volatility
    "value",       # Book-to-price ratio
    "growth",      # Historical growth
    "liquidity",   # Trading activity
]

# Backtest settings
BACKTEST_CONFIG = {
    "n_bins": 10,              # Number of quantile groups
    "target_col": "return_1d", # Target return column
    "code_col": "code",        # Stock identifier column
}