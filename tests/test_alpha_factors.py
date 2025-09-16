"""Tests for alpha factor calculations."""

import pytest
import numpy as np
import pandas as pd
from src.features.alpha101_factors import Alpha101Calculator


def create_test_data():
    """Create sample market data for testing alpha factors."""
    dates = pd.date_range(start='2023-01-01', periods=30, freq='B')
    data = {
        'close': 100 + np.random.randn(30),
        'open': 100 + np.random.randn(30),
        'high': 102 + np.random.randn(30),
        'low': 98 + np.random.randn(30),
        'volume': 1000 + np.random.randint(0, 1000, 30),
        'vwap': 100 + np.random.randn(30),
    }
    df = pd.DataFrame(data, index=dates)
    df['returns'] = df['close'].pct_change()
    return df


def test_initialization():
    """Test Alpha101Calculator initialization."""
    df = create_test_data()
    calculator = Alpha101Calculator(df)
    assert calculator.df is not None
    assert not calculator.df.equals(df)  # Should be a copy


def test_missing_columns():
    """Test error handling for missing required columns."""
    df = pd.DataFrame({'close': [100]})
    with pytest.raises(ValueError) as exc_info:
        Alpha101Calculator(df)
    assert "Missing required columns" in str(exc_info.value)


def test_alpha001():
    """Test alpha001 factor calculation."""
    df = create_test_data()
    calculator = Alpha101Calculator(df)
    result = calculator.alpha001()
    
    assert isinstance(result, pd.Series)
    assert len(result) == len(df)
    assert not result.empty
    assert result.dtype == float


def test_alpha002():
    """Test alpha002 factor calculation."""
    df = create_test_data()
    calculator = Alpha101Calculator(df)
    result = calculator.alpha002()
    
    assert isinstance(result, pd.Series)
    assert len(result) == len(df)
    assert not result.empty
    assert result.dtype == float


def test_alpha003():
    """Test alpha003 factor calculation."""
    df = create_test_data()
    calculator = Alpha101Calculator(df)
    result = calculator.alpha003()
    
    assert isinstance(result, pd.Series)
    assert len(result) == len(df)
    assert not result.empty
    assert result.dtype == float


def test_calculate_all():
    """Test calculation of all alpha factors."""
    df = create_test_data()
    calculator = Alpha101Calculator(df)
    results = calculator.calculate_all()
    
    assert isinstance(results, dict)
    assert len(results) > 0
    assert all(isinstance(v, pd.Series) for v in results.values())
    assert all(len(v) == len(df) for v in results.values())


def test_rank_function():
    """Test the internal rank function."""
    df = create_test_data()
    calculator = Alpha101Calculator(df)
    result = calculator._rank(df['close'])
    
    assert isinstance(result, pd.Series)
    assert len(result) == len(df)
    assert result.min() >= 0
    assert result.max() <= 1


if __name__ == '__main__':
    pytest.main([__file__])