"""Tests for shadow feature calculations."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from src.features.calculate_factor import compute_shadow_features


def create_test_data():
    """Create sample market data for testing."""
    dates = pd.date_range(start='2023-01-01', periods=30, freq='B')
    codes = ['AAPL', 'MSFT']
    
    data = []
    for code in codes:
        for date in dates:
            close = 100 + np.random.randn()
            data.append({
                'code': code,
                'date': date,
                'close': close,
                'open': close + np.random.randn(),
                'high': close + abs(np.random.randn()),
                'low': close - abs(np.random.randn()),
                'volume': 1000 + np.random.randint(0, 1000),
                'money': 100000 + np.random.randint(0, 10000)
            })
    
    return pd.DataFrame(data)


def test_required_columns():
    """Test that missing columns raise appropriate error."""
    df = pd.DataFrame({'code': ['AAPL'], 'date': ['2023-01-01']})
    
    with pytest.raises(ValueError) as exc_info:
        compute_shadow_features(df)
    assert "Missing required columns" in str(exc_info.value)


def test_basic_shadow_calculations():
    """Test basic shadow length calculations."""
    df = create_test_data()
    result = compute_shadow_features(df)
    
    # Test upshadow calculation
    assert 'upshadow' in result.columns
    assert (result['upshadow'] >= 0).all()
    assert (result['upshadow'] == result['high'] - result[['close', 'open']].max(axis=1)).all()
    
    # Test downshadow calculation
    assert 'downshadow' in result.columns
    assert (result['downshadow'] >= 0).all()
    assert (result['downshadow'] == result[['close', 'open']].min(axis=1) - result['low']).all()


def test_vwap_calculation():
    """Test VWAP calculation."""
    df = create_test_data()
    result = compute_shadow_features(df)
    
    # Test VWAP
    assert 'vwap' in result.columns
    expected_vwap = df['money'] / df['volume']
    pd.testing.assert_series_equal(result['vwap'], expected_vwap, check_names=False)


def test_window_calculations():
    """Test that window-based calculations use correct periods."""
    df = create_test_data()
    short_win = 5
    long_win = 10
    result = compute_shadow_features(df, short_win=short_win, long_win=long_win)
    
    # First few rows of windowed calcs should be NaN
    for col in ['up_mean', 'down_mean', 'up_std', 'down_std']:
        assert result[col].iloc[:long_win-1].isna().all()
        assert not result[col].iloc[long_win:].isna().all()


def test_no_negative_shadows():
    """Test that shadow lengths are never negative."""
    df = create_test_data()
    result = compute_shadow_features(df)
    
    shadow_cols = ['upshadow', 'downshadow', 'wm_up', 'wm_down']
    for col in shadow_cols:
        assert (result[col] >= 0).all()


def test_handle_zero_volume():
    """Test handling of zero volume edge case."""
    df = create_test_data()
    df.loc[0, 'volume'] = 0
    result = compute_shadow_features(df)
    
    # VWAP should be NaN when volume is zero
    assert pd.isna(result.loc[0, 'vwap'])


if __name__ == '__main__':
    pytest.main([__file__])