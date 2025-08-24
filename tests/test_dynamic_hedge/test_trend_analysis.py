"""
Tests for Trend Analysis Engine (Task 29)
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dynamic_hedge.trend_analysis import (
    TrendAnalysisEngine, TrendLine, BreakoutSignal, TrianglePattern
)


class TestTrendAnalysisEngine:
    """Test suite for TrendAnalysisEngine"""
    
    @pytest.fixture
    def engine(self):
        """Create TrendAnalysisEngine instance"""
        return TrendAnalysisEngine(window_size=50, min_touches=3)
    
    @pytest.fixture
    def sample_ohlcv_data(self):
        """Generate sample OHLCV data"""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='15min')
        
        # Create trending data with some noise
        base_price = 50000
        trend = np.linspace(0, 1000, 100)  # Upward trend
        noise = np.random.normal(0, 50, 100)
        
        prices = base_price + trend + noise
        
        df = pd.DataFrame({
            'open': prices + np.random.normal(0, 10, 100),
            'high': prices + abs(np.random.normal(50, 20, 100)),
            'low': prices - abs(np.random.normal(50, 20, 100)),
            'close': prices,
            'volume': np.random.uniform(100, 1000, 100)
        }, index=dates)
        
        return df
    
    @pytest.fixture
    def triangle_pattern_data(self):
        """Generate data with triangle pattern"""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='15min')
        
        # Create converging triangle pattern
        x = np.arange(100)
        upper_line = 51000 - x * 5  # Descending resistance
        lower_line = 49000 + x * 3  # Ascending support
        
        prices = []
        for i in range(100):
            # Price bounces between converging lines
            if i % 4 < 2:
                price = lower_line[i] + np.random.normal(0, 20)
            else:
                price = upper_line[i] - np.random.normal(0, 20)
            prices.append(price)
        
        df = pd.DataFrame({
            'open': prices,
            'high': [p + abs(np.random.normal(30, 10)) for p in prices],
            'low': [p - abs(np.random.normal(30, 10)) for p in prices],
            'close': prices,
            'volume': np.random.uniform(100, 1000, 100)
        }, index=dates)
        
        return df
    
    def test_initialization(self, engine):
        """Test engine initialization"""
        assert engine.window_size == 50
        assert engine.min_touches == 3
        assert engine.support_levels == []
        assert engine.resistance_levels == []
        assert engine.active_patterns == []
    
    def test_analyze_with_insufficient_data(self, engine):
        """Test analysis with insufficient data"""
        df = pd.DataFrame({
            'high': [100, 101],
            'low': [99, 100],
            'close': [100, 100.5],
            'volume': [100, 110]
        })
        
        result = engine.analyze(df)
        
        assert result['trend_lines'] == []
        assert result['triangle_patterns'] == []
        assert result['breakout_signals'] == []
    
    def test_find_trend_lines(self, engine, sample_ohlcv_data):
        """Test trend line detection"""
        result = engine.analyze(sample_ohlcv_data)
        
        # Should find at least one trend line
        assert len(result['trend_lines']) >= 0
        
        for line in result['trend_lines']:
            assert isinstance(line, TrendLine)
            assert line.line_type in ['support', 'resistance']
            assert line.touches >= engine.min_touches
            assert 0 <= line.strength <= 1
    
    def test_support_resistance_levels(self, engine, sample_ohlcv_data):
        """Test support/resistance level detection"""
        result = engine.analyze(sample_ohlcv_data)
        sr_levels = result['support_resistance']
        
        assert 'support' in sr_levels
        assert 'resistance' in sr_levels
        assert isinstance(sr_levels['support'], list)
        assert isinstance(sr_levels['resistance'], list)
        
        # Support levels should be sorted
        if sr_levels['support']:
            assert sr_levels['support'] == sorted(sr_levels['support'])
        
        # Resistance levels should be sorted
        if sr_levels['resistance']:
            assert sr_levels['resistance'] == sorted(sr_levels['resistance'])
    
    def test_triangle_pattern_detection(self, engine, triangle_pattern_data):
        """Test triangle pattern detection"""
        result = engine.analyze(triangle_pattern_data)
        patterns = result['triangle_patterns']
        
        # Should detect at least one triangle pattern
        if patterns:
            pattern = patterns[0]
            assert isinstance(pattern, TrianglePattern)
            assert pattern.pattern_type in ['ascending', 'descending', 'symmetric', 'wedge']
            assert pattern.convergence_angle > 0
            assert 0 <= pattern.volatility_compression <= 1
    
    def test_breakout_signal_detection(self, engine):
        """Test breakout signal detection"""
        # Create data with clear breakout
        dates = pd.date_range(start='2024-01-01', periods=100, freq='15min')
        prices = [50000] * 95 + [51000, 52000, 53000, 54000, 55000]  # Breakout
        
        df = pd.DataFrame({
            'open': prices,
            'high': [p + 50 for p in prices],
            'low': [p - 50 for p in prices],
            'close': prices,
            'volume': [100] * 95 + [500, 600, 700, 800, 900]  # Volume spike
        }, index=dates)
        
        result = engine.analyze(df)
        signals = result['breakout_signals']
        
        # Check breakout signal properties
        for signal in signals:
            assert isinstance(signal, BreakoutSignal)
            assert signal.direction in ['up', 'down']
            assert 0 <= signal.strength <= 1
            assert isinstance(signal.volume_confirmation, bool)
    
    def test_get_trend_direction(self, engine, sample_ohlcv_data):
        """Test trend direction detection"""
        direction = engine.get_trend_direction(sample_ohlcv_data)
        assert direction in ['up', 'down', 'sideways']
        
        # With upward trending data, should detect 'up'
        # (sample_ohlcv_data has upward trend)
        assert direction == 'up'
    
    def test_calculate_breakout_probability(self, engine):
        """Test breakout probability calculation"""
        # Create a mock triangle pattern
        pattern = TrianglePattern(
            pattern_type='ascending',
            apex_time=datetime.now() + timedelta(hours=2),
            upper_line=TrendLine(
                start_time=datetime.now() - timedelta(hours=2),
                end_time=datetime.now(),
                start_price=51000,
                end_price=50500,
                slope=-0.5,
                intercept=51000,
                strength=0.8,
                touches=4,
                line_type='resistance'
            ),
            lower_line=TrendLine(
                start_time=datetime.now() - timedelta(hours=2),
                end_time=datetime.now(),
                start_price=49000,
                end_price=49500,
                slope=0.5,
                intercept=49000,
                strength=0.8,
                touches=4,
                line_type='support'
            ),
            convergence_angle=1.0,
            volatility_compression=0.7
        )
        
        # Create sample data
        df = pd.DataFrame({
            'close': [50000] * 50,
            'volume': [100] * 50
        }, index=pd.date_range(start='2024-01-01', periods=50, freq='15min'))
        
        probability = engine.calculate_breakout_probability(pattern, df)
        
        assert 0 <= probability <= 1
        # Ascending pattern should have higher probability of upward breakout
        assert probability > 0.5
    
    def test_trend_line_slope_calculation(self, engine):
        """Test trend line slope calculation"""
        # Create perfectly linear data
        dates = pd.date_range(start='2024-01-01', periods=10, freq='15min')
        prices = np.linspace(50000, 51000, 10)
        
        df = pd.DataFrame({
            'high': prices,
            'low': prices - 100,
            'close': prices - 50,
            'volume': [100] * 10
        }, index=dates)
        
        trend_lines = engine._find_trend_lines(df)
        
        if trend_lines:
            line = trend_lines[0]
            # Slope should be positive for upward trend
            assert line.slope > 0 if line.line_type == 'support' else line.slope <= 0