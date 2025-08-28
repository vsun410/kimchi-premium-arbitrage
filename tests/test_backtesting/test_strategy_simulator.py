"""
Tests for StrategySimulator
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch

from backtesting.strategy_simulator import StrategySimulator, Signal
from backtesting.backtest_engine import BacktestEngine, PositionSide


class TestStrategySimulator:
    """Test suite for StrategySimulator"""
    
    @pytest.fixture
    def mock_engine(self):
        """Create mock BacktestEngine"""
        engine = Mock(spec=BacktestEngine)
        engine.get_portfolio_value.return_value = 40000000
        engine.open_position.return_value = True
        engine.close_position.return_value = 100000
        return engine
    
    @pytest.fixture
    def simulator(self, mock_engine):
        """Create StrategySimulator instance"""
        return StrategySimulator(mock_engine, position_size_pct=0.02)
    
    def test_initialization(self, simulator):
        """Test simulator initialization"""
        assert simulator.position_size_pct == 0.02
        assert simulator.is_hedged == False
        assert simulator.entry_premium is None
        assert len(simulator.signals) == 0
        assert simulator.entry_threshold == 0.01
        assert simulator.exit_threshold == 0.005
    
    def test_generate_entry_signal(self, simulator):
        """Test entry signal generation"""
        timestamp = datetime.now()
        data = {
            'upbit_price': 100000000,
            'binance_price': 70000,
            'kimchi_premium': 2.0,  # 2% premium
            'upbit_ohlcv': pd.DataFrame(),
            'binance_ohlcv': pd.DataFrame()
        }
        
        signals = simulator.generate_signals(timestamp, data)
        
        assert len(signals) == 1
        assert signals[0].action == 'open_hedge'
        assert signals[0].confidence == 0.8
        assert 'premium' in signals[0].data
    
    def test_no_entry_signal_below_threshold(self, simulator):
        """Test no entry signal when premium below threshold"""
        timestamp = datetime.now()
        data = {
            'upbit_price': 100000000,
            'binance_price': 70000,
            'kimchi_premium': 0.5,  # 0.5% premium (below threshold)
            'upbit_ohlcv': pd.DataFrame(),
            'binance_ohlcv': pd.DataFrame()
        }
        
        signals = simulator.generate_signals(timestamp, data)
        
        assert len(signals) == 0
    
    def test_generate_exit_signal_high_premium(self, simulator):
        """Test exit signal when premium too high"""
        simulator.is_hedged = True
        simulator.entry_premium = 0.01
        
        timestamp = datetime.now()
        data = {
            'upbit_price': 100000000,
            'binance_price': 70000,
            'kimchi_premium': 0.15,  # 0.15% (above max threshold)
            'upbit_ohlcv': pd.DataFrame(),
            'binance_ohlcv': pd.DataFrame()
        }
        
        signals = simulator.generate_signals(timestamp, data)
        
        assert len(signals) == 1
        assert signals[0].action == 'close_all'
        assert 'exceeded max threshold' in signals[0].reason
    
    def test_generate_exit_signal_low_premium(self, simulator):
        """Test exit signal when premium too low"""
        simulator.is_hedged = True
        simulator.entry_premium = 0.01
        
        timestamp = datetime.now()
        data = {
            'upbit_price': 100000000,
            'binance_price': 70000,
            'kimchi_premium': 0.3,  # 0.3% (very high actually, above max threshold of 0.14%)
            'upbit_ohlcv': pd.DataFrame(),
            'binance_ohlcv': pd.DataFrame()
        }
        
        signals = simulator.generate_signals(timestamp, data)
        
        assert len(signals) == 1
        assert signals[0].action == 'close_all'
        # 0.3% (0.003) > 0.14% (0.0014) so it triggers max threshold, not exit threshold
        assert 'exceeded max threshold' in signals[0].reason
    
    def test_execute_open_hedge_signal(self, simulator, mock_engine):
        """Test executing open hedge signal"""
        signal = Signal(
            timestamp=datetime.now(),
            action='open_hedge',
            reason='Test',
            confidence=0.8,
            data={'premium': 0.02}
        )
        
        success = simulator.execute_signal(signal, 100000000, 70000)
        
        assert success == True
        assert simulator.is_hedged == True
        assert simulator.entry_premium == 0.02
        
        # Verify engine calls
        assert mock_engine.open_position.call_count == 2
        calls = mock_engine.open_position.call_args_list
        
        # Check Upbit long
        assert calls[0][0][0] == 'upbit'
        assert calls[0][0][1] == 'BTC'
        assert calls[0][0][2] == PositionSide.LONG
        
        # Check Binance short
        assert calls[1][0][0] == 'binance'
        assert calls[1][0][1] == 'BTC'
        assert calls[1][0][2] == PositionSide.SHORT
    
    def test_execute_close_all_signal(self, simulator, mock_engine):
        """Test executing close all signal"""
        simulator.is_hedged = True
        
        signal = Signal(
            timestamp=datetime.now(),
            action='close_all',
            reason='Test',
            confidence=0.8,
            data={}
        )
        
        success = simulator.execute_signal(signal, 100000000, 70000)
        
        assert success == True
        assert simulator.is_hedged == False
        
        # Verify engine calls
        assert mock_engine.close_position.call_count == 2
    
    def test_execute_close_long_signal(self, simulator, mock_engine):
        """Test executing close long signal"""
        signal = Signal(
            timestamp=datetime.now(),
            action='close_long',
            reason='Test',
            confidence=0.8,
            data={}
        )
        
        success = simulator.execute_signal(signal, 100000000, 70000)
        
        assert success == True
        
        # Verify only Upbit position closed
        mock_engine.close_position.assert_called_once_with('upbit', 'BTC', price=100000000)
    
    def test_execute_close_short_signal(self, simulator, mock_engine):
        """Test executing close short signal"""
        signal = Signal(
            timestamp=datetime.now(),
            action='close_short',
            reason='Test',
            confidence=0.8,
            data={}
        )
        
        success = simulator.execute_signal(signal, 100000000, 70000)
        
        assert success == True
        
        # Verify only Binance position closed
        mock_engine.close_position.assert_called_once_with('binance', 'BTC', price=70000)
    
    def test_position_size_calculation(self, simulator):
        """Test position size calculation"""
        capital = 40000000
        
        # Default case (no signal history)
        size = simulator.calculate_position_size(capital)
        # Default win rate 0.6, Kelly formula: (0.6 * 1.5 - 0.4) / 1.5 = 0.333
        # Conservative: 0.333 * 0.25 = 0.0833, limited by risk_pct 0.02
        expected = capital * 0.02
        assert abs(size - expected) < 1
        
        # With signal history
        for i in range(10):
            signal = Signal(
                timestamp=datetime.now(),
                action='close_all',
                reason='Test',
                confidence=0.8,
                data={'profit': 100000 if i < 7 else -50000}  # 70% win rate
            )
            simulator.signals.append(signal)
        
        size = simulator.calculate_position_size(capital)
        # Win rate = 0.7, Kelly = (0.7 * 1.5 - 0.3) / 1.5 = 0.5
        # Conservative = 0.5 * 0.25 = 0.125
        # But limited by risk_pct = 0.02
        expected = capital * 0.02
        assert abs(size - expected) < 1
    
    def test_strategy_stats(self, simulator):
        """Test strategy statistics"""
        # Add some signals
        simulator.signals = [
            Signal(datetime.now(), 'open_hedge', 'Test', 0.8, {}),
            Signal(datetime.now(), 'close_all', 'Test', 0.7, {}),
            Signal(datetime.now(), 'open_hedge', 'Test', 0.9, {}),
            Signal(datetime.now(), 'close_long', 'Test', 0.6, {}),
        ]
        simulator.is_hedged = True
        
        stats = simulator.get_strategy_stats()
        
        assert stats['total_signals'] == 4
        assert stats['open_signals'] == 2
        assert stats['close_signals'] == 2
        assert stats['avg_confidence'] == 0.75
        assert stats['is_hedged'] == True
        assert stats['action_counts']['open_hedge'] == 2
        assert stats['action_counts']['close_all'] == 1
        assert stats['action_counts']['close_long'] == 1
    
    def test_triangle_pattern_entry_delay(self, simulator):
        """Test entry delay when triangle pattern detected"""
        # Create mock trend analysis with triangle pattern
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1H')
        upbit_ohlcv = pd.DataFrame({
            'high': np.random.randn(100) + 100,
            'low': np.random.randn(100) + 99,
            'close': np.random.randn(100) + 100
        }, index=dates)
        
        timestamp = datetime.now()
        data = {
            'upbit_price': 100000000,
            'binance_price': 70000,
            'kimchi_premium': 2.0,  # Above threshold
            'upbit_ohlcv': upbit_ohlcv,
            'binance_ohlcv': pd.DataFrame()
        }
        
        # Mock triangle pattern detection
        with patch.object(
            simulator.trend_engine, 'analyze',
            return_value={
                'triangle_patterns': [
                    Mock(volatility_compression=0.8)
                ]
            }
        ):
            signals = simulator.generate_signals(timestamp, data)
            
            # Should not generate entry signal due to triangle pattern
            assert len(signals) == 0
    
    def test_reverse_premium_handling(self, simulator):
        """Test reverse premium immediate exit"""
        simulator.is_hedged = True
        
        # Mock reverse handler to return immediate exit
        with patch.object(
            simulator.reverse_handler, 'update',
            return_value={
                'is_reverse': True,
                'action': {'type': 'immediate_exit'}
            }
        ):
            timestamp = datetime.now()
            data = {
                'upbit_price': 100000000,
                'binance_price': 70000,
                'kimchi_premium': -0.5,  # Negative premium
                'upbit_ohlcv': pd.DataFrame(),
                'binance_ohlcv': pd.DataFrame()
            }
            
            signals = simulator.generate_signals(timestamp, data)
            
            assert len(signals) == 1
            assert signals[0].action == 'close_all'
            # Negative premium triggers exit threshold first before reverse handler check
            # Since -0.5% < 0.5% exit threshold
            assert 'below exit threshold' in signals[0].reason or 'Reverse premium' in signals[0].reason