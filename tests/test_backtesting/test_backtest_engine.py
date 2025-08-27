"""
Tests for BacktestEngine
"""

import pytest
from datetime import datetime, timedelta

from backtesting.backtest_engine import (
    BacktestEngine, Trade, Position, Portfolio,
    OrderSide, PositionSide
)


class TestBacktestEngine:
    """Test suite for BacktestEngine"""
    
    @pytest.fixture
    def engine(self):
        """Create BacktestEngine instance"""
        initial_capital = {'KRW': 20000000, 'USD': 15000}
        return BacktestEngine(initial_capital, fee_rate=0.001)
    
    def test_initialization(self, engine):
        """Test engine initialization"""
        assert engine.cash['KRW'] == 20000000
        assert engine.cash['USD'] == 15000
        assert engine.fee_rate == 0.001
        assert len(engine.positions) == 0
        assert len(engine.trades) == 0
        assert engine.total_trades == 0
    
    def test_update_time(self, engine):
        """Test time and price update"""
        timestamp = datetime.now()
        prices = {
            'upbit_BTC': 100000000,
            'binance_BTC': 70000
        }
        
        engine.update_time(timestamp, prices)
        
        assert engine.current_time == timestamp
        assert engine.current_prices == prices
    
    def test_open_position_long(self, engine):
        """Test opening a long position"""
        engine.current_time = datetime.now()
        engine.current_prices = {
            'upbit_BTC': 100000000,
            'binance_BTC': 70000
        }
        
        # Open Upbit long position
        success = engine.open_position(
            exchange='upbit',
            symbol='BTC',
            side=PositionSide.LONG,
            amount=0.1,
            price=100000000
        )
        
        assert success == True
        assert 'upbit_BTC' in engine.positions
        
        position = engine.positions['upbit_BTC']
        assert position.side == PositionSide.LONG
        assert position.amount == 0.1
        assert position.entry_price == 100000000
        
        # Check capital deduction
        expected_cost = 100000000 * 0.1 * (1 + 0.001)
        assert abs(engine.cash['KRW'] - (20000000 - expected_cost)) < 1
    
    def test_open_position_short(self, engine):
        """Test opening a short position"""
        engine.current_time = datetime.now()
        
        # Open Binance short position
        success = engine.open_position(
            exchange='binance',
            symbol='BTC',
            side=PositionSide.SHORT,
            amount=0.1,
            price=70000
        )
        
        assert success == True
        assert 'binance_BTC' in engine.positions
        
        position = engine.positions['binance_BTC']
        assert position.side == PositionSide.SHORT
        assert position.amount == 0.1
        assert position.entry_price == 70000
    
    def test_open_position_insufficient_capital(self, engine):
        """Test opening position with insufficient capital"""
        engine.current_time = datetime.now()
        
        # Try to open position larger than capital
        success = engine.open_position(
            exchange='upbit',
            symbol='BTC',
            side=PositionSide.LONG,
            amount=10,  # Too large
            price=100000000
        )
        
        assert success == False
        assert len(engine.positions) == 0
    
    def test_close_position(self, engine):
        """Test closing a position"""
        engine.current_time = datetime.now()
        
        # First open a position
        engine.open_position('upbit', 'BTC', PositionSide.LONG, 0.1, 100000000)
        
        # Update price
        engine.current_prices = {'upbit_BTC': 105000000}
        
        # Close position with profit
        pnl = engine.close_position('upbit', 'BTC', price=105000000)
        
        # Check PnL calculation
        expected_pnl = (105000000 - 100000000) * 0.1 - (105000000 * 0.1 * 0.001)
        assert abs(pnl - expected_pnl) < 1  # Allow small rounding error
        
        # Position should be removed
        assert 'upbit_BTC' not in engine.positions
    
    def test_close_nonexistent_position(self, engine):
        """Test closing a position that doesn't exist"""
        engine.current_time = datetime.now()
        
        pnl = engine.close_position('upbit', 'BTC')
        
        assert pnl == 0
        assert len(engine.positions) == 0
    
    def test_portfolio_value_calculation(self, engine):
        """Test portfolio value calculation"""
        engine.current_time = datetime.now()
        
        # Initial value
        initial_value = engine.get_portfolio_value()
        assert initial_value == 20000000 + 15000 * 1350
        
        # Open positions
        engine.open_position('upbit', 'BTC', PositionSide.LONG, 0.1, 100000000)
        engine.open_position('binance', 'BTC', PositionSide.SHORT, 0.1, 70000)
        
        # Update prices
        engine.current_prices = {
            'upbit_BTC': 105000000,
            'binance_BTC': 69000
        }
        
        # Update position prices
        engine.positions['upbit_BTC'].current_price = 105000000
        engine.positions['binance_BTC'].current_price = 69000
        
        # Calculate new value
        portfolio_value = engine.get_portfolio_value()
        
        # Should account for position values and remaining cash
        assert portfolio_value > 0
    
    def test_performance_metrics(self, engine):
        """Test performance metrics calculation"""
        engine.current_time = datetime.now()
        
        # Simulate some trades
        engine.open_position('upbit', 'BTC', PositionSide.LONG, 0.01, 100000000)
        engine.record_portfolio()
        
        engine.current_time += timedelta(hours=1)
        engine.close_position('upbit', 'BTC', price=101000000)
        engine.record_portfolio()
        
        # Get metrics
        metrics = engine.get_performance_metrics()
        
        assert 'total_return' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics
        assert 'win_rate' in metrics
        assert 'total_trades' in metrics
        assert metrics['total_trades'] == 2  # Open and close
    
    def test_trade_recording(self, engine):
        """Test trade recording"""
        engine.current_time = datetime.now()
        
        # Open position records a trade
        engine.open_position('upbit', 'BTC', PositionSide.LONG, 0.1, 100000000)
        
        assert len(engine.trades) == 1
        trade = engine.trades[0]
        
        assert isinstance(trade, Trade)
        assert trade.exchange == 'upbit'
        assert trade.symbol == 'BTC'
        assert trade.side == OrderSide.BUY
        assert trade.amount == 0.1
        assert trade.price == 100000000
    
    def test_position_pnl_calculation(self):
        """Test Position PnL calculation"""
        position = Position(
            exchange='upbit',
            symbol='BTC',
            side=PositionSide.LONG,
            amount=0.1,
            entry_price=100000000,
            current_price=105000000,
            opened_at=datetime.now(),
            trades=[]
        )
        
        assert position.pnl == (105000000 - 100000000) * 0.1
        assert position.pnl_pct == 5.0  # 5% profit
        
        # Test short position
        short_position = Position(
            exchange='binance',
            symbol='BTC',
            side=PositionSide.SHORT,
            amount=0.1,
            entry_price=70000,
            current_price=69000,
            opened_at=datetime.now(),
            trades=[]
        )
        
        assert short_position.pnl == (70000 - 69000) * 0.1  # Profit on short