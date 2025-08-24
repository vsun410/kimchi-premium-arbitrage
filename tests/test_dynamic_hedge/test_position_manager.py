"""
Tests for Dynamic Position Manager (Task 30)
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from dynamic_hedge.position_manager import (
    DynamicPositionManager, Position, PositionType, PositionStatus,
    HedgeState, ExitCondition
)


class TestDynamicPositionManager:
    """Test suite for DynamicPositionManager"""
    
    @pytest.fixture
    def manager(self):
        """Create DynamicPositionManager instance"""
        return DynamicPositionManager(capital_per_exchange=20000000)
    
    @pytest.fixture
    def market_data_normal(self):
        """Normal market data"""
        return {
            'kimchi_premium': 0.008,  # 0.8%
            'breakout_signal': None,
            'breakout_strength': 0,
            'current_pnl': 100000,
            'upbit': 100000000,  # 1억원
            'binance': 65000  # $65,000
        }
    
    @pytest.fixture
    def market_data_high_premium(self):
        """High premium market data"""
        return {
            'kimchi_premium': 0.0015,  # 0.15% (above 0.14% threshold)
            'breakout_signal': None,
            'breakout_strength': 0,
            'current_pnl': 200000,
            'upbit': 101500000,
            'binance': 65000
        }
    
    @pytest.fixture
    def market_data_breakout_up(self):
        """Upward breakout market data"""
        return {
            'kimchi_premium': 0.005,
            'breakout_signal': 'up',
            'breakout_strength': 0.025,  # 2.5%
            'current_pnl': 50000,
            'upbit': 102000000,
            'binance': 66000
        }
    
    def test_initialization(self, manager):
        """Test manager initialization"""
        assert manager.capital_per_exchange == 20000000
        assert manager.hedge_state.total_capital == 40000000
        assert manager.hedge_state.used_capital == 0
        assert manager.hedge_state.is_hedged == False
        assert len(manager.exit_conditions) == 4  # Default conditions
    
    def test_exit_conditions_setup(self, manager):
        """Test default exit conditions"""
        conditions = manager.exit_conditions
        
        # Check condition types
        condition_types = [c.condition_type for c in conditions]
        assert 'premium' in condition_types
        assert 'breakout_up' in condition_types
        assert 'breakout_down' in condition_types
        assert 'stop_loss' in condition_types
        
        # Check priorities
        stop_loss = next(c for c in conditions if c.condition_type == 'stop_loss')
        assert stop_loss.priority == 0  # Highest priority
    
    @pytest.mark.asyncio
    async def test_open_hedge_position(self, manager):
        """Test opening hedge position"""
        upbit_price = 100000000  # 1억원
        binance_price = 65000  # $65,000
        position_size = 0.1  # 0.1 BTC
        
        success = await manager.open_hedge_position(
            upbit_price, binance_price, position_size
        )
        
        assert success == True
        assert manager.hedge_state.is_hedged == True
        assert manager.hedge_state.upbit_position is not None
        assert manager.hedge_state.binance_position is not None
        
        # Check positions
        upbit_pos = manager.hedge_state.upbit_position
        assert upbit_pos.position_type == PositionType.LONG
        assert upbit_pos.size == position_size
        assert upbit_pos.entry_price == upbit_price
        
        binance_pos = manager.hedge_state.binance_position
        assert binance_pos.position_type == PositionType.SHORT
        assert binance_pos.size == position_size
        assert binance_pos.entry_price == binance_price
    
    @pytest.mark.asyncio
    async def test_open_position_insufficient_capital(self, manager):
        """Test opening position with insufficient capital"""
        upbit_price = 1000000000  # 10억원
        binance_price = 650000  # $650,000
        position_size = 100  # 100 BTC (too large)
        
        success = await manager.open_hedge_position(
            upbit_price, binance_price, position_size
        )
        
        assert success == False
        assert manager.hedge_state.is_hedged == False
    
    @pytest.mark.asyncio
    async def test_check_exit_conditions_high_premium(self, manager, market_data_high_premium):
        """Test exit condition check with high premium"""
        # Open position first
        await manager.open_hedge_position(100000000, 65000, 0.1)
        
        # Check conditions
        triggered = manager.check_exit_conditions(market_data_high_premium)
        
        assert triggered is not None
        assert triggered.condition_type == 'premium'
        assert triggered.action == 'close_all'
    
    @pytest.mark.asyncio
    async def test_check_exit_conditions_breakout(self, manager, market_data_breakout_up):
        """Test exit condition check with breakout"""
        # Open position first
        await manager.open_hedge_position(100000000, 65000, 0.1)
        
        # Check conditions
        triggered = manager.check_exit_conditions(market_data_breakout_up)
        
        assert triggered is not None
        assert triggered.condition_type == 'breakout_up'
        assert triggered.action == 'close_short'
    
    @pytest.mark.asyncio
    async def test_execute_exit_close_all(self, manager):
        """Test executing exit with close all"""
        # Open position
        await manager.open_hedge_position(100000000, 65000, 0.1)
        
        # Create exit condition
        condition = ExitCondition(
            condition_type='premium',
            threshold=0.0014,
            action='close_all',
            priority=1
        )
        
        # Execute exit
        current_prices = {'upbit': 101000000, 'binance': 64000}
        result = await manager.execute_exit(condition, current_prices)
        
        assert result['success'] == True
        assert 'upbit_long' in result['closed_positions']
        assert 'binance_short' in result['closed_positions']
        assert manager.hedge_state.is_hedged == False
    
    @pytest.mark.asyncio
    async def test_execute_exit_close_long_only(self, manager):
        """Test executing exit closing long position only"""
        # Open position
        await manager.open_hedge_position(100000000, 65000, 0.1)
        
        # Create exit condition
        condition = ExitCondition(
            condition_type='breakout_down',
            threshold=-0.02,
            action='close_long',
            priority=2
        )
        
        # Execute exit
        current_prices = {'upbit': 98000000, 'binance': 64000}
        result = await manager.execute_exit(condition, current_prices)
        
        assert result['success'] == True
        assert 'upbit_long' in result['closed_positions']
        assert 'binance_short' not in result['closed_positions']
        assert manager.hedge_state.upbit_position is None
        assert manager.hedge_state.binance_position is not None
        assert manager.hedge_state.is_hedged == False  # Not fully hedged
    
    def test_calculate_reentry_timing_cooldown(self, manager):
        """Test reentry timing during cooldown"""
        # Set last exit time
        manager.last_exit_time = datetime.now() - timedelta(minutes=10)
        
        market_conditions = {
            'volatility': 0.02,
            'trend': 'sideways',
            'premium_trend': 'stable',
            'volume': 100
        }
        
        result = manager.calculate_reentry_timing(market_conditions)
        
        assert result['can_reenter'] == False
        assert result['recommended_wait'] > 0  # Still in cooldown
    
    def test_calculate_reentry_timing_optimal(self, manager):
        """Test reentry timing with optimal conditions"""
        # No recent exit
        manager.last_exit_time = None
        
        market_conditions = {
            'volatility': 0.02,  # Within optimal range
            'trend': 'sideways',
            'premium_trend': 'stable',
            'volume': 100
        }
        
        result = manager.calculate_reentry_timing(market_conditions)
        
        assert result['can_reenter'] == True
        assert result['recommended_wait'] == 0
        assert 'optimal_conditions' in result
    
    def test_calculate_reentry_timing_high_volatility(self, manager):
        """Test reentry timing with high volatility"""
        manager.last_exit_time = None
        
        market_conditions = {
            'volatility': 0.05,  # High volatility
            'trend': 'up',
            'premium_trend': 'increasing',
            'volume': 200
        }
        
        result = manager.calculate_reentry_timing(market_conditions)
        
        assert result['can_reenter'] == False
        assert result['recommended_wait'] > 0
    
    @pytest.mark.asyncio
    async def test_get_position_status(self, manager):
        """Test getting position status"""
        # Before opening position
        status = manager.get_position_status()
        assert status['is_hedged'] == False
        assert status['hedge_ratio'] == 0
        assert status['positions'] == {}
        
        # After opening position
        await manager.open_hedge_position(100000000, 65000, 0.1)
        status = manager.get_position_status()
        
        assert status['is_hedged'] == True
        assert status['hedge_ratio'] == 1.0  # Perfect hedge
        assert 'upbit' in status['positions']
        assert 'binance' in status['positions']
        assert status['positions']['upbit']['type'] == 'long'
        assert status['positions']['binance']['type'] == 'short'
    
    def test_calculate_optimal_position_size(self, manager):
        """Test optimal position size calculation"""
        available_capital = 10000000  # 1000만원
        risk_level = 0.02  # 2% risk
        
        position_size = manager.calculate_optimal_position_size(
            available_capital, risk_level
        )
        
        assert position_size > 0
        assert position_size < 1  # Should be reasonable size
    
    def test_position_pnl_calculation(self):
        """Test position PnL calculation"""
        # Long position
        long_pos = Position(
            exchange='upbit',
            symbol='BTC/KRW',
            position_type=PositionType.LONG,
            size=0.1,
            entry_price=100000000,
            current_price=100000000,
            status=PositionStatus.OPEN,
            opened_at=datetime.now(),
            fees=10000
        )
        
        # Price increased
        pnl = long_pos.calculate_pnl(101000000)
        assert pnl == (101000000 - 100000000) * 0.1 - 10000
        
        # Short position
        short_pos = Position(
            exchange='binance',
            symbol='BTC/USDT',
            position_type=PositionType.SHORT,
            size=0.1,
            entry_price=65000,
            current_price=65000,
            status=PositionStatus.OPEN,
            opened_at=datetime.now(),
            fees=10
        )
        
        # Price decreased (profit for short)
        pnl = short_pos.calculate_pnl(64000)
        assert pnl == (65000 - 64000) * 0.1 - 10
    
    def test_hedge_state_properties(self):
        """Test HedgeState properties"""
        state = HedgeState(total_capital=40000000)
        
        assert state.available_capital == 40000000
        assert state.hedge_ratio == 0
        
        # Add positions
        state.upbit_position = Position(
            exchange='upbit',
            symbol='BTC/KRW',
            position_type=PositionType.LONG,
            size=0.1,
            entry_price=100000000,
            current_price=100000000,
            status=PositionStatus.OPEN,
            opened_at=datetime.now()
        )
        
        state.binance_position = Position(
            exchange='binance',
            symbol='BTC/USDT',
            position_type=PositionType.SHORT,
            size=0.1,
            entry_price=65000,
            current_price=65000,
            status=PositionStatus.OPEN,
            opened_at=datetime.now()
        )
        
        state.used_capital = 10000000
        
        assert state.available_capital == 30000000
        assert state.hedge_ratio == 1.0  # Equal sizes