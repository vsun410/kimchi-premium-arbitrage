"""
멀티 전략 시스템 테스트
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import numpy as np

from strategies.multi_strategy import (
    BaseStrategy,
    MarketData,
    TradingSignal,
    SignalType,
    StrategyStatus,
    ThresholdStrategy,
    MovingAverageStrategy,
    BollingerBandsStrategy,
    StrategyManager,
    AllocationMethod,
    SignalAggregation
)


class TestBaseStrategy:
    """베이스 전략 테스트"""
    
    def test_strategy_initialization(self):
        """전략 초기화 테스트"""
        # 테스트용 구체 클래스
        class TestStrategy(BaseStrategy):
            def analyze(self, market_data):
                return None
            
            def calculate_position_size(self, signal):
                return 0.01
            
            def should_close_position(self, market_data):
                return False
        
        strategy = TestStrategy("Test", {}, 1_000_000)
        
        assert strategy.name == "Test"
        assert strategy.initial_capital == 1_000_000
        assert strategy.current_capital == 1_000_000
        assert strategy.position == 0
        assert strategy.status == StrategyStatus.ACTIVE
    
    def test_market_data(self):
        """시장 데이터 테스트"""
        data = MarketData(
            timestamp=datetime.now(),
            upbit_price=100_000_000,
            binance_price=70_000,
            exchange_rate=1400,
            kimchi_premium=2.04,
            volume_upbit=100,
            volume_binance=200,
            bid_ask_spread_upbit=0.1,
            bid_ask_spread_binance=0.05
        )
        
        data_dict = data.to_dict()
        assert 'timestamp' in data_dict
        assert data_dict['kimchi_premium'] == 2.04
    
    def test_trading_signal(self):
        """거래 신호 테스트"""
        signal = TradingSignal(
            timestamp=datetime.now(),
            strategy_name="Test",
            signal_type=SignalType.BUY,
            confidence=0.8,
            suggested_amount=0.01,
            reason="Test signal"
        )
        
        assert signal.signal_type == SignalType.BUY
        assert signal.confidence == 0.8
        assert "BUY" in str(signal)


class TestThresholdStrategy:
    """임계값 전략 테스트"""
    
    def test_initialization(self):
        """초기화 테스트"""
        config = {
            'entry_threshold': 3.0,
            'exit_threshold': 1.5,
            'stop_loss': -2.0
        }
        strategy = ThresholdStrategy(config=config)
        
        assert strategy.config['entry_threshold'] == 3.0
        assert strategy.config['exit_threshold'] == 1.5
        assert strategy.last_exit_time is None
    
    def test_entry_signal(self):
        """진입 신호 테스트"""
        strategy = ThresholdStrategy()
        
        # 김프가 임계값 이상
        market_data = MarketData(
            timestamp=datetime.now(),
            upbit_price=100_000_000,
            binance_price=70_000,
            exchange_rate=1400,
            kimchi_premium=3.5,  # > 3.0
            volume_upbit=100,
            volume_binance=200,
            bid_ask_spread_upbit=0.1,
            bid_ask_spread_binance=0.05
        )
        
        signal = strategy.analyze(market_data)
        
        assert signal is not None
        assert signal.signal_type == SignalType.BUY
        assert signal.confidence > 0
    
    def test_no_signal_below_threshold(self):
        """임계값 미만에서 신호 없음"""
        strategy = ThresholdStrategy()
        
        market_data = MarketData(
            timestamp=datetime.now(),
            upbit_price=100_000_000,
            binance_price=70_000,
            exchange_rate=1400,
            kimchi_premium=2.0,  # < 3.0
            volume_upbit=100,
            volume_binance=200,
            bid_ask_spread_upbit=0.1,
            bid_ask_spread_binance=0.05
        )
        
        signal = strategy.analyze(market_data)
        assert signal is None
    
    def test_position_sizing(self):
        """포지션 크기 계산 테스트"""
        strategy = ThresholdStrategy()
        
        signal = TradingSignal(
            timestamp=datetime.now(),
            strategy_name="Test",
            signal_type=SignalType.BUY,
            confidence=0.8,
            suggested_amount=0,
            reason="Test",
            metadata={'upbit_price': 100_000_000}
        )
        
        size = strategy.calculate_position_size(signal)
        
        assert size > 0
        assert size <= 1  # 합리적인 크기
    
    def test_exit_condition(self):
        """청산 조건 테스트"""
        strategy = ThresholdStrategy()
        strategy.position = 0.01
        strategy.entry_price = 100_000_000
        strategy.entry_time = datetime.now() - timedelta(minutes=10)
        
        # 김프가 청산 임계값 이하
        market_data = MarketData(
            timestamp=datetime.now(),
            upbit_price=100_000_000,
            binance_price=70_000,
            exchange_rate=1400,
            kimchi_premium=1.0,  # < 1.5
            volume_upbit=100,
            volume_binance=200,
            bid_ask_spread_upbit=0.1,
            bid_ask_spread_binance=0.05
        )
        
        should_close = strategy.should_close_position(market_data)
        assert should_close is True


class TestMovingAverageStrategy:
    """이동평균 전략 테스트"""
    
    def test_initialization(self):
        """초기화 테스트"""
        config = {
            'short_window': 10,
            'long_window': 30,
            'ma_type': 'ema'
        }
        strategy = MovingAverageStrategy(config=config)
        
        assert strategy.config['short_window'] == 10
        assert strategy.config['long_window'] == 30
        assert strategy.config['ma_type'] == 'ema'
    
    def test_ma_calculation(self):
        """이동평균 계산 테스트"""
        strategy = MovingAverageStrategy()
        
        # 테스트 데이터
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        
        # SMA
        sma = strategy._calculate_ma(data, 5, 'sma')
        assert sma == 8.0  # (6+7+8+9+10)/5
        
        # EMA
        ema = strategy._calculate_ma(data, 5, 'ema')
        assert ema > 0
        
        # WMA
        wma = strategy._calculate_ma(data, 5, 'wma')
        assert wma > sma  # 가중평균은 단순평균보다 큼
    
    def test_crossover_signal(self):
        """MA 크로스오버 신호 테스트"""
        strategy = MovingAverageStrategy(
            config={'short_window': 3, 'long_window': 5}
        )
        
        # 상승 추세 데이터 시뮬레이션
        for i in range(10):
            premium = 2.0 + i * 0.5  # 상승 추세
            strategy._update_premium_history(premium)
        
        market_data = MarketData(
            timestamp=datetime.now(),
            upbit_price=100_000_000,
            binance_price=70_000,
            exchange_rate=1400,
            kimchi_premium=7.0,
            volume_upbit=100,
            volume_binance=200,
            bid_ask_spread_upbit=0.1,
            bid_ask_spread_binance=0.05
        )
        
        signal = strategy.analyze(market_data)
        
        # 상승 추세에서 신호 발생
        assert signal is not None
        assert signal.signal_type == SignalType.BUY


class TestBollingerBandsStrategy:
    """볼린저 밴드 전략 테스트"""
    
    def test_initialization(self):
        """초기화 테스트"""
        config = {
            'bb_period': 20,
            'bb_std': 2.0
        }
        strategy = BollingerBandsStrategy(config=config)
        
        assert strategy.config['bb_period'] == 20
        assert strategy.config['bb_std'] == 2.0
    
    def test_bollinger_bands_calculation(self):
        """볼린저 밴드 계산 테스트"""
        strategy = BollingerBandsStrategy(
            config={'bb_period': 5}
        )
        
        # 테스트 데이터 (평균 3, 표준편차 약 1.4)
        for value in [1, 2, 3, 4, 5]:
            strategy._update_premium_history(value)
        
        result = strategy._calculate_bollinger_bands()
        assert result is not None
        
        upper, middle, lower, width, pct_b = result
        
        assert middle == 3.0  # 평균
        assert upper > middle
        assert lower < middle
        assert width > 0
        assert 0 <= pct_b <= 1.5  # 대략적인 범위
    
    def test_upper_band_touch_signal(self):
        """상단 밴드 터치 신호 테스트"""
        strategy = BollingerBandsStrategy(
            config={'bb_period': 5, 'entry_bb_pct': 0.9}
        )
        
        # 안정적인 데이터 후 급등
        for value in [3, 3, 3, 3, 3]:
            strategy._update_premium_history(value)
        
        # 급등 시뮬레이션
        strategy._update_premium_history(5.0)
        
        market_data = MarketData(
            timestamp=datetime.now(),
            upbit_price=100_000_000,
            binance_price=70_000,
            exchange_rate=1400,
            kimchi_premium=5.0,
            volume_upbit=100,
            volume_binance=200,
            bid_ask_spread_upbit=0.1,
            bid_ask_spread_binance=0.05
        )
        
        signal = strategy.analyze(market_data)
        
        # 상단 밴드 근처에서 신호 발생
        assert signal is not None or strategy.bb_pct[-1] < 0.9


class TestStrategyManager:
    """전략 매니저 테스트"""
    
    def test_initialization(self):
        """초기화 테스트"""
        manager = StrategyManager(10_000_000)
        
        assert manager.initial_capital == 10_000_000
        assert manager.current_capital == 10_000_000
        assert len(manager.strategies) == 0
        assert manager.portfolio.total_capital == 10_000_000
    
    def test_add_strategy(self):
        """전략 추가 테스트"""
        manager = StrategyManager()
        
        strategy1 = ThresholdStrategy()
        strategy2 = MovingAverageStrategy()
        
        assert manager.add_strategy(strategy1) is True
        assert manager.add_strategy(strategy2) is True
        
        assert len(manager.strategies) == 2
        assert strategy1.name in manager.strategies
        assert strategy2.name in manager.strategies
    
    def test_duplicate_strategy(self):
        """중복 전략 추가 방지"""
        manager = StrategyManager()
        
        strategy = ThresholdStrategy()
        
        assert manager.add_strategy(strategy) is True
        assert manager.add_strategy(strategy) is False  # 중복
    
    def test_remove_strategy(self):
        """전략 제거 테스트"""
        manager = StrategyManager()
        
        strategy = ThresholdStrategy()
        manager.add_strategy(strategy)
        
        assert manager.remove_strategy(strategy.name) is True
        assert len(manager.strategies) == 0
    
    def test_remove_strategy_with_position(self):
        """포지션 있는 전략 제거 방지"""
        manager = StrategyManager()
        
        strategy = ThresholdStrategy()
        strategy.position = 0.01  # 포지션 있음
        manager.add_strategy(strategy)
        
        assert manager.remove_strategy(strategy.name) is False
    
    @pytest.mark.asyncio
    async def test_analyze_market(self):
        """시장 분석 테스트"""
        manager = StrategyManager()
        
        # Mock 전략 추가
        mock_strategy = Mock(spec=BaseStrategy)
        mock_strategy.name = "MockStrategy"
        mock_strategy.status = StrategyStatus.ACTIVE
        mock_strategy.update = Mock(return_value=TradingSignal(
            timestamp=datetime.now(),
            strategy_name="MockStrategy",
            signal_type=SignalType.BUY,
            confidence=0.7,
            suggested_amount=0.01,
            reason="Test"
        ))
        
        manager.strategies["MockStrategy"] = mock_strategy
        
        market_data = MarketData(
            timestamp=datetime.now(),
            upbit_price=100_000_000,
            binance_price=70_000,
            exchange_rate=1400,
            kimchi_premium=3.0,
            volume_upbit=100,
            volume_binance=200,
            bid_ask_spread_upbit=0.1,
            bid_ask_spread_binance=0.05
        )
        
        signals = await manager.analyze_market(market_data)
        
        assert len(signals) == 1
        assert signals[0].signal_type == SignalType.BUY
    
    def test_signal_aggregation_unanimous(self):
        """만장일치 신호 통합 테스트"""
        manager = StrategyManager(
            config={'signal_aggregation': SignalAggregation.UNANIMOUS}
        )
        
        # 3개 전략 추가
        for i in range(3):
            strategy = Mock(spec=BaseStrategy)
            strategy.name = f"Strategy{i}"
            manager.strategies[strategy.name] = strategy
        
        # 모든 전략이 BUY 신호
        signals = [
            TradingSignal(
                timestamp=datetime.now(),
                strategy_name=f"Strategy{i}",
                signal_type=SignalType.BUY,
                confidence=0.7,
                suggested_amount=0.01,
                reason="Test"
            )
            for i in range(3)
        ]
        
        aggregated = manager.aggregate_signals(signals)
        assert aggregated is not None
        assert aggregated.signal_type == SignalType.BUY
    
    def test_signal_aggregation_majority(self):
        """과반수 신호 통합 테스트"""
        manager = StrategyManager(
            config={'signal_aggregation': SignalAggregation.MAJORITY}
        )
        
        # 3개 전략 추가
        for i in range(3):
            strategy = Mock(spec=BaseStrategy)
            strategy.name = f"Strategy{i}"
            manager.strategies[strategy.name] = strategy
        
        # 2개 BUY, 1개 HOLD
        signals = [
            TradingSignal(
                timestamp=datetime.now(),
                strategy_name="Strategy0",
                signal_type=SignalType.BUY,
                confidence=0.7,
                suggested_amount=0.01,
                reason="Test"
            ),
            TradingSignal(
                timestamp=datetime.now(),
                strategy_name="Strategy1",
                signal_type=SignalType.BUY,
                confidence=0.6,
                suggested_amount=0.01,
                reason="Test"
            ),
            TradingSignal(
                timestamp=datetime.now(),
                strategy_name="Strategy2",
                signal_type=SignalType.HOLD,
                confidence=0.5,
                suggested_amount=0,
                reason="Test"
            )
        ]
        
        aggregated = manager.aggregate_signals(signals)
        assert aggregated is not None
        assert aggregated.signal_type == SignalType.BUY
    
    def test_signal_aggregation_best(self):
        """최고 신뢰도 신호 선택 테스트"""
        manager = StrategyManager(
            config={'signal_aggregation': SignalAggregation.BEST}
        )
        
        signals = [
            TradingSignal(
                timestamp=datetime.now(),
                strategy_name="Strategy0",
                signal_type=SignalType.BUY,
                confidence=0.5,
                suggested_amount=0.01,
                reason="Test"
            ),
            TradingSignal(
                timestamp=datetime.now(),
                strategy_name="Strategy1",
                signal_type=SignalType.SELL,
                confidence=0.9,  # 최고 신뢰도
                suggested_amount=0.01,
                reason="Test"
            )
        ]
        
        aggregated = manager.aggregate_signals(signals)
        assert aggregated is not None
        assert aggregated.signal_type == SignalType.SELL
        assert aggregated.confidence == 0.9
    
    def test_capital_allocation_equal(self):
        """균등 자본 배분 테스트"""
        manager = StrategyManager(
            initial_capital=3_000_000,
            config={'allocation_method': AllocationMethod.EQUAL}
        )
        
        # 3개 전략 추가
        strategies = [
            ThresholdStrategy(),
            MovingAverageStrategy(),
            BollingerBandsStrategy()
        ]
        
        for strategy in strategies:
            manager.add_strategy(strategy)
        
        # 균등 배분 확인
        for strategy in manager.strategies.values():
            assert strategy.current_capital == 1_000_000
    
    def test_portfolio_metrics(self):
        """포트폴리오 메트릭 테스트"""
        manager = StrategyManager()
        
        strategy = ThresholdStrategy()
        strategy.performance.total_pnl = 100_000
        strategy.performance.total_pnl_pct = 10
        manager.add_strategy(strategy)
        
        manager._update_portfolio_metrics()
        
        status = manager.get_portfolio_status()
        
        assert 'total_capital' in status
        assert 'total_pnl' in status
        assert 'active_strategies' in status
    
    def test_emergency_stop(self):
        """긴급 정지 테스트"""
        manager = StrategyManager(
            config={'emergency_stop_loss': -0.1}
        )
        
        strategy = ThresholdStrategy()
        manager.add_strategy(strategy)
        
        # 큰 손실 시뮬레이션
        manager.portfolio.total_pnl_pct = -11  # -11%
        
        should_rebalance = manager._check_risk_limits()
        assert should_rebalance is True
        
        # 전략이 정지되었는지 확인
        assert strategy.status == StrategyStatus.STOPPED


if __name__ == "__main__":
    pytest.main([__file__, "-v"])