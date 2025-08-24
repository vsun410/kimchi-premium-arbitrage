"""
멀티 전략 시스템 패키지
여러 전략을 동시에 실행하고 관리하는 시스템
"""

from .base_strategy import (
    BaseStrategy,
    MarketData,
    TradingSignal,
    SignalType,
    StrategyStatus,
    StrategyPerformance
)

from .threshold_strategy import ThresholdStrategy
from .ma_strategy import MovingAverageStrategy
from .bollinger_strategy import BollingerBandsStrategy
from .strategy_manager import (
    StrategyManager,
    AllocationMethod,
    SignalAggregation,
    PortfolioMetrics
)

__all__ = [
    # Base classes
    'BaseStrategy',
    'MarketData',
    'TradingSignal',
    'SignalType',
    'StrategyStatus',
    'StrategyPerformance',
    
    # Strategies
    'ThresholdStrategy',
    'MovingAverageStrategy',
    'BollingerBandsStrategy',
    
    # Manager
    'StrategyManager',
    'AllocationMethod',
    'SignalAggregation',
    'PortfolioMetrics'
]

__version__ = '1.0.0'