"""
멀티 전략 매니저 (Multi-Strategy Manager)
여러 전략을 조율하고 자본을 배분하는 중앙 관리자
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from enum import Enum
import logging
from dataclasses import dataclass, field
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor

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

logger = logging.getLogger(__name__)


class AllocationMethod(Enum):
    """자본 배분 방식"""
    EQUAL = "equal"              # 균등 배분
    PERFORMANCE = "performance"   # 성과 기반 배분
    VOLATILITY = "volatility"     # 변동성 기반 배분
    KELLY = "kelly"              # 켈리 공식 기반
    RISK_PARITY = "risk_parity"  # 리스크 패리티


class SignalAggregation(Enum):
    """신호 통합 방식"""
    UNANIMOUS = "unanimous"       # 모든 전략 동의
    MAJORITY = "majority"         # 과반수 동의
    WEIGHTED = "weighted"         # 가중 평균
    BEST = "best"                # 최고 신뢰도 선택


@dataclass
class PortfolioMetrics:
    """포트폴리오 메트릭"""
    total_capital: float
    allocated_capital: float
    free_capital: float
    total_positions: int
    total_pnl: float
    total_pnl_pct: float
    sharpe_ratio: float
    max_drawdown: float
    correlation_matrix: Dict = field(default_factory=dict)
    strategy_weights: Dict = field(default_factory=dict)
    last_rebalance: datetime = field(default_factory=datetime.now)


class StrategyManager:
    """
    멀티 전략 매니저
    
    여러 전략을 관리하고 조율하는 중앙 컨트롤러
    """
    
    def __init__(
        self,
        initial_capital: float = 10_000_000,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        초기화
        
        Args:
            initial_capital: 초기 자본금
            config: 매니저 설정
        """
        # 기본 설정
        self.default_config = {
            'allocation_method': AllocationMethod.PERFORMANCE,
            'signal_aggregation': SignalAggregation.WEIGHTED,
            'max_strategies': 10,
            'max_concurrent_positions': 3,
            'capital_per_position': 0.3,  # 포지션당 최대 자본 비율
            'min_strategy_capital': 500_000,  # 전략별 최소 자본
            'rebalance_interval': 3600,  # 재조정 주기 (초)
            'correlation_threshold': 0.8,  # 상관관계 임계값
            'risk_limit_daily': 0.05,  # 일일 리스크 한도 (5%)
            'emergency_stop_loss': -0.1,  # 긴급 손절 (-10%)
            'performance_window': 30,  # 성과 평가 기간 (일)
            'strategy_timeout': 5.0  # 전략 응답 타임아웃 (초)
        }
        
        if config:
            self.default_config.update(config)
        
        self.config = self.default_config
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        
        # 전략 저장소
        self.strategies: Dict[str, BaseStrategy] = {}
        self.strategy_weights: Dict[str, float] = {}
        
        # 포트폴리오 메트릭
        self.portfolio = PortfolioMetrics(
            total_capital=initial_capital,
            allocated_capital=0,
            free_capital=initial_capital,
            total_positions=0,
            total_pnl=0,
            total_pnl_pct=0,
            sharpe_ratio=0,
            max_drawdown=0
        )
        
        # 신호 이력
        self.signal_history: List[Dict] = []
        self.execution_history: List[Dict] = []
        
        # 비동기 실행을 위한 executor
        self.executor = ThreadPoolExecutor(max_workers=5)
        
        # 마지막 재조정 시간
        self.last_rebalance = datetime.now()
        
        logger.info(
            f"StrategyManager initialized with {initial_capital:,.0f} KRW capital"
        )
    
    def add_strategy(
        self,
        strategy: BaseStrategy,
        weight: float = None
    ) -> bool:
        """
        전략 추가
        
        Args:
            strategy: 추가할 전략
            weight: 초기 가중치 (None이면 자동 계산)
            
        Returns:
            추가 성공 여부
        """
        if len(self.strategies) >= self.config['max_strategies']:
            logger.warning(
                f"Cannot add strategy: max strategies ({self.config['max_strategies']}) reached"
            )
            return False
        
        if strategy.name in self.strategies:
            logger.warning(f"Strategy '{strategy.name}' already exists")
            return False
        
        # 전략 추가
        self.strategies[strategy.name] = strategy
        
        # 가중치 설정
        if weight is None:
            # 균등 배분으로 시작
            weight = 1.0 / (len(self.strategies))
            self._rebalance_weights()
        else:
            self.strategy_weights[strategy.name] = weight
        
        # 자본 할당
        self._allocate_capital()
        
        logger.info(f"Strategy '{strategy.name}' added with weight {weight:.2%}")
        return True
    
    def remove_strategy(self, strategy_name: str) -> bool:
        """
        전략 제거
        
        Args:
            strategy_name: 제거할 전략 이름
            
        Returns:
            제거 성공 여부
        """
        if strategy_name not in self.strategies:
            logger.warning(f"Strategy '{strategy_name}' not found")
            return False
        
        # 전략이 포지션을 보유중인지 체크
        strategy = self.strategies[strategy_name]
        if strategy.position != 0:
            logger.warning(
                f"Cannot remove strategy '{strategy_name}': has open position"
            )
            return False
        
        # 전략 제거
        del self.strategies[strategy_name]
        del self.strategy_weights[strategy_name]
        
        # 가중치 재조정
        self._rebalance_weights()
        self._allocate_capital()
        
        logger.info(f"Strategy '{strategy_name}' removed")
        return True
    
    async def analyze_market(
        self,
        market_data: MarketData
    ) -> List[TradingSignal]:
        """
        시장 분석 (비동기)
        
        모든 전략에 시장 데이터를 전달하고 신호 수집
        
        Args:
            market_data: 시장 데이터
            
        Returns:
            거래 신호 리스트
        """
        signals = []
        
        # 재조정이 필요한지 체크
        if self._should_rebalance():
            await self._async_rebalance()
        
        # 모든 전략에서 동시에 신호 수집
        tasks = []
        for name, strategy in self.strategies.items():
            if strategy.status == StrategyStatus.ACTIVE:
                task = asyncio.create_task(
                    self._get_strategy_signal(strategy, market_data)
                )
                tasks.append((name, task))
        
        # 타임아웃 처리
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*[task for _, task in tasks], return_exceptions=True),
                timeout=self.config['strategy_timeout']
            )
            
            for (name, _), result in zip(tasks, results):
                if isinstance(result, Exception):
                    logger.error(f"Strategy '{name}' error: {result}")
                elif result is not None:
                    signals.append(result)
                    
        except asyncio.TimeoutError:
            logger.error("Strategy analysis timeout")
        
        return signals
    
    async def _get_strategy_signal(
        self,
        strategy: BaseStrategy,
        market_data: MarketData
    ) -> Optional[TradingSignal]:
        """
        개별 전략에서 신호 가져오기 (비동기)
        
        Args:
            strategy: 전략
            market_data: 시장 데이터
            
        Returns:
            거래 신호
        """
        try:
            # 동기 함수를 비동기로 실행
            loop = asyncio.get_event_loop()
            signal = await loop.run_in_executor(
                self.executor,
                strategy.update,
                market_data
            )
            return signal
        except Exception as e:
            logger.error(f"Error getting signal from {strategy.name}: {e}")
            return None
    
    def aggregate_signals(
        self,
        signals: List[TradingSignal]
    ) -> Optional[TradingSignal]:
        """
        신호 통합
        
        여러 전략의 신호를 통합하여 최종 신호 생성
        
        Args:
            signals: 신호 리스트
            
        Returns:
            통합된 신호
        """
        if not signals:
            return None
        
        method = self.config['signal_aggregation']
        
        if method == SignalAggregation.UNANIMOUS:
            # 모든 전략이 동일한 신호
            signal_types = set(s.signal_type for s in signals)
            if len(signal_types) == 1 and len(signals) == len(self.strategies):
                return self._create_aggregated_signal(signals)
        
        elif method == SignalAggregation.MAJORITY:
            # 과반수 동의
            buy_signals = [s for s in signals if s.signal_type == SignalType.BUY]
            sell_signals = [s for s in signals if s.signal_type in [SignalType.SELL, SignalType.CLOSE]]
            
            threshold = len(self.strategies) / 2
            if len(buy_signals) > threshold:
                return self._create_aggregated_signal(buy_signals)
            elif len(sell_signals) > threshold:
                return self._create_aggregated_signal(sell_signals)
        
        elif method == SignalAggregation.WEIGHTED:
            # 가중 평균
            return self._weighted_aggregation(signals)
        
        elif method == SignalAggregation.BEST:
            # 최고 신뢰도 선택
            best_signal = max(signals, key=lambda s: s.confidence)
            if best_signal.confidence >= 0.6:  # 최소 신뢰도
                return best_signal
        
        return None
    
    def _create_aggregated_signal(
        self,
        signals: List[TradingSignal]
    ) -> TradingSignal:
        """
        통합 신호 생성
        
        Args:
            signals: 신호 리스트
            
        Returns:
            통합된 신호
        """
        # 평균 신뢰도 계산
        avg_confidence = sum(s.confidence for s in signals) / len(signals)
        
        # 가장 일반적인 신호 타입
        signal_type = max(set(s.signal_type for s in signals),
                         key=lambda x: sum(1 for s in signals if s.signal_type == x))
        
        # 제안 수량 합계
        total_amount = sum(s.suggested_amount for s in signals)
        
        # 통합 신호 생성
        aggregated = TradingSignal(
            timestamp=datetime.now(),
            strategy_name="StrategyManager",
            signal_type=signal_type,
            confidence=avg_confidence,
            suggested_amount=total_amount,
            reason=f"Aggregated signal from {len(signals)} strategies",
            metadata={
                'source_strategies': [s.strategy_name for s in signals],
                'individual_confidences': {s.strategy_name: s.confidence for s in signals}
            }
        )
        
        return aggregated
    
    def _weighted_aggregation(
        self,
        signals: List[TradingSignal]
    ) -> Optional[TradingSignal]:
        """
        가중 평균 신호 통합
        
        Args:
            signals: 신호 리스트
            
        Returns:
            가중 평균된 신호
        """
        # 신호별 가중치 적용
        weighted_confidence = 0
        total_weight = 0
        
        buy_weight = 0
        sell_weight = 0
        
        for signal in signals:
            weight = self.strategy_weights.get(signal.strategy_name, 0)
            weighted_confidence += signal.confidence * weight
            total_weight += weight
            
            if signal.signal_type == SignalType.BUY:
                buy_weight += weight * signal.confidence
            elif signal.signal_type in [SignalType.SELL, SignalType.CLOSE]:
                sell_weight += weight * signal.confidence
        
        if total_weight == 0:
            return None
        
        # 최종 신호 타입 결정
        if buy_weight > sell_weight and buy_weight > 0.3:
            signal_type = SignalType.BUY
        elif sell_weight > buy_weight and sell_weight > 0.3:
            signal_type = SignalType.SELL
        else:
            return None
        
        # 가중 평균 신뢰도
        avg_confidence = weighted_confidence / total_weight
        
        if avg_confidence < 0.5:  # 최소 신뢰도
            return None
        
        return TradingSignal(
            timestamp=datetime.now(),
            strategy_name="StrategyManager",
            signal_type=signal_type,
            confidence=avg_confidence,
            suggested_amount=0,
            reason=f"Weighted aggregation (buy:{buy_weight:.2f}, sell:{sell_weight:.2f})",
            metadata={
                'source_strategies': [s.strategy_name for s in signals],
                'buy_weight': buy_weight,
                'sell_weight': sell_weight
            }
        )
    
    def _allocate_capital(self):
        """
        자본 배분
        
        각 전략에 자본을 할당
        """
        method = self.config['allocation_method']
        
        if method == AllocationMethod.EQUAL:
            # 균등 배분
            capital_per_strategy = self.current_capital / len(self.strategies)
            for strategy in self.strategies.values():
                strategy.current_capital = capital_per_strategy
        
        elif method == AllocationMethod.PERFORMANCE:
            # 성과 기반 배분
            self._performance_based_allocation()
        
        elif method == AllocationMethod.VOLATILITY:
            # 변동성 기반 배분
            self._volatility_based_allocation()
        
        elif method == AllocationMethod.KELLY:
            # 켈리 공식 기반
            self._kelly_allocation()
        
        elif method == AllocationMethod.RISK_PARITY:
            # 리스크 패리티
            self._risk_parity_allocation()
        
        # 포트폴리오 메트릭 업데이트
        self._update_portfolio_metrics()
    
    def _performance_based_allocation(self):
        """성과 기반 자본 배분"""
        # 각 전략의 성과 점수 계산
        scores = {}
        for name, strategy in self.strategies.items():
            perf = strategy.performance
            # 승률과 수익률을 고려한 점수
            score = (perf.win_rate * 0.5 + 
                    (1 + perf.total_pnl_pct/100) * 0.5)
            scores[name] = max(0.1, score)  # 최소 점수 보장
        
        # 점수 정규화
        total_score = sum(scores.values())
        if total_score > 0:
            for name, strategy in self.strategies.items():
                weight = scores[name] / total_score
                self.strategy_weights[name] = weight
                strategy.current_capital = self.current_capital * weight
    
    def _volatility_based_allocation(self):
        """변동성 기반 자본 배분 (낮은 변동성에 더 많은 자본)"""
        # 간단한 구현: 균등 배분으로 대체
        capital_per_strategy = self.current_capital / len(self.strategies)
        for strategy in self.strategies.values():
            strategy.current_capital = capital_per_strategy
    
    def _kelly_allocation(self):
        """켈리 공식 기반 자본 배분"""
        for name, strategy in self.strategies.items():
            perf = strategy.performance
            if perf.total_trades > 0:
                # 켈리 비율 = (p * b - q) / b
                # p: 승률, q: 패률, b: 평균 이익/손실 비율
                p = perf.win_rate
                q = 1 - p
                b = abs(perf.avg_profit / perf.avg_loss) if perf.avg_loss != 0 else 1
                
                kelly_ratio = (p * b - q) / b if b != 0 else 0
                kelly_ratio = max(0, min(0.25, kelly_ratio))  # 0~25% 제한
                
                self.strategy_weights[name] = kelly_ratio
            else:
                self.strategy_weights[name] = 0.1  # 기본값
        
        # 정규화
        total_weight = sum(self.strategy_weights.values())
        if total_weight > 0:
            for name in self.strategy_weights:
                self.strategy_weights[name] /= total_weight
                self.strategies[name].current_capital = (
                    self.current_capital * self.strategy_weights[name]
                )
    
    def _risk_parity_allocation(self):
        """리스크 패리티 자본 배분"""
        # 간단한 구현: 균등 배분
        capital_per_strategy = self.current_capital / len(self.strategies)
        for strategy in self.strategies.values():
            strategy.current_capital = capital_per_strategy
    
    def _should_rebalance(self) -> bool:
        """
        재조정이 필요한지 확인
        
        Returns:
            재조정 필요 여부
        """
        # 시간 기반 재조정
        time_since_rebalance = (datetime.now() - self.last_rebalance).total_seconds()
        if time_since_rebalance >= self.config['rebalance_interval']:
            return True
        
        # 성과 편차가 클 때
        if self._check_performance_deviation():
            return True
        
        # 리스크 한도 초과
        if self._check_risk_limits():
            return True
        
        return False
    
    async def _async_rebalance(self):
        """비동기 재조정"""
        logger.info("Rebalancing portfolio...")
        
        # 가중치 재계산
        self._rebalance_weights()
        
        # 자본 재배분
        self._allocate_capital()
        
        # 메트릭 업데이트
        self._update_portfolio_metrics()
        
        self.last_rebalance = datetime.now()
        logger.info("Portfolio rebalanced")
    
    def _rebalance_weights(self):
        """가중치 재조정"""
        if not self.strategies:
            return
        
        # 균등 배분으로 초기화
        equal_weight = 1.0 / len(self.strategies)
        for name in self.strategies:
            self.strategy_weights[name] = equal_weight
    
    def _check_performance_deviation(self) -> bool:
        """성과 편차 체크"""
        if len(self.strategies) < 2:
            return False
        
        performances = [s.performance.total_pnl_pct for s in self.strategies.values()]
        if performances:
            max_perf = max(performances)
            min_perf = min(performances)
            deviation = max_perf - min_perf
            return deviation > 20  # 20% 이상 편차
        
        return False
    
    def _check_risk_limits(self) -> bool:
        """리스크 한도 체크"""
        # 일일 손실 한도
        daily_loss = self.portfolio.total_pnl_pct
        if daily_loss < -self.config['risk_limit_daily'] * 100:
            return True
        
        # 긴급 손절
        if daily_loss < self.config['emergency_stop_loss'] * 100:
            logger.warning(f"Emergency stop loss triggered: {daily_loss:.2f}%")
            self._emergency_stop()
            return True
        
        return False
    
    def _emergency_stop(self):
        """긴급 정지"""
        logger.critical("EMERGENCY STOP - Closing all positions")
        
        # 모든 전략 정지
        for strategy in self.strategies.values():
            strategy.stop()
            # 포지션이 있으면 청산 신호 생성
            if strategy.position != 0:
                # 실제 구현에서는 청산 로직 추가
                pass
    
    def _update_portfolio_metrics(self):
        """포트폴리오 메트릭 업데이트"""
        # 전체 자본
        self.portfolio.total_capital = self.current_capital
        
        # 할당된 자본
        allocated = sum(
            s.current_capital for s in self.strategies.values()
            if s.position != 0
        )
        self.portfolio.allocated_capital = allocated
        self.portfolio.free_capital = self.current_capital - allocated
        
        # 포지션 수
        self.portfolio.total_positions = sum(
            1 for s in self.strategies.values() if s.position != 0
        )
        
        # 총 PnL
        total_pnl = sum(s.performance.total_pnl for s in self.strategies.values())
        self.portfolio.total_pnl = total_pnl
        self.portfolio.total_pnl_pct = (total_pnl / self.initial_capital) * 100
        
        # 전략 가중치
        self.portfolio.strategy_weights = self.strategy_weights.copy()
    
    def get_portfolio_status(self) -> Dict:
        """
        포트폴리오 상태 조회
        
        Returns:
            포트폴리오 상태 정보
        """
        return {
            'total_capital': self.portfolio.total_capital,
            'allocated_capital': self.portfolio.allocated_capital,
            'free_capital': self.portfolio.free_capital,
            'total_positions': self.portfolio.total_positions,
            'total_pnl': self.portfolio.total_pnl,
            'total_pnl_pct': f"{self.portfolio.total_pnl_pct:.2f}%",
            'active_strategies': len([s for s in self.strategies.values() 
                                    if s.status == StrategyStatus.ACTIVE]),
            'strategy_weights': {
                name: f"{weight:.2%}" 
                for name, weight in self.strategy_weights.items()
            },
            'last_rebalance': self.portfolio.last_rebalance.isoformat()
        }
    
    def get_all_strategies_status(self) -> List[Dict]:
        """
        모든 전략의 상태 조회
        
        Returns:
            전략별 상태 리스트
        """
        statuses = []
        for strategy in self.strategies.values():
            status = strategy.get_performance_summary()
            status['allocated_capital'] = strategy.current_capital
            status['weight'] = f"{self.strategy_weights.get(strategy.name, 0):.2%}"
            statuses.append(status)
        
        return statuses