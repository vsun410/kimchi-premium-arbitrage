"""
멀티 전략 시스템 - 베이스 전략 클래스
모든 전략이 상속받아야 할 인터페이스 정의
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """거래 신호 타입"""
    BUY = "BUY"        # 매수 (진입)
    SELL = "SELL"      # 매도 (청산)
    HOLD = "HOLD"      # 대기
    CLOSE = "CLOSE"    # 포지션 종료


class StrategyStatus(Enum):
    """전략 상태"""
    ACTIVE = "active"      # 활성
    PAUSED = "paused"      # 일시정지
    STOPPED = "stopped"    # 중지
    ERROR = "error"        # 에러


@dataclass
class MarketData:
    """시장 데이터"""
    timestamp: datetime
    upbit_price: float      # 업비트 BTC 가격 (KRW)
    binance_price: float    # 바이낸스 BTC 가격 (USDT)
    exchange_rate: float    # USD/KRW 환율
    kimchi_premium: float   # 김치 프리미엄 (%)
    volume_upbit: float     # 업비트 거래량
    volume_binance: float   # 바이낸스 거래량
    bid_ask_spread_upbit: float   # 업비트 스프레드
    bid_ask_spread_binance: float # 바이낸스 스프레드
    
    def to_dict(self) -> Dict:
        """딕셔너리 변환"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'upbit_price': self.upbit_price,
            'binance_price': self.binance_price,
            'exchange_rate': self.exchange_rate,
            'kimchi_premium': self.kimchi_premium,
            'volume_upbit': self.volume_upbit,
            'volume_binance': self.volume_binance,
            'bid_ask_spread_upbit': self.bid_ask_spread_upbit,
            'bid_ask_spread_binance': self.bid_ask_spread_binance
        }


@dataclass
class TradingSignal:
    """거래 신호"""
    timestamp: datetime
    strategy_name: str
    signal_type: SignalType
    confidence: float       # 신뢰도 (0~1)
    suggested_amount: float # 제안 거래량
    reason: str            # 신호 이유
    metadata: Dict = field(default_factory=dict)  # 추가 정보
    
    def __str__(self) -> str:
        return (
            f"[{self.strategy_name}] {self.signal_type.value} "
            f"(신뢰도: {self.confidence:.2%}, 수량: {self.suggested_amount:.4f})"
        )


@dataclass
class StrategyPerformance:
    """전략 성과"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0
    total_pnl_pct: float = 0
    max_drawdown: float = 0
    sharpe_ratio: float = 0
    win_rate: float = 0
    avg_profit: float = 0
    avg_loss: float = 0
    best_trade: float = 0
    worst_trade: float = 0
    current_position: float = 0
    last_update: datetime = field(default_factory=datetime.now)
    
    def update_metrics(self):
        """메트릭 업데이트"""
        if self.total_trades > 0:
            self.win_rate = self.winning_trades / self.total_trades
        
        if self.winning_trades > 0:
            self.avg_profit = (
                sum([p for p in [self.total_pnl] if p > 0]) / self.winning_trades
            )
        
        if self.losing_trades > 0:
            self.avg_loss = (
                sum([p for p in [self.total_pnl] if p < 0]) / self.losing_trades
            )
        
        self.last_update = datetime.now()


class BaseStrategy(ABC):
    """
    베이스 전략 클래스
    
    모든 전략이 구현해야 할 인터페이스 정의
    """
    
    def __init__(
        self,
        name: str,
        config: Dict[str, Any],
        initial_capital: float = 1_000_000  # 초기 자본금 (KRW)
    ):
        """
        초기화
        
        Args:
            name: 전략 이름
            config: 전략 설정
            initial_capital: 초기 자본금
        """
        self.name = name
        self.config = config
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        
        # 상태
        self.status = StrategyStatus.ACTIVE
        self.position = 0  # 현재 포지션
        self.entry_price = 0  # 진입 가격
        self.entry_time = None  # 진입 시간
        
        # 성과 추적
        self.performance = StrategyPerformance()
        
        # 거래 이력
        self.trade_history: List[Dict] = []
        self.signal_history: List[TradingSignal] = []
        
        # 데이터 버퍼
        self.data_buffer: List[MarketData] = []
        self.buffer_size = config.get('buffer_size', 100)
        
        logger.info(f"Strategy '{name}' initialized with capital: {initial_capital:,.0f} KRW")
    
    @abstractmethod
    def analyze(self, market_data: MarketData) -> Optional[TradingSignal]:
        """
        시장 데이터 분석 및 신호 생성
        
        Args:
            market_data: 현재 시장 데이터
            
        Returns:
            거래 신호 또는 None
        """
        pass
    
    @abstractmethod
    def calculate_position_size(self, signal: TradingSignal) -> float:
        """
        포지션 크기 계산
        
        Args:
            signal: 거래 신호
            
        Returns:
            포지션 크기
        """
        pass
    
    @abstractmethod
    def should_close_position(self, market_data: MarketData) -> bool:
        """
        포지션 청산 여부 결정
        
        Args:
            market_data: 현재 시장 데이터
            
        Returns:
            청산 여부
        """
        pass
    
    def update(self, market_data: MarketData) -> Optional[TradingSignal]:
        """
        전략 업데이트 및 신호 생성
        
        Args:
            market_data: 시장 데이터
            
        Returns:
            거래 신호
        """
        # 상태 체크
        if self.status != StrategyStatus.ACTIVE:
            return None
        
        # 데이터 버퍼 업데이트
        self._update_buffer(market_data)
        
        # 포지션이 있는 경우 청산 체크
        if self.position != 0:
            if self.should_close_position(market_data):
                signal = self._create_close_signal(market_data)
                self._record_signal(signal)
                return signal
        
        # 새로운 신호 분석
        signal = self.analyze(market_data)
        
        if signal:
            # 포지션 크기 계산
            signal.suggested_amount = self.calculate_position_size(signal)
            self._record_signal(signal)
        
        return signal
    
    def _update_buffer(self, market_data: MarketData):
        """데이터 버퍼 업데이트"""
        self.data_buffer.append(market_data)
        
        # 버퍼 크기 유지
        if len(self.data_buffer) > self.buffer_size:
            self.data_buffer = self.data_buffer[-self.buffer_size:]
    
    def _create_close_signal(self, market_data: MarketData) -> TradingSignal:
        """청산 신호 생성"""
        return TradingSignal(
            timestamp=market_data.timestamp,
            strategy_name=self.name,
            signal_type=SignalType.CLOSE,
            confidence=1.0,
            suggested_amount=abs(self.position),
            reason="Position close condition met",
            metadata={
                'entry_price': self.entry_price,
                'current_price': market_data.upbit_price,
                'position_duration': (
                    (market_data.timestamp - self.entry_time).total_seconds() / 3600
                    if self.entry_time else 0
                )
            }
        )
    
    def _record_signal(self, signal: TradingSignal):
        """신호 기록"""
        self.signal_history.append(signal)
        
        # 최대 1000개 유지
        if len(self.signal_history) > 1000:
            self.signal_history = self.signal_history[-1000:]
    
    def execute_trade(self, signal: TradingSignal, execution_price: float) -> bool:
        """
        거래 실행 (시뮬레이션)
        
        Args:
            signal: 거래 신호
            execution_price: 체결 가격
            
        Returns:
            실행 성공 여부
        """
        try:
            if signal.signal_type == SignalType.BUY:
                # 매수
                self.position = signal.suggested_amount
                self.entry_price = execution_price
                self.entry_time = signal.timestamp
                
                trade = {
                    'timestamp': signal.timestamp,
                    'type': 'BUY',
                    'amount': signal.suggested_amount,
                    'price': execution_price,
                    'reason': signal.reason
                }
                
            elif signal.signal_type in [SignalType.SELL, SignalType.CLOSE]:
                # 매도/청산
                if self.position > 0:
                    pnl = (execution_price - self.entry_price) * self.position
                    pnl_pct = (execution_price / self.entry_price - 1) * 100
                    
                    # 성과 업데이트
                    self.performance.total_trades += 1
                    self.performance.total_pnl += pnl
                    self.performance.total_pnl_pct += pnl_pct
                    
                    if pnl > 0:
                        self.performance.winning_trades += 1
                        self.performance.best_trade = max(self.performance.best_trade, pnl)
                    else:
                        self.performance.losing_trades += 1
                        self.performance.worst_trade = min(self.performance.worst_trade, pnl)
                    
                    self.performance.update_metrics()
                    
                    trade = {
                        'timestamp': signal.timestamp,
                        'type': 'SELL',
                        'amount': self.position,
                        'price': execution_price,
                        'pnl': pnl,
                        'pnl_pct': pnl_pct,
                        'reason': signal.reason
                    }
                    
                    # 포지션 초기화
                    self.position = 0
                    self.entry_price = 0
                    self.entry_time = None
                else:
                    return False
            else:
                return False
            
            # 거래 기록
            self.trade_history.append(trade)
            logger.info(f"[{self.name}] Trade executed: {trade}")
            
            return True
            
        except Exception as e:
            logger.error(f"[{self.name}] Trade execution failed: {e}")
            return False
    
    def get_performance_summary(self) -> Dict:
        """
        성과 요약 조회
        
        Returns:
            성과 요약 딕셔너리
        """
        return {
            'strategy_name': self.name,
            'status': self.status.value,
            'current_position': self.position,
            'total_trades': self.performance.total_trades,
            'win_rate': f"{self.performance.win_rate:.2%}",
            'total_pnl': self.performance.total_pnl,
            'total_pnl_pct': f"{self.performance.total_pnl_pct:.2f}%",
            'best_trade': self.performance.best_trade,
            'worst_trade': self.performance.worst_trade,
            'current_capital': self.current_capital,
            'last_update': self.performance.last_update.isoformat()
        }
    
    def reset(self):
        """전략 리셋"""
        self.position = 0
        self.entry_price = 0
        self.entry_time = None
        self.current_capital = self.initial_capital
        self.performance = StrategyPerformance()
        self.trade_history = []
        self.signal_history = []
        self.data_buffer = []
        self.status = StrategyStatus.ACTIVE
        
        logger.info(f"Strategy '{self.name}' has been reset")
    
    def pause(self):
        """전략 일시정지"""
        self.status = StrategyStatus.PAUSED
        logger.info(f"Strategy '{self.name}' paused")
    
    def resume(self):
        """전략 재개"""
        self.status = StrategyStatus.ACTIVE
        logger.info(f"Strategy '{self.name}' resumed")
    
    def stop(self):
        """전략 중지"""
        self.status = StrategyStatus.STOPPED
        logger.info(f"Strategy '{self.name}' stopped")