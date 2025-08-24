"""
Base Strategy Class
모든 전략의 베이스 클래스
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Any, List
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class Signal:
    """거래 신호"""
    timestamp: datetime
    action: str  # 'buy', 'sell', 'hold', 'close'
    symbol: str
    exchange: str
    amount: float
    price: Optional[float] = None  # None for market order
    order_type: str = 'market'  # 'market' or 'limit'
    confidence: float = 0.0  # 0.0 ~ 1.0
    reason: str = ""
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict:
        """딕셔너리 변환"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'action': self.action,
            'symbol': self.symbol,
            'exchange': self.exchange,
            'amount': self.amount,
            'price': self.price,
            'order_type': self.order_type,
            'confidence': self.confidence,
            'reason': self.reason,
            'metadata': self.metadata or {}
        }


@dataclass
class Position:
    """포지션 정보"""
    id: str
    symbol: str
    exchange: str
    side: str  # 'long' or 'short'
    amount: float
    entry_price: float
    current_price: float
    pnl: float
    pnl_percent: float
    opened_at: datetime
    metadata: Dict[str, Any] = None


class BaseStrategy(ABC):
    """
    전략 베이스 클래스
    
    모든 전략은 이 클래스를 상속받아 구현
    """
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        self.name = name
        self.config = config or {}
        self.is_running = False
        self.positions: List[Position] = []
        self.performance = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'win_rate': 0.0
        }
        
    @abstractmethod
    async def analyze(self, market_data: Dict[str, Any]) -> Optional[Signal]:
        """
        시장 데이터 분석 및 신호 생성
        
        Args:
            market_data: 시장 데이터
            
        Returns:
            Signal or None
        """
        pass
    
    @abstractmethod
    async def calculate_position_size(self, signal: Signal, capital: float) -> float:
        """
        포지션 크기 계산
        
        Args:
            signal: 거래 신호
            capital: 가용 자본
            
        Returns:
            포지션 크기
        """
        pass
    
    @abstractmethod
    async def should_close_position(self, position: Position, market_data: Dict) -> bool:
        """
        포지션 청산 여부 결정
        
        Args:
            position: 현재 포지션
            market_data: 시장 데이터
            
        Returns:
            청산 여부
        """
        pass
    
    async def start(self):
        """전략 시작"""
        self.is_running = True
        logger.info(f"Strategy {self.name} started")
    
    async def stop(self):
        """전략 중지"""
        self.is_running = False
        logger.info(f"Strategy {self.name} stopped")
    
    async def on_signal(self, signal: Signal):
        """신호 발생 시 처리"""
        logger.info(f"Signal generated: {signal.action} {signal.symbol} @ {signal.price}")
    
    async def on_position_opened(self, position: Position):
        """포지션 오픈 시 처리"""
        self.positions.append(position)
        logger.info(f"Position opened: {position.id}")
    
    async def on_position_closed(self, position: Position):
        """포지션 청산 시 처리"""
        # 성과 업데이트
        self.performance['total_trades'] += 1
        if position.pnl > 0:
            self.performance['winning_trades'] += 1
        else:
            self.performance['losing_trades'] += 1
        
        self.performance['total_pnl'] += position.pnl
        
        if self.performance['total_trades'] > 0:
            self.performance['win_rate'] = (
                self.performance['winning_trades'] / self.performance['total_trades']
            )
        
        # 포지션 제거
        self.positions = [p for p in self.positions if p.id != position.id]
        logger.info(f"Position closed: {position.id}, PnL: {position.pnl:.2f}")
    
    def get_performance(self) -> Dict[str, Any]:
        """성과 조회"""
        return self.performance.copy()
    
    def get_active_positions(self) -> List[Position]:
        """활성 포지션 조회"""
        return self.positions.copy()
    
    def update_config(self, config: Dict[str, Any]):
        """설정 업데이트"""
        self.config.update(config)
        logger.info(f"Strategy {self.name} config updated")
    
    def validate_signal(self, signal: Signal) -> bool:
        """신호 유효성 검증"""
        # 기본 검증
        if signal.amount <= 0:
            logger.warning(f"Invalid signal: amount <= 0")
            return False
        
        if signal.action not in ['buy', 'sell', 'hold', 'close']:
            logger.warning(f"Invalid signal: unknown action {signal.action}")
            return False
        
        if signal.order_type == 'limit' and signal.price is None:
            logger.warning(f"Invalid signal: limit order without price")
            return False
        
        return True
    
    def calculate_risk_metrics(self) -> Dict[str, float]:
        """리스크 지표 계산"""
        if not self.positions:
            return {
                'total_exposure': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0
            }
        
        total_exposure = sum(p.amount * p.current_price for p in self.positions)
        
        # 간단한 최대 낙폭 계산
        pnls = [p.pnl for p in self.positions]
        cumulative_pnl = []
        cum = 0
        for pnl in pnls:
            cum += pnl
            cumulative_pnl.append(cum)
        
        if cumulative_pnl:
            peak = cumulative_pnl[0]
            max_dd = 0
            for value in cumulative_pnl:
                if value > peak:
                    peak = value
                dd = (peak - value) / peak if peak > 0 else 0
                max_dd = max(max_dd, dd)
        else:
            max_dd = 0
        
        return {
            'total_exposure': total_exposure,
            'max_drawdown': max_dd,
            'sharpe_ratio': 0.0  # TODO: 구현 필요
        }