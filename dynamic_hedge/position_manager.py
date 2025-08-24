"""
Task 30: Dynamic Position Manager
기본 헤지 상태 관리 및 조건부 청산 로직
"""

import asyncio
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class PositionType(Enum):
    """포지션 타입"""
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"


class PositionStatus(Enum):
    """포지션 상태"""
    OPEN = "open"
    CLOSING = "closing"
    CLOSED = "closed"
    PENDING = "pending"


@dataclass
class Position:
    """포지션 정보"""
    exchange: str
    symbol: str
    position_type: PositionType
    size: float
    entry_price: float
    current_price: float
    status: PositionStatus
    opened_at: datetime
    closed_at: Optional[datetime] = None
    pnl: float = 0.0
    fees: float = 0.0
    
    def calculate_pnl(self, current_price: float) -> float:
        """손익 계산"""
        if self.position_type == PositionType.LONG:
            return (current_price - self.entry_price) * self.size - self.fees
        else:  # SHORT
            return (self.entry_price - current_price) * self.size - self.fees


@dataclass
class HedgeState:
    """헤지 상태 정보"""
    upbit_position: Optional[Position] = None
    binance_position: Optional[Position] = None
    total_capital: float = 40000000  # 4000만원
    used_capital: float = 0
    is_hedged: bool = False
    last_update: datetime = field(default_factory=datetime.now)
    
    @property
    def available_capital(self) -> float:
        """사용 가능한 자본"""
        return self.total_capital - self.used_capital
    
    @property
    def hedge_ratio(self) -> float:
        """헤지 비율 (1.0 = 완전 헤지)"""
        if not self.upbit_position or not self.binance_position:
            return 0.0
        return min(
            self.upbit_position.size / self.binance_position.size,
            self.binance_position.size / self.upbit_position.size
        )


@dataclass
class ExitCondition:
    """청산 조건"""
    condition_type: str  # 'breakout', 'premium', 'stop_loss', 'take_profit'
    threshold: float
    action: str  # 'close_all', 'close_long', 'close_short'
    priority: int  # 우선순위 (낮을수록 높음)


class DynamicPositionManager:
    """
    동적 포지션 관리자
    - 기본 헤지 상태 관리
    - 조건부 청산 로직
    - 재진입 타이밍 계산
    """
    
    def __init__(self, capital_per_exchange: float = 20000000):
        """
        Args:
            capital_per_exchange: 거래소별 자본금 (기본 2000만원)
        """
        self.capital_per_exchange = capital_per_exchange
        self.hedge_state = HedgeState(total_capital=capital_per_exchange * 2)
        self.exit_conditions: List[ExitCondition] = self._setup_exit_conditions()
        self.position_history: List[Position] = []
        self.last_exit_time: Optional[datetime] = None
        self.cooldown_period = timedelta(minutes=30)  # 재진입 쿨다운
        
    def _setup_exit_conditions(self) -> List[ExitCondition]:
        """기본 청산 조건 설정"""
        return [
            # 김프 14만원(0.0014 BTC) 이상 시 전체 청산
            ExitCondition(
                condition_type='premium',
                threshold=0.0014,
                action='close_all',
                priority=1
            ),
            # 상승 돌파 시 숏 포지션만 청산
            ExitCondition(
                condition_type='breakout_up',
                threshold=0.02,  # 2% 돌파
                action='close_short',
                priority=2
            ),
            # 하락 돌파 시 롱 포지션만 청산
            ExitCondition(
                condition_type='breakout_down',
                threshold=-0.02,  # -2% 돌파
                action='close_long',
                priority=2
            ),
            # 손실 제한
            ExitCondition(
                condition_type='stop_loss',
                threshold=-0.05,  # -5% 손실
                action='close_all',
                priority=0
            ),
        ]
    
    async def open_hedge_position(self, upbit_price: float, binance_price: float,
                                 position_size: float) -> bool:
        """
        헤지 포지션 오픈
        
        Args:
            upbit_price: 업비트 현재가
            binance_price: 바이낸스 현재가
            position_size: 포지션 크기 (BTC)
            
        Returns:
            성공 여부
        """
        try:
            # 자본금 확인
            required_capital = position_size * (upbit_price + binance_price)
            if required_capital > self.hedge_state.available_capital:
                logger.warning(f"Insufficient capital: required={required_capital}, available={self.hedge_state.available_capital}")
                return False
            
            # 업비트 롱 포지션
            self.hedge_state.upbit_position = Position(
                exchange='upbit',
                symbol='BTC/KRW',
                position_type=PositionType.LONG,
                size=position_size,
                entry_price=upbit_price,
                current_price=upbit_price,
                status=PositionStatus.OPEN,
                opened_at=datetime.now()
            )
            
            # 바이낸스 숏 포지션
            self.hedge_state.binance_position = Position(
                exchange='binance',
                symbol='BTC/USDT',
                position_type=PositionType.SHORT,
                size=position_size,
                entry_price=binance_price,
                current_price=binance_price,
                status=PositionStatus.OPEN,
                opened_at=datetime.now()
            )
            
            # 상태 업데이트
            self.hedge_state.used_capital = required_capital
            self.hedge_state.is_hedged = True
            self.hedge_state.last_update = datetime.now()
            
            logger.info(f"Hedge position opened: size={position_size} BTC, upbit={upbit_price}, binance={binance_price}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to open hedge position: {e}")
            return False
    
    def check_exit_conditions(self, market_data: Dict) -> Optional[ExitCondition]:
        """
        청산 조건 확인
        
        Args:
            market_data: {
                'kimchi_premium': float,
                'breakout_signal': Optional[str],  # 'up', 'down', None
                'breakout_strength': float,
                'current_pnl': float
            }
            
        Returns:
            충족된 청산 조건 또는 None
        """
        if not self.hedge_state.is_hedged:
            return None
            
        triggered_conditions = []
        
        # 각 조건 확인
        for condition in self.exit_conditions:
            if self._check_single_condition(condition, market_data):
                triggered_conditions.append(condition)
        
        # 우선순위가 가장 높은 조건 반환
        if triggered_conditions:
            return min(triggered_conditions, key=lambda x: x.priority)
            
        return None
    
    def _check_single_condition(self, condition: ExitCondition, 
                               market_data: Dict) -> bool:
        """
        단일 청산 조건 확인
        
        Args:
            condition: 청산 조건
            market_data: 시장 데이터
            
        Returns:
            조건 충족 여부
        """
        if condition.condition_type == 'premium':
            return market_data.get('kimchi_premium', 0) >= condition.threshold
            
        elif condition.condition_type == 'breakout_up':
            return (market_data.get('breakout_signal') == 'up' and 
                   market_data.get('breakout_strength', 0) >= condition.threshold)
            
        elif condition.condition_type == 'breakout_down':
            return (market_data.get('breakout_signal') == 'down' and 
                   market_data.get('breakout_strength', 0) <= condition.threshold)
            
        elif condition.condition_type == 'stop_loss':
            pnl_ratio = market_data.get('current_pnl', 0) / self.hedge_state.used_capital
            return pnl_ratio <= condition.threshold
            
        return False
    
    async def execute_exit(self, condition: ExitCondition, 
                          current_prices: Dict) -> Dict:
        """
        청산 실행
        
        Args:
            condition: 청산 조건
            current_prices: {'upbit': float, 'binance': float}
            
        Returns:
            실행 결과
        """
        result = {
            'success': False,
            'closed_positions': [],
            'total_pnl': 0,
            'action': condition.action
        }
        
        try:
            if condition.action == 'close_all':
                # 전체 청산
                if self.hedge_state.upbit_position:
                    pnl = self._close_position(
                        self.hedge_state.upbit_position, 
                        current_prices['upbit']
                    )
                    result['closed_positions'].append('upbit_long')
                    result['total_pnl'] += pnl
                    
                if self.hedge_state.binance_position:
                    pnl = self._close_position(
                        self.hedge_state.binance_position,
                        current_prices['binance']
                    )
                    result['closed_positions'].append('binance_short')
                    result['total_pnl'] += pnl
                    
                self.hedge_state.is_hedged = False
                
            elif condition.action == 'close_long':
                # 롱 포지션만 청산
                if self.hedge_state.upbit_position:
                    pnl = self._close_position(
                        self.hedge_state.upbit_position,
                        current_prices['upbit']
                    )
                    result['closed_positions'].append('upbit_long')
                    result['total_pnl'] += pnl
                    self.hedge_state.upbit_position = None
                    
            elif condition.action == 'close_short':
                # 숏 포지션만 청산
                if self.hedge_state.binance_position:
                    pnl = self._close_position(
                        self.hedge_state.binance_position,
                        current_prices['binance']
                    )
                    result['closed_positions'].append('binance_short')
                    result['total_pnl'] += pnl
                    self.hedge_state.binance_position = None
            
            # 상태 업데이트
            self.last_exit_time = datetime.now()
            self.hedge_state.last_update = datetime.now()
            
            # 부분 청산 후 헤지 상태 재평가
            if self.hedge_state.upbit_position and not self.hedge_state.binance_position:
                self.hedge_state.is_hedged = False
            elif not self.hedge_state.upbit_position and self.hedge_state.binance_position:
                self.hedge_state.is_hedged = False
                
            result['success'] = True
            logger.info(f"Exit executed: {condition.action}, PnL={result['total_pnl']}")
            
        except Exception as e:
            logger.error(f"Failed to execute exit: {e}")
            
        return result
    
    def _close_position(self, position: Position, current_price: float) -> float:
        """
        포지션 청산
        
        Args:
            position: 포지션 객체
            current_price: 현재가
            
        Returns:
            실현 손익
        """
        position.current_price = current_price
        position.pnl = position.calculate_pnl(current_price)
        position.status = PositionStatus.CLOSED
        position.closed_at = datetime.now()
        
        # 히스토리에 추가
        self.position_history.append(position)
        
        # 자본금 해제
        self.hedge_state.used_capital -= position.size * position.entry_price
        
        return position.pnl
    
    def calculate_reentry_timing(self, market_conditions: Dict) -> Dict:
        """
        재진입 타이밍 계산
        
        Args:
            market_conditions: {
                'volatility': float,
                'trend': str,  # 'up', 'down', 'sideways'
                'premium_trend': str,  # 'increasing', 'decreasing', 'stable'
                'volume': float
            }
            
        Returns:
            {
                'can_reenter': bool,
                'recommended_wait': int,  # 분 단위
                'optimal_conditions': Dict
            }
        """
        # 쿨다운 확인
        if self.last_exit_time:
            time_since_exit = datetime.now() - self.last_exit_time
            if time_since_exit < self.cooldown_period:
                remaining = (self.cooldown_period - time_since_exit).seconds // 60
                return {
                    'can_reenter': False,
                    'recommended_wait': remaining,
                    'optimal_conditions': {}
                }
        
        # 시장 조건 평가
        volatility = market_conditions.get('volatility', 0)
        trend = market_conditions.get('trend', 'sideways')
        premium_trend = market_conditions.get('premium_trend', 'stable')
        
        # 최적 재진입 조건
        optimal_conditions = {
            'volatility_range': (0.01, 0.03),  # 1-3% 변동성
            'preferred_trend': 'sideways',
            'premium_stable': True
        }
        
        # 재진입 가능 여부 판단
        can_reenter = (
            0.01 <= volatility <= 0.03 and
            trend == 'sideways' and
            premium_trend == 'stable'
        )
        
        # 권장 대기 시간 계산
        if not can_reenter:
            if volatility > 0.03:
                recommended_wait = int(volatility * 1000)  # 높은 변동성일수록 더 대기
            else:
                recommended_wait = 15  # 기본 15분
        else:
            recommended_wait = 0
            
        return {
            'can_reenter': can_reenter,
            'recommended_wait': recommended_wait,
            'optimal_conditions': optimal_conditions
        }
    
    def get_position_status(self) -> Dict:
        """
        현재 포지션 상태 조회
        
        Returns:
            포지션 상태 정보
        """
        status = {
            'is_hedged': self.hedge_state.is_hedged,
            'hedge_ratio': self.hedge_state.hedge_ratio,
            'used_capital': self.hedge_state.used_capital,
            'available_capital': self.hedge_state.available_capital,
            'positions': {}
        }
        
        if self.hedge_state.upbit_position:
            pos = self.hedge_state.upbit_position
            status['positions']['upbit'] = {
                'type': pos.position_type.value,
                'size': pos.size,
                'entry_price': pos.entry_price,
                'current_pnl': pos.pnl,
                'status': pos.status.value
            }
            
        if self.hedge_state.binance_position:
            pos = self.hedge_state.binance_position
            status['positions']['binance'] = {
                'type': pos.position_type.value,
                'size': pos.size,
                'entry_price': pos.entry_price,
                'current_pnl': pos.pnl,
                'status': pos.status.value
            }
            
        return status
    
    def calculate_optimal_position_size(self, available_capital: float,
                                       risk_level: float = 0.02) -> float:
        """
        최적 포지션 크기 계산 (Kelly Criterion 변형)
        
        Args:
            available_capital: 사용 가능 자본
            risk_level: 리스크 레벨 (0.02 = 2%)
            
        Returns:
            최적 포지션 크기 (BTC)
        """
        # Kelly Criterion: f = (p * b - q) / b
        # p: 승률, q: 패율, b: 승리 시 배율
        
        # 과거 데이터 기반 승률 계산
        if len(self.position_history) >= 10:
            wins = sum(1 for p in self.position_history[-10:] if p.pnl > 0)
            win_rate = wins / 10
        else:
            win_rate = 0.6  # 기본 승률 60%
            
        # 평균 수익/손실 비율
        if self.position_history:
            profits = [p.pnl for p in self.position_history if p.pnl > 0]
            losses = [abs(p.pnl) for p in self.position_history if p.pnl < 0]
            
            if profits and losses:
                avg_profit = sum(profits) / len(profits)
                avg_loss = sum(losses) / len(losses)
                profit_loss_ratio = avg_profit / avg_loss
            else:
                profit_loss_ratio = 1.5
        else:
            profit_loss_ratio = 1.5
            
        # Kelly 비율 계산
        kelly_ratio = (win_rate * profit_loss_ratio - (1 - win_rate)) / profit_loss_ratio
        
        # 보수적 적용 (Kelly의 25%)
        conservative_ratio = kelly_ratio * 0.25
        
        # 리스크 레벨 적용
        final_ratio = min(conservative_ratio, risk_level)
        
        # BTC 가격 추정 (실제로는 현재가 사용)
        btc_price_estimate = 100000000  # 1억원
        
        # 포지션 크기 계산
        position_value = available_capital * final_ratio
        position_size = position_value / btc_price_estimate
        
        return position_size