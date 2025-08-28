"""
Position Tracker for Realtime Trading
실시간 포지션 추적
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class PositionInfo:
    """포지션 정보"""
    exchange: str
    symbol: str
    side: str  # 'long' or 'short'
    amount: float
    entry_price: float
    entry_time: datetime
    current_price: float = 0
    pnl: float = 0
    pnl_pct: float = 0
    high_price: float = 0
    low_price: float = 0
    duration_hours: float = 0


class PositionTracker:
    """
    포지션 추적 시스템
    - 실시간 포지션 모니터링
    - PnL 계산
    - 리스크 메트릭 추적
    """
    
    def __init__(self):
        """초기화"""
        self.positions: Dict[str, PositionInfo] = {}
        self.position_history: List[Dict] = []
        self.total_pnl = 0
        self.realized_pnl = 0
        self.unrealized_pnl = 0
        
        logger.info("PositionTracker initialized")
    
    def add_position(self,
                    exchange: str,
                    symbol: str,
                    side: str,
                    amount: float,
                    entry_price: float) -> PositionInfo:
        """포지션 추가"""
        key = f"{exchange}_{symbol}"
        
        position = PositionInfo(
            exchange=exchange,
            symbol=symbol,
            side=side,
            amount=amount,
            entry_price=entry_price,
            entry_time=datetime.now(),
            current_price=entry_price,
            high_price=entry_price,
            low_price=entry_price
        )
        
        self.positions[key] = position
        logger.info(f"Position added: {key} - {side} {amount} @ {entry_price}")
        
        return position
    
    def update_position(self, exchange: str, symbol: str, current_price: float):
        """포지션 업데이트"""
        key = f"{exchange}_{symbol}"
        
        if key not in self.positions:
            return
        
        position = self.positions[key]
        position.current_price = current_price
        
        # High/Low 업데이트
        position.high_price = max(position.high_price, current_price)
        position.low_price = min(position.low_price, current_price)
        
        # PnL 계산
        if position.side == 'long':
            position.pnl = (current_price - position.entry_price) * position.amount
        else:  # short
            position.pnl = (position.entry_price - current_price) * position.amount
        
        # PnL 퍼센트
        if position.entry_price > 0:
            position.pnl_pct = (position.pnl / (position.entry_price * position.amount)) * 100
        
        # 보유 시간
        position.duration_hours = (datetime.now() - position.entry_time).total_seconds() / 3600
        
        # 미실현 손익 업데이트
        self._update_unrealized_pnl()
    
    def close_position(self, exchange: str, symbol: str, exit_price: float) -> Optional[float]:
        """포지션 청산"""
        key = f"{exchange}_{symbol}"
        
        if key not in self.positions:
            return None
        
        position = self.positions[key]
        
        # 최종 PnL 계산
        if position.side == 'long':
            final_pnl = (exit_price - position.entry_price) * position.amount
        else:  # short
            final_pnl = (position.entry_price - exit_price) * position.amount
        
        # 실현 손익 업데이트
        self.realized_pnl += final_pnl
        
        # 히스토리에 추가
        self.position_history.append({
            'exchange': position.exchange,
            'symbol': position.symbol,
            'side': position.side,
            'amount': position.amount,
            'entry_price': position.entry_price,
            'exit_price': exit_price,
            'entry_time': position.entry_time,
            'exit_time': datetime.now(),
            'pnl': final_pnl,
            'pnl_pct': (final_pnl / (position.entry_price * position.amount)) * 100,
            'duration_hours': position.duration_hours
        })
        
        # 포지션 제거
        del self.positions[key]
        
        # 미실현 손익 업데이트
        self._update_unrealized_pnl()
        
        logger.info(f"Position closed: {key} - PnL: {final_pnl:,.0f}")
        return final_pnl
    
    def _update_unrealized_pnl(self):
        """미실현 손익 업데이트"""
        self.unrealized_pnl = sum(pos.pnl for pos in self.positions.values())
        self.total_pnl = self.realized_pnl + self.unrealized_pnl
    
    def get_position(self, exchange: str, symbol: str) -> Optional[PositionInfo]:
        """포지션 조회"""
        key = f"{exchange}_{symbol}"
        return self.positions.get(key)
    
    def get_all_positions(self) -> Dict[str, PositionInfo]:
        """모든 포지션 조회"""
        return self.positions.copy()
    
    def get_statistics(self) -> Dict:
        """통계 정보"""
        total_positions = len(self.positions)
        long_positions = sum(1 for p in self.positions.values() if p.side == 'long')
        short_positions = total_positions - long_positions
        
        # 히스토리 통계
        total_trades = len(self.position_history)
        winning_trades = sum(1 for h in self.position_history if h['pnl'] > 0)
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # 평균 보유 시간
        avg_duration = (
            sum(h['duration_hours'] for h in self.position_history) / total_trades
            if total_trades > 0 else 0
        )
        
        return {
            'open_positions': total_positions,
            'long_positions': long_positions,
            'short_positions': short_positions,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate': win_rate,
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': self.unrealized_pnl,
            'total_pnl': self.total_pnl,
            'avg_duration_hours': avg_duration
        }
    
    def get_risk_metrics(self) -> Dict:
        """리스크 메트릭"""
        if not self.positions:
            return {
                'max_drawdown': 0,
                'exposure': 0,
                'position_concentration': 0
            }
        
        # 최대 손실
        max_loss = min(pos.pnl for pos in self.positions.values()) if self.positions else 0
        
        # 노출도 (총 포지션 가치)
        total_exposure = sum(
            pos.amount * pos.current_price 
            for pos in self.positions.values()
        )
        
        # 포지션 집중도
        position_sizes = [pos.amount * pos.current_price for pos in self.positions.values()]
        max_position = max(position_sizes) if position_sizes else 0
        concentration = (max_position / total_exposure * 100) if total_exposure > 0 else 0
        
        return {
            'max_drawdown': max_loss,
            'exposure': total_exposure,
            'position_concentration': concentration
        }
    
    def reset(self):
        """초기화"""
        self.positions.clear()
        self.position_history.clear()
        self.total_pnl = 0
        self.realized_pnl = 0
        self.unrealized_pnl = 0
        logger.info("PositionTracker reset")