"""
Risk Manager for Realtime Trading
실시간 리스크 관리
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RiskLimits:
    """리스크 한도"""
    max_position_size: float = 0.1  # 최대 포지션 크기 (BTC)
    max_daily_loss: float = 1000000  # 일일 최대 손실 (KRW)
    max_drawdown: float = 0.05  # 최대 낙폭 (5%)
    max_exposure: float = 0.5  # 최대 노출도 (자본 대비 50%)
    max_daily_trades: int = 20  # 일일 최대 거래 횟수
    min_trade_interval: int = 60  # 최소 거래 간격 (초)
    max_consecutive_losses: int = 3  # 연속 손실 한도


class RiskManager:
    """
    리스크 관리 시스템
    - 포지션 크기 제한
    - 손실 한도 관리
    - 노출도 관리
    """
    
    def __init__(self, limits: Optional[RiskLimits] = None, capital: float = 40000000):
        """
        Args:
            limits: 리스크 한도 설정
            capital: 총 자본금 (KRW)
        """
        self.limits = limits or RiskLimits()
        self.capital = capital
        
        # 일일 통계
        self.daily_trades = 0
        self.daily_pnl = 0
        self.daily_reset_time = datetime.now().replace(hour=0, minute=0, second=0)
        
        # 거래 기록
        self.trade_history: List[Dict] = []
        self.last_trade_time: Optional[datetime] = None
        self.consecutive_losses = 0
        
        # 최대 낙폭 추적
        self.peak_capital = capital
        self.current_drawdown = 0
        
        logger.info(f"RiskManager initialized with capital: {capital:,.0f}")
    
    def check_position_size(self, amount: float, price: float) -> bool:
        """
        포지션 크기 체크
        
        Args:
            amount: 거래 수량
            price: 가격
            
        Returns:
            허용 여부
        """
        if amount > self.limits.max_position_size:
            logger.warning(f"Position size {amount} exceeds limit {self.limits.max_position_size}")
            return False
        
        # 노출도 체크
        position_value = amount * price
        exposure_ratio = position_value / self.capital
        
        if exposure_ratio > self.limits.max_exposure:
            logger.warning(f"Exposure {exposure_ratio:.2%} exceeds limit {self.limits.max_exposure:.2%}")
            return False
        
        return True
    
    def check_daily_limits(self) -> bool:
        """일일 한도 체크"""
        # 일일 리셋 체크
        self._check_daily_reset()
        
        # 일일 거래 횟수
        if self.daily_trades >= self.limits.max_daily_trades:
            logger.warning(f"Daily trade limit reached: {self.daily_trades}")
            return False
        
        # 일일 손실 한도
        if self.daily_pnl < -self.limits.max_daily_loss:
            logger.warning(f"Daily loss limit reached: {self.daily_pnl:,.0f}")
            return False
        
        return True
    
    def check_trade_interval(self) -> bool:
        """거래 간격 체크"""
        if not self.last_trade_time:
            return True
        
        elapsed = (datetime.now() - self.last_trade_time).total_seconds()
        
        if elapsed < self.limits.min_trade_interval:
            logger.debug(f"Trade interval too short: {elapsed:.0f}s < {self.limits.min_trade_interval}s")
            return False
        
        return True
    
    def check_consecutive_losses(self) -> bool:
        """연속 손실 체크"""
        if self.consecutive_losses >= self.limits.max_consecutive_losses:
            logger.warning(f"Consecutive losses limit reached: {self.consecutive_losses}")
            return False
        
        return True
    
    def check_drawdown(self) -> bool:
        """낙폭 체크"""
        current_capital = self.capital + self.daily_pnl
        
        # Peak 업데이트
        if current_capital > self.peak_capital:
            self.peak_capital = current_capital
        
        # 낙폭 계산
        if self.peak_capital > 0:
            self.current_drawdown = (self.peak_capital - current_capital) / self.peak_capital
        
        if self.current_drawdown > self.limits.max_drawdown:
            logger.warning(f"Drawdown {self.current_drawdown:.2%} exceeds limit {self.limits.max_drawdown:.2%}")
            return False
        
        return True
    
    def can_open_position(self, amount: float, price: float) -> bool:
        """
        포지션 오픈 가능 여부 종합 체크
        
        Args:
            amount: 거래 수량
            price: 가격
            
        Returns:
            허용 여부
        """
        checks = [
            ('Position Size', self.check_position_size(amount, price)),
            ('Daily Limits', self.check_daily_limits()),
            ('Trade Interval', self.check_trade_interval()),
            ('Consecutive Losses', self.check_consecutive_losses()),
            ('Drawdown', self.check_drawdown())
        ]
        
        failed_checks = [name for name, result in checks if not result]
        
        if failed_checks:
            logger.info(f"Risk checks failed: {', '.join(failed_checks)}")
            return False
        
        return True
    
    def record_trade(self, pnl: float, success: bool = True):
        """
        거래 기록
        
        Args:
            pnl: 손익
            success: 성공 여부
        """
        self.daily_trades += 1
        self.daily_pnl += pnl
        self.last_trade_time = datetime.now()
        
        # 연속 손실 업데이트
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        
        # 거래 기록 저장
        self.trade_history.append({
            'timestamp': datetime.now(),
            'pnl': pnl,
            'success': success,
            'daily_trades': self.daily_trades,
            'daily_pnl': self.daily_pnl,
            'consecutive_losses': self.consecutive_losses
        })
        
        logger.info(f"Trade recorded: PnL={pnl:,.0f}, Daily PnL={self.daily_pnl:,.0f}")
    
    def _check_daily_reset(self):
        """일일 리셋 체크"""
        now = datetime.now()
        reset_time = now.replace(hour=0, minute=0, second=0)
        
        if reset_time > self.daily_reset_time:
            self.daily_trades = 0
            self.daily_pnl = 0
            self.daily_reset_time = reset_time
            logger.info("Daily limits reset")
    
    def get_risk_status(self) -> Dict:
        """리스크 상태 조회"""
        self._check_daily_reset()
        
        return {
            'daily_trades': self.daily_trades,
            'daily_trades_remaining': self.limits.max_daily_trades - self.daily_trades,
            'daily_pnl': self.daily_pnl,
            'daily_loss_remaining': self.limits.max_daily_loss + self.daily_pnl,
            'consecutive_losses': self.consecutive_losses,
            'current_drawdown': self.current_drawdown,
            'can_trade': self.check_daily_limits() and self.check_consecutive_losses() and self.check_drawdown()
        }
    
    def get_statistics(self) -> Dict:
        """통계 정보"""
        if not self.trade_history:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'avg_pnl': 0,
                'max_win': 0,
                'max_loss': 0,
                'total_pnl': 0
            }
        
        winning_trades = sum(1 for t in self.trade_history if t['pnl'] > 0)
        losing_trades = sum(1 for t in self.trade_history if t['pnl'] < 0)
        total_trades = len(self.trade_history)
        
        pnls = [t['pnl'] for t in self.trade_history]
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': (winning_trades / total_trades * 100) if total_trades > 0 else 0,
            'avg_pnl': sum(pnls) / total_trades if total_trades > 0 else 0,
            'max_win': max(pnls) if pnls else 0,
            'max_loss': min(pnls) if pnls else 0,
            'total_pnl': sum(pnls)
        }
    
    def update_limits(self, limits: RiskLimits):
        """리스크 한도 업데이트"""
        self.limits = limits
        logger.info("Risk limits updated")
    
    def reset(self):
        """초기화"""
        self.daily_trades = 0
        self.daily_pnl = 0
        self.consecutive_losses = 0
        self.trade_history.clear()
        self.last_trade_time = None
        self.peak_capital = self.capital
        self.current_drawdown = 0
        logger.info("RiskManager reset")