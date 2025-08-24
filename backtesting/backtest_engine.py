"""
Backtest Engine for Kimchi Premium Arbitrage
백테스팅 시뮬레이션 핵심 엔진
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class OrderSide(Enum):
    """주문 방향"""
    BUY = "buy"
    SELL = "sell"


class PositionSide(Enum):
    """포지션 방향"""
    LONG = "long"
    SHORT = "short"


@dataclass
class Trade:
    """거래 기록"""
    timestamp: datetime
    exchange: str
    symbol: str
    side: OrderSide
    price: float
    amount: float
    fee: float
    position_side: PositionSide
    
    @property
    def value(self) -> float:
        """거래 금액"""
        return self.price * self.amount
    
    @property
    def cost(self) -> float:
        """총 비용 (수수료 포함)"""
        return self.value + self.fee


@dataclass
class Position:
    """포지션 정보"""
    exchange: str
    symbol: str
    side: PositionSide
    amount: float
    entry_price: float
    current_price: float
    opened_at: datetime
    trades: List[Trade] = field(default_factory=list)
    
    @property
    def value(self) -> float:
        """현재 가치"""
        return self.current_price * self.amount
    
    @property
    def pnl(self) -> float:
        """미실현 손익"""
        if self.side == PositionSide.LONG:
            return (self.current_price - self.entry_price) * self.amount
        else:  # SHORT
            return (self.entry_price - self.current_price) * self.amount
    
    @property
    def pnl_pct(self) -> float:
        """손익률 (%)"""
        if self.entry_price == 0:
            return 0
        return self.pnl / (self.entry_price * self.amount) * 100


@dataclass
class Portfolio:
    """포트폴리오 상태"""
    timestamp: datetime
    cash: Dict[str, float]  # {'KRW': 20000000, 'USD': 15000}
    positions: Dict[str, Position]  # {'upbit_BTC': Position, 'binance_BTC': Position}
    total_value: float
    realized_pnl: float
    unrealized_pnl: float
    
    @property
    def total_pnl(self) -> float:
        """총 손익"""
        return self.realized_pnl + self.unrealized_pnl


class BacktestEngine:
    """
    백테스팅 엔진
    - 시뮬레이션 시간 관리
    - 포지션 추적
    - 손익 계산
    - 거래 비용 적용
    """
    
    def __init__(self, initial_capital: Dict[str, float], 
                 fee_rate: float = 0.001):
        """
        Args:
            initial_capital: 초기 자본 {'KRW': 20000000, 'USD': 15000}
            fee_rate: 거래 수수료율 (0.1%)
        """
        self.initial_capital = initial_capital.copy()
        self.cash = initial_capital.copy()
        self.fee_rate = fee_rate
        
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.portfolio_history: List[Portfolio] = []
        
        self.current_time: Optional[datetime] = None
        self.current_prices: Dict[str, float] = {}
        
        # 통계
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_fees = 0
        self.max_drawdown = 0
        self.peak_value = sum(initial_capital.values())
        
    def update_time(self, timestamp: datetime, prices: Dict[str, float]):
        """
        시뮬레이션 시간 업데이트
        
        Args:
            timestamp: 현재 시간
            prices: 현재 가격 {'upbit_BTC': 150000000, 'binance_BTC': 100000}
        """
        self.current_time = timestamp
        self.current_prices = prices
        
        # 포지션 현재가 업데이트
        for pos_id, position in self.positions.items():
            if pos_id in prices:
                position.current_price = prices[pos_id]
    
    def open_position(self, exchange: str, symbol: str, side: PositionSide,
                     amount: float, price: Optional[float] = None) -> bool:
        """
        포지션 오픈
        
        Args:
            exchange: 거래소
            symbol: 심볼
            side: 포지션 방향
            amount: 수량
            price: 가격 (None이면 현재가 사용)
            
        Returns:
            성공 여부
        """
        pos_id = f"{exchange}_{symbol}"
        
        # 현재가 사용
        if price is None:
            if pos_id not in self.current_prices:
                logger.error(f"No price for {pos_id}")
                return False
            price = self.current_prices[pos_id]
        
        # 자본금 확인
        currency = 'KRW' if exchange == 'upbit' else 'USD'
        required_capital = price * amount
        fee = required_capital * self.fee_rate
        
        if self.cash.get(currency, 0) < required_capital + fee:
            logger.warning(f"Insufficient capital for {pos_id}: required={required_capital + fee}, available={self.cash.get(currency, 0)}")
            return False
        
        # 거래 기록
        order_side = OrderSide.BUY if side == PositionSide.LONG else OrderSide.SELL
        trade = Trade(
            timestamp=self.current_time,
            exchange=exchange,
            symbol=symbol,
            side=order_side,
            price=price,
            amount=amount,
            fee=fee,
            position_side=side
        )
        self.trades.append(trade)
        
        # 자본금 차감
        self.cash[currency] -= (required_capital + fee)
        self.total_fees += fee
        
        # 포지션 생성 또는 업데이트
        if pos_id in self.positions:
            # 기존 포지션에 추가
            position = self.positions[pos_id]
            total_amount = position.amount + amount
            position.entry_price = (position.entry_price * position.amount + price * amount) / total_amount
            position.amount = total_amount
            position.trades.append(trade)
        else:
            # 새 포지션 생성
            self.positions[pos_id] = Position(
                exchange=exchange,
                symbol=symbol,
                side=side,
                amount=amount,
                entry_price=price,
                current_price=price,
                opened_at=self.current_time,
                trades=[trade]
            )
        
        self.total_trades += 1
        logger.debug(f"Opened {side.value} position: {pos_id} @ {price}, amount={amount}")
        return True
    
    def close_position(self, exchange: str, symbol: str, 
                      amount: Optional[float] = None,
                      price: Optional[float] = None) -> float:
        """
        포지션 종료
        
        Args:
            exchange: 거래소
            symbol: 심볼
            amount: 종료할 수량 (None이면 전체)
            price: 가격 (None이면 현재가 사용)
            
        Returns:
            실현 손익
        """
        pos_id = f"{exchange}_{symbol}"
        
        if pos_id not in self.positions:
            logger.warning(f"No position to close: {pos_id}")
            return 0
        
        position = self.positions[pos_id]
        
        # 현재가 사용
        if price is None:
            if pos_id not in self.current_prices:
                logger.error(f"No price for {pos_id}")
                return 0
            price = self.current_prices[pos_id]
        
        # 종료 수량
        if amount is None or amount > position.amount:
            amount = position.amount
        
        # 손익 계산
        if position.side == PositionSide.LONG:
            pnl = (price - position.entry_price) * amount
        else:  # SHORT
            pnl = (position.entry_price - price) * amount
        
        # 거래 비용
        currency = 'KRW' if exchange == 'upbit' else 'USD'
        trade_value = price * amount
        fee = trade_value * self.fee_rate
        
        # 거래 기록
        order_side = OrderSide.SELL if position.side == PositionSide.LONG else OrderSide.BUY
        trade = Trade(
            timestamp=self.current_time,
            exchange=exchange,
            symbol=symbol,
            side=order_side,
            price=price,
            amount=amount,
            fee=fee,
            position_side=position.side
        )
        self.trades.append(trade)
        
        # 자본금 추가 (수수료 차감)
        self.cash[currency] += (trade_value - fee)
        self.total_fees += fee
        
        # 포지션 업데이트
        position.amount -= amount
        position.trades.append(trade)
        
        if position.amount <= 0:
            # 포지션 완전 종료
            del self.positions[pos_id]
        
        # 통계 업데이트
        self.total_trades += 1
        if pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        
        logger.debug(f"Closed position: {pos_id} @ {price}, amount={amount}, PnL={pnl}")
        return pnl - fee  # 수수료 차감한 실현 손익
    
    def get_portfolio_value(self) -> float:
        """포트폴리오 총 가치 계산"""
        # 현금 가치 (USD는 환율 적용)
        cash_value = self.cash.get('KRW', 0) + self.cash.get('USD', 0) * 1350
        
        # 포지션 가치
        position_value = 0
        for pos_id, position in self.positions.items():
            if 'upbit' in pos_id:
                position_value += position.value
            else:  # binance
                position_value += position.value * 1350
        
        return cash_value + position_value
    
    def record_portfolio(self):
        """현재 포트폴리오 상태 기록"""
        # 미실현 손익 계산
        unrealized_pnl = sum(pos.pnl for pos in self.positions.values())
        
        # 실현 손익 계산 (거래 기록에서)
        realized_pnl = 0
        for trade in self.trades:
            if trade.side == OrderSide.SELL and trade.position_side == PositionSide.LONG:
                # 롱 포지션 청산
                entry_trades = [t for t in self.trades 
                              if t.exchange == trade.exchange 
                              and t.symbol == trade.symbol
                              and t.side == OrderSide.BUY
                              and t.timestamp < trade.timestamp]
                if entry_trades:
                    avg_entry = sum(t.price * t.amount for t in entry_trades) / sum(t.amount for t in entry_trades)
                    realized_pnl += (trade.price - avg_entry) * trade.amount
        
        # 포트폴리오 기록
        portfolio = Portfolio(
            timestamp=self.current_time,
            cash=self.cash.copy(),
            positions=self.positions.copy(),
            total_value=self.get_portfolio_value(),
            realized_pnl=realized_pnl,
            unrealized_pnl=unrealized_pnl
        )
        self.portfolio_history.append(portfolio)
        
        # Drawdown 계산
        current_value = portfolio.total_value
        if current_value > self.peak_value:
            self.peak_value = current_value
        else:
            drawdown = (self.peak_value - current_value) / self.peak_value
            self.max_drawdown = max(self.max_drawdown, drawdown)
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """성과 지표 계산"""
        if not self.portfolio_history:
            return {}
        
        # 수익률 계산
        initial_value = sum(self.initial_capital.values())
        final_value = self.get_portfolio_value()
        total_return = (final_value - initial_value) / initial_value
        
        # 일일 수익률
        returns = []
        for i in range(1, len(self.portfolio_history)):
            prev_value = self.portfolio_history[i-1].total_value
            curr_value = self.portfolio_history[i].total_value
            if prev_value > 0:
                returns.append((curr_value - prev_value) / prev_value)
        
        if returns:
            returns_array = np.array(returns)
            
            # Sharpe Ratio (연율화, 무위험 수익률 0 가정)
            sharpe_ratio = np.mean(returns_array) / np.std(returns_array) * np.sqrt(252) if np.std(returns_array) > 0 else 0
            
            # Calmar Ratio
            annual_return = total_return * 365 / max(1, (self.current_time - self.portfolio_history[0].timestamp).days)
            calmar_ratio = annual_return / self.max_drawdown if self.max_drawdown > 0 else 0
        else:
            sharpe_ratio = 0
            calmar_ratio = 0
        
        # Win Rate
        win_rate = self.winning_trades / max(1, self.winning_trades + self.losing_trades)
        
        return {
            'total_return': total_return * 100,  # %
            'sharpe_ratio': sharpe_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': self.max_drawdown * 100,  # %
            'win_rate': win_rate * 100,  # %
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'total_fees': self.total_fees,
            'final_value': final_value
        }