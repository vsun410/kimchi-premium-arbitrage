"""
Paper Trading System
모의 거래 시스템
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PaperOrder:
    """모의 주문"""
    order_id: str
    timestamp: datetime
    exchange: str
    symbol: str
    side: str  # 'buy' or 'sell'
    order_type: str  # 'market' or 'limit'
    amount: float
    price: Optional[float] = None  # None for market orders
    status: str = 'pending'  # 'pending', 'filled', 'cancelled'
    filled_price: Optional[float] = None
    filled_time: Optional[datetime] = None
    fee: Optional[float] = None


@dataclass
class PaperPosition:
    """모의 포지션"""
    exchange: str
    symbol: str
    side: str  # 'long' or 'short'
    amount: float
    entry_price: float
    entry_time: datetime
    current_price: float = 0
    pnl: float = 0
    pnl_pct: float = 0


@dataclass
class PaperBalance:
    """모의 잔고"""
    currency: str
    total: float
    available: float
    locked: float = 0


class PaperTrader:
    """
    모의 거래 시스템
    - 실제 거래 없이 시뮬레이션
    - 주문 체결 시뮬레이션
    - 손익 추적
    """
    
    def __init__(self, 
                 initial_balance: Dict[str, float],
                 fee_rate: float = 0.001,
                 slippage: float = 0.0005):
        """
        Args:
            initial_balance: 초기 잔고 {'KRW': 20000000, 'USD': 15000}
            fee_rate: 수수료율 (0.1%)
            slippage: 슬리피지 (0.05%)
        """
        self.fee_rate = fee_rate
        self.slippage = slippage
        
        # 잔고 초기화
        self.balances = {}
        for currency, amount in initial_balance.items():
            self.balances[currency] = PaperBalance(
                currency=currency,
                total=amount,
                available=amount
            )
        
        # 주문 및 포지션
        self.orders: List[PaperOrder] = []
        self.positions: Dict[str, PaperPosition] = {}
        
        # 거래 기록
        self.trade_history = []
        self.order_counter = 0
        
        # 통계
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0
        self.total_fees = 0
        
        logger.info(f"PaperTrader initialized with balance: {initial_balance}")
    
    def place_order(self, 
                   exchange: str,
                   symbol: str,
                   side: str,
                   order_type: str,
                   amount: float,
                   price: Optional[float] = None) -> PaperOrder:
        """
        주문 생성
        
        Args:
            exchange: 거래소 ('upbit' or 'binance')
            symbol: 심볼 ('BTC/KRW' or 'BTC/USDT')
            side: 매수/매도 ('buy' or 'sell')
            order_type: 주문 타입 ('market' or 'limit')
            amount: 수량
            price: 가격 (limit 주문인 경우)
            
        Returns:
            생성된 주문
        """
        # 주문 ID 생성
        self.order_counter += 1
        order_id = f"{exchange}_{self.order_counter:06d}"
        
        # 주문 생성
        order = PaperOrder(
            order_id=order_id,
            timestamp=datetime.now(),
            exchange=exchange,
            symbol=symbol,
            side=side,
            order_type=order_type,
            amount=amount,
            price=price,
            status='pending'
        )
        
        self.orders.append(order)
        logger.info(f"Order placed: {order_id} - {side} {amount} {symbol} on {exchange}")
        
        return order
    
    def execute_order(self, order: PaperOrder, market_price: float) -> bool:
        """
        주문 체결 시뮬레이션
        
        Args:
            order: 주문
            market_price: 현재 시장가
            
        Returns:
            체결 성공 여부
        """
        if order.status != 'pending':
            return False
        
        # 슬리피지 적용
        if order.side == 'buy':
            execution_price = market_price * (1 + self.slippage)
        else:
            execution_price = market_price * (1 - self.slippage)
        
        # limit 주문 체크
        if order.order_type == 'limit' and order.price:
            if order.side == 'buy' and execution_price > order.price:
                return False  # 매수 가격이 지정가보다 높음
            elif order.side == 'sell' and execution_price < order.price:
                return False  # 매도 가격이 지정가보다 낮음
        
        # 잔고 체크
        total_cost = execution_price * order.amount
        fee = total_cost * self.fee_rate
        
        currency = 'KRW' if order.exchange == 'upbit' else 'USD'
        
        if order.side == 'buy':
            required = total_cost + fee
            if self.balances[currency].available < required:
                logger.warning(f"Insufficient balance for order {order.order_id}")
                order.status = 'cancelled'
                return False
        
        # 주문 체결
        order.status = 'filled'
        order.filled_price = execution_price
        order.filled_time = datetime.now()
        order.fee = fee
        
        # 잔고 업데이트
        if order.side == 'buy':
            self.balances[currency].available -= (total_cost + fee)
            self.balances[currency].total -= (total_cost + fee)
        else:
            self.balances[currency].available += (total_cost - fee)
            self.balances[currency].total += (total_cost - fee)
        
        # 수수료 누적
        self.total_fees += fee
        
        # 거래 기록
        self.trade_history.append({
            'order_id': order.order_id,
            'timestamp': order.filled_time,
            'exchange': order.exchange,
            'symbol': order.symbol,
            'side': order.side,
            'amount': order.amount,
            'price': execution_price,
            'fee': fee
        })
        
        self.total_trades += 1
        
        logger.info(f"Order filled: {order.order_id} at {execution_price}")
        return True
    
    def open_position(self,
                     exchange: str,
                     symbol: str,
                     side: str,
                     amount: float,
                     price: float) -> Optional[PaperPosition]:
        """
        포지션 오픈
        
        Args:
            exchange: 거래소
            symbol: 심볼
            side: 'long' or 'short'
            amount: 수량
            price: 진입 가격
            
        Returns:
            생성된 포지션
        """
        position_key = f"{exchange}_{symbol}"
        
        # 이미 포지션이 있는지 체크
        if position_key in self.positions:
            logger.warning(f"Position already exists: {position_key}")
            return None
        
        # 포지션 생성
        position = PaperPosition(
            exchange=exchange,
            symbol=symbol,
            side=side,
            amount=amount,
            entry_price=price,
            entry_time=datetime.now(),
            current_price=price
        )
        
        self.positions[position_key] = position
        
        logger.info(f"Position opened: {position_key} - {side} {amount} at {price}")
        return position
    
    def close_position(self,
                      exchange: str,
                      symbol: str,
                      price: float) -> Optional[float]:
        """
        포지션 청산
        
        Args:
            exchange: 거래소
            symbol: 심볼
            price: 청산 가격
            
        Returns:
            실현 손익
        """
        position_key = f"{exchange}_{symbol}"
        
        if position_key not in self.positions:
            logger.warning(f"No position found: {position_key}")
            return None
        
        position = self.positions[position_key]
        
        # PnL 계산
        if position.side == 'long':
            pnl = (price - position.entry_price) * position.amount
        else:  # short
            pnl = (position.entry_price - price) * position.amount
        
        # 수수료 차감
        fee = abs(price * position.amount * self.fee_rate)
        pnl -= fee
        
        # 통계 업데이트
        self.total_pnl += pnl
        if pnl > 0:
            self.winning_trades += 1
        
        # 포지션 제거
        del self.positions[position_key]
        
        logger.info(f"Position closed: {position_key} - PnL: {pnl:,.0f}")
        return pnl
    
    def update_positions(self, prices: Dict[str, float]):
        """
        포지션 가격 업데이트
        
        Args:
            prices: {'upbit_BTC': 50000000, 'binance_BTC': 36000}
        """
        for key, position in self.positions.items():
            price_key = f"{position.exchange}_{position.symbol.split('/')[0]}"
            
            if price_key in prices:
                position.current_price = prices[price_key]
                
                # PnL 계산
                if position.side == 'long':
                    position.pnl = (position.current_price - position.entry_price) * position.amount
                else:  # short
                    position.pnl = (position.entry_price - position.current_price) * position.amount
                
                position.pnl_pct = (position.pnl / (position.entry_price * position.amount)) * 100
    
    def get_balance(self, currency: str) -> Optional[PaperBalance]:
        """잔고 조회"""
        return self.balances.get(currency)
    
    def get_position(self, exchange: str, symbol: str) -> Optional[PaperPosition]:
        """포지션 조회"""
        position_key = f"{exchange}_{symbol}"
        return self.positions.get(position_key)
    
    def get_statistics(self) -> Dict:
        """통계 정보 반환"""
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        
        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': win_rate,
            'total_pnl': self.total_pnl,
            'total_fees': self.total_fees,
            'net_pnl': self.total_pnl - self.total_fees,
            'positions': len(self.positions),
            'pending_orders': sum(1 for o in self.orders if o.status == 'pending')
        }
    
    def save_history(self, filepath: str):
        """거래 기록 저장"""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'trade_history': self.trade_history,
            'statistics': self.get_statistics(),
            'final_balances': {
                currency: {
                    'total': balance.total,
                    'available': balance.available
                }
                for currency, balance in self.balances.items()
            }
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Trade history saved to {filepath}")
    
    def reset(self):
        """모든 상태 초기화"""
        # 잔고만 유지하고 나머지 초기화
        for balance in self.balances.values():
            balance.locked = 0
        
        self.orders.clear()
        self.positions.clear()
        self.trade_history.clear()
        
        self.order_counter = 0
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0
        self.total_fees = 0
        
        logger.info("PaperTrader reset")