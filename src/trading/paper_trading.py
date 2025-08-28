"""
Paper Trading System
시뮬레이션 거래로 실제 자금 없이 전략 테스트
"""

import asyncio
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
import uuid

from src.utils.logger import LoggerManager

logger = LoggerManager(__name__)


class OrderStatus(Enum):
    """주문 상태"""
    PENDING = "pending"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class OrderType(Enum):
    """주문 유형"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """주문 방향"""
    BUY = "buy"
    SELL = "sell"
    LONG = "long"   # 선물 롱
    SHORT = "short"  # 선물 숏


@dataclass
class PaperOrder:
    """가상 주문"""
    order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    exchange: str = ""
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    order_type: OrderType = OrderType.MARKET
    amount: float = 0
    price: float = 0
    executed_amount: float = 0
    executed_price: float = 0
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    filled_at: Optional[datetime] = None
    fee: float = 0
    fee_currency: str = ""
    
    def to_dict(self) -> Dict:
        """딕셔너리 변환"""
        data = asdict(self)
        data['side'] = self.side.value
        data['order_type'] = self.order_type.value
        data['status'] = self.status.value
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        data['filled_at'] = self.filled_at.isoformat() if self.filled_at else None
        return data


@dataclass
class PaperPosition:
    """가상 포지션"""
    position_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    exchange: str = ""
    symbol: str = ""
    side: str = ""  # long/short/buy/sell
    amount: float = 0
    entry_price: float = 0
    current_price: float = 0
    realized_pnl: float = 0
    unrealized_pnl: float = 0
    margin: float = 0  # 선물용
    leverage: float = 1  # 선물용
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    closed_at: Optional[datetime] = None
    is_open: bool = True
    
    def calculate_pnl(self, current_price: float) -> float:
        """PnL 계산"""
        self.current_price = current_price
        
        if self.side in ["buy", "long"]:
            self.unrealized_pnl = (current_price - self.entry_price) * self.amount
        else:  # sell, short
            self.unrealized_pnl = (self.entry_price - current_price) * self.amount
        
        return self.unrealized_pnl
    
    def to_dict(self) -> Dict:
        """딕셔너리 변환"""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        data['closed_at'] = self.closed_at.isoformat() if self.closed_at else None
        return data


class PaperTradingEngine:
    """Paper Trading 엔진"""
    
    def __init__(self, initial_balance: Dict[str, float], fees: Dict[str, float] = None):
        """
        초기화
        
        Args:
            initial_balance: 초기 잔고 {'upbit': {'KRW': 20000000}, 'binance': {'USDT': 14000}}
            fees: 거래 수수료 {'upbit': 0.0005, 'binance': 0.0004}
        """
        self.initial_balance = initial_balance
        self.balance = json.loads(json.dumps(initial_balance))  # Deep copy
        self.fees = fees or {'upbit': 0.0005, 'binance': 0.0004}
        
        # 주문 및 포지션
        self.orders: List[PaperOrder] = []
        self.positions: Dict[str, PaperPosition] = {}
        self.order_history: List[PaperOrder] = []
        self.position_history: List[PaperPosition] = []
        
        # 가격 피드
        self.current_prices = {}
        self.orderbooks = {}
        
        # 통계
        self.stats = {
            'total_orders': 0,
            'filled_orders': 0,
            'cancelled_orders': 0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0,
            'total_fees': 0,
            'max_drawdown': 0,
            'peak_balance': sum(initial_balance.get('upbit', {}).values()) + 
                           sum(initial_balance.get('binance', {}).values())
        }
        
        logger.info("Paper Trading Engine initialized")
        logger.info(f"Initial balance: {self.balance}")
    
    def update_price(self, exchange: str, symbol: str, price: float):
        """가격 업데이트"""
        key = f"{exchange}:{symbol}"
        self.current_prices[key] = price
        
        # 포지션 PnL 업데이트
        for pos_id, position in self.positions.items():
            if position.exchange == exchange and position.symbol == symbol and position.is_open:
                position.calculate_pnl(price)
    
    def update_orderbook(self, exchange: str, symbol: str, orderbook: Dict):
        """오더북 업데이트"""
        key = f"{exchange}:{symbol}"
        self.orderbooks[key] = orderbook
    
    async def place_order(self, exchange: str, symbol: str, side: OrderSide, 
                          amount: float, order_type: OrderType = OrderType.MARKET,
                          price: float = None) -> PaperOrder:
        """
        주문 실행
        
        Args:
            exchange: 거래소
            symbol: 심볼
            side: 매수/매도
            amount: 수량
            order_type: 주문 유형
            price: 가격 (지정가 주문시)
        
        Returns:
            실행된 주문
        """
        # 주문 생성
        order = PaperOrder(
            exchange=exchange,
            symbol=symbol,
            side=side,
            order_type=order_type,
            amount=amount,
            price=price or 0
        )
        
        # 잔고 확인
        if not self._check_balance(order):
            order.status = OrderStatus.REJECTED
            logger.warning(f"Order rejected due to insufficient balance: {order.order_id}")
            self.order_history.append(order)
            return order
        
        # 주문 추가
        self.orders.append(order)
        self.stats['total_orders'] += 1
        
        # 시장가 주문은 즉시 체결
        if order_type == OrderType.MARKET:
            await self._fill_market_order(order)
        else:
            # 지정가 주문은 대기 상태로
            logger.info(f"Limit order placed: {order.order_id}")
        
        return order
    
    def _check_balance(self, order: PaperOrder) -> bool:
        """잔고 확인"""
        exchange = order.exchange
        
        if exchange not in self.balance:
            return False
        
        # 현물 거래
        if order.exchange == 'upbit':
            if order.side == OrderSide.BUY:
                # KRW 잔고 확인
                required = order.amount * order.price if order.price else \
                          order.amount * self._get_current_price(exchange, order.symbol)
                required *= (1 + self.fees[exchange])  # 수수료 포함
                
                return self.balance[exchange].get('KRW', 0) >= required
            else:  # SELL
                # BTC 잔고 확인
                asset = order.symbol.split('/')[0]
                return self.balance[exchange].get(asset, 0) >= order.amount
        
        # 선물 거래 (바이낸스)
        elif order.exchange == 'binance':
            if order.side in [OrderSide.LONG, OrderSide.SHORT]:
                # USDT 마진 확인
                required_margin = order.amount * self._get_current_price(exchange, order.symbol) / 10  # 10x 레버리지 가정
                return self.balance[exchange].get('USDT', 0) >= required_margin
            else:
                # 현물 거래
                if order.side == OrderSide.BUY:
                    required = order.amount * self._get_current_price(exchange, order.symbol)
                    required *= (1 + self.fees[exchange])
                    return self.balance[exchange].get('USDT', 0) >= required
                else:
                    asset = order.symbol.split('/')[0]
                    return self.balance[exchange].get(asset, 0) >= order.amount
        
        return False
    
    async def _fill_market_order(self, order: PaperOrder):
        """시장가 주문 체결"""
        price = self._get_current_price(order.exchange, order.symbol)
        if not price:
            order.status = OrderStatus.REJECTED
            logger.error(f"No price available for {order.symbol} on {order.exchange}")
            return
        
        # 체결 처리
        order.executed_amount = order.amount
        order.executed_price = price
        order.status = OrderStatus.FILLED
        order.filled_at = datetime.now()
        
        # 수수료 계산
        fee_rate = self.fees.get(order.exchange, 0.001)
        order.fee = order.executed_amount * order.executed_price * fee_rate
        
        if order.exchange == 'upbit':
            order.fee_currency = 'KRW'
        else:
            order.fee_currency = 'USDT'
        
        # 잔고 업데이트
        self._update_balance(order)
        
        # 포지션 업데이트
        self._update_position(order)
        
        # 통계 업데이트
        self.stats['filled_orders'] += 1
        self.stats['total_fees'] += order.fee
        
        # 기록
        self.order_history.append(order)
        self.orders.remove(order)
        
        logger.info(f"Order filled: {order.order_id} - {order.side.value} {order.executed_amount} "
                   f"{order.symbol} @ {order.executed_price}")
    
    def _update_balance(self, order: PaperOrder):
        """잔고 업데이트"""
        exchange = order.exchange
        
        if order.exchange == 'upbit':
            if order.side == OrderSide.BUY:
                # KRW 차감, BTC 증가
                self.balance[exchange]['KRW'] -= (order.executed_amount * order.executed_price + order.fee)
                asset = order.symbol.split('/')[0]
                self.balance[exchange][asset] = self.balance[exchange].get(asset, 0) + order.executed_amount
            else:  # SELL
                # BTC 차감, KRW 증가
                asset = order.symbol.split('/')[0]
                self.balance[exchange][asset] -= order.executed_amount
                self.balance[exchange]['KRW'] += (order.executed_amount * order.executed_price - order.fee)
        
        elif order.exchange == 'binance':
            if order.side in [OrderSide.BUY, OrderSide.LONG]:
                self.balance[exchange]['USDT'] -= (order.executed_amount * order.executed_price + order.fee)
                if order.side == OrderSide.BUY:
                    asset = order.symbol.split('/')[0]
                    self.balance[exchange][asset] = self.balance[exchange].get(asset, 0) + order.executed_amount
            else:  # SELL, SHORT
                if order.side == OrderSide.SELL:
                    asset = order.symbol.split('/')[0]
                    self.balance[exchange][asset] -= order.executed_amount
                    self.balance[exchange]['USDT'] += (order.executed_amount * order.executed_price - order.fee)
    
    def _update_position(self, order: PaperOrder):
        """포지션 업데이트"""
        position_key = f"{order.exchange}:{order.symbol}"
        
        if order.side in [OrderSide.BUY, OrderSide.LONG]:
            if position_key in self.positions and self.positions[position_key].is_open:
                # 기존 포지션에 추가
                position = self.positions[position_key]
                total_amount = position.amount + order.executed_amount
                position.entry_price = (position.entry_price * position.amount + 
                                       order.executed_price * order.executed_amount) / total_amount
                position.amount = total_amount
                position.updated_at = datetime.now()
            else:
                # 새 포지션 생성
                position = PaperPosition(
                    exchange=order.exchange,
                    symbol=order.symbol,
                    side=order.side.value,
                    amount=order.executed_amount,
                    entry_price=order.executed_price
                )
                self.positions[position_key] = position
        
        elif order.side in [OrderSide.SELL, OrderSide.SHORT]:
            if position_key in self.positions and self.positions[position_key].is_open:
                # 포지션 청산
                position = self.positions[position_key]
                
                if position.amount <= order.executed_amount:
                    # 전체 청산
                    position.realized_pnl = position.calculate_pnl(order.executed_price)
                    position.is_open = False
                    position.closed_at = datetime.now()
                    self.position_history.append(position)
                    del self.positions[position_key]
                    
                    # 통계 업데이트
                    self.stats['total_trades'] += 1
                    self.stats['total_pnl'] += position.realized_pnl
                    if position.realized_pnl > 0:
                        self.stats['winning_trades'] += 1
                    else:
                        self.stats['losing_trades'] += 1
                else:
                    # 부분 청산
                    closed_amount = order.executed_amount
                    remaining_amount = position.amount - closed_amount
                    
                    # 실현 손익 계산
                    partial_pnl = position.calculate_pnl(order.executed_price) * (closed_amount / position.amount)
                    position.realized_pnl += partial_pnl
                    position.amount = remaining_amount
                    position.updated_at = datetime.now()
                    
                    self.stats['total_pnl'] += partial_pnl
    
    def _get_current_price(self, exchange: str, symbol: str) -> Optional[float]:
        """현재가 조회"""
        key = f"{exchange}:{symbol}"
        return self.current_prices.get(key)
    
    async def cancel_order(self, order_id: str) -> bool:
        """주문 취소"""
        for order in self.orders:
            if order.order_id == order_id and order.status == OrderStatus.PENDING:
                order.status = OrderStatus.CANCELLED
                order.updated_at = datetime.now()
                self.order_history.append(order)
                self.orders.remove(order)
                self.stats['cancelled_orders'] += 1
                logger.info(f"Order cancelled: {order_id}")
                return True
        return False
    
    def get_balance(self) -> Dict[str, Dict[str, float]]:
        """현재 잔고 조회"""
        return self.balance
    
    def get_positions(self) -> Dict[str, PaperPosition]:
        """현재 포지션 조회"""
        return self.positions
    
    def get_open_orders(self) -> List[PaperOrder]:
        """미체결 주문 조회"""
        return self.orders
    
    def get_portfolio_value(self) -> float:
        """포트폴리오 가치 계산"""
        total_value = 0
        
        # 현금 잔고
        if 'upbit' in self.balance:
            total_value += self.balance['upbit'].get('KRW', 0)
            # BTC를 KRW로 환산
            for asset, amount in self.balance['upbit'].items():
                if asset != 'KRW' and amount > 0:
                    price = self._get_current_price('upbit', f'{asset}/KRW')
                    if price:
                        total_value += amount * price
        
        if 'binance' in self.balance:
            # USDT를 KRW로 환산 (1 USDT = 1400 KRW 가정)
            usdt_to_krw = 1400
            total_value += self.balance['binance'].get('USDT', 0) * usdt_to_krw
            # BTC를 USDT로 환산 후 KRW로
            for asset, amount in self.balance['binance'].items():
                if asset != 'USDT' and amount > 0:
                    price = self._get_current_price('binance', f'{asset}/USDT')
                    if price:
                        total_value += amount * price * usdt_to_krw
        
        # 미실현 손익
        for position in self.positions.values():
            if position.is_open:
                total_value += position.unrealized_pnl
        
        # 최대 낙폭 업데이트
        if total_value > self.stats['peak_balance']:
            self.stats['peak_balance'] = total_value
        else:
            drawdown = (self.stats['peak_balance'] - total_value) / self.stats['peak_balance']
            self.stats['max_drawdown'] = max(self.stats['max_drawdown'], drawdown)
        
        return total_value
    
    def get_statistics(self) -> Dict[str, Any]:
        """통계 조회"""
        stats = self.stats.copy()
        
        # 승률 계산
        if stats['total_trades'] > 0:
            stats['win_rate'] = stats['winning_trades'] / stats['total_trades']
        else:
            stats['win_rate'] = 0
        
        # 현재 포트폴리오 가치
        stats['portfolio_value'] = self.get_portfolio_value()
        
        # 초기 대비 수익률
        initial_value = self.stats['peak_balance']  # 초기값으로 사용
        stats['total_return'] = (stats['portfolio_value'] - initial_value) / initial_value if initial_value > 0 else 0
        
        return stats
    
    def reset(self):
        """엔진 리셋"""
        self.balance = json.loads(json.dumps(self.initial_balance))
        self.orders.clear()
        self.positions.clear()
        self.order_history.clear()
        self.position_history.clear()
        self.current_prices.clear()
        self.orderbooks.clear()
        
        # 통계 리셋
        initial_value = sum(self.initial_balance.get('upbit', {}).values()) + \
                       sum(self.initial_balance.get('binance', {}).values())
        self.stats = {
            'total_orders': 0,
            'filled_orders': 0,
            'cancelled_orders': 0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0,
            'total_fees': 0,
            'max_drawdown': 0,
            'peak_balance': initial_value
        }
        
        logger.info("Paper Trading Engine reset")
    
    def export_history(self, filepath: str):
        """거래 기록 내보내기"""
        history = {
            'orders': [order.to_dict() for order in self.order_history],
            'positions': [pos.to_dict() for pos in self.position_history],
            'statistics': self.get_statistics()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Trading history exported to {filepath}")


class PaperTradingManager:
    """Paper Trading 매니저"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        초기화
        
        Args:
            config: Paper Trading 설정
        """
        self.config = config
        self.engine = PaperTradingEngine(
            initial_balance=config.get('initial_balance', {
                'upbit': {'KRW': 20000000},
                'binance': {'USDT': 14000}
            }),
            fees=config.get('fees', {
                'upbit': 0.0005,
                'binance': 0.0004
            })
        )
        
        self.is_running = False
        self.update_task = None
        
        logger.info("Paper Trading Manager initialized")
    
    async def start(self):
        """Paper Trading 시작"""
        if self.is_running:
            logger.warning("Paper Trading already running")
            return
        
        self.is_running = True
        self.update_task = asyncio.create_task(self._update_loop())
        logger.info("Paper Trading started")
    
    async def stop(self):
        """Paper Trading 중지"""
        self.is_running = False
        
        if self.update_task:
            self.update_task.cancel()
            try:
                await self.update_task
            except asyncio.CancelledError:
                pass
        
        # 결과 저장
        self.engine.export_history(f"paper_trading_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        logger.info("Paper Trading stopped")
    
    async def _update_loop(self):
        """업데이트 루프"""
        while self.is_running:
            try:
                # 지정가 주문 체결 확인
                await self._check_pending_orders()
                
                # 통계 로깅
                if asyncio.get_event_loop().time() % 60 == 0:  # 1분마다
                    stats = self.engine.get_statistics()
                    logger.info(f"Paper Trading Stats: {stats}")
                
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in update loop: {e}")
    
    async def _check_pending_orders(self):
        """미체결 주문 체결 확인"""
        for order in list(self.engine.orders):
            if order.status == OrderStatus.PENDING:
                current_price = self.engine._get_current_price(order.exchange, order.symbol)
                
                if current_price:
                    # 지정가 주문 체결 조건 확인
                    if order.order_type == OrderType.LIMIT:
                        if (order.side in [OrderSide.BUY, OrderSide.LONG] and current_price <= order.price) or \
                           (order.side in [OrderSide.SELL, OrderSide.SHORT] and current_price >= order.price):
                            await self.engine._fill_market_order(order)
    
    def on_price_update(self, exchange: str, symbol: str, price: float):
        """가격 업데이트 콜백"""
        self.engine.update_price(exchange, symbol, price)
    
    def on_orderbook_update(self, exchange: str, symbol: str, orderbook: Dict):
        """오더북 업데이트 콜백"""
        self.engine.update_orderbook(exchange, symbol, orderbook)
    
    async def execute_signal(self, signal: Dict[str, Any]) -> Optional[PaperOrder]:
        """
        거래 신호 실행
        
        Args:
            signal: 거래 신호
                - exchange: 거래소
                - symbol: 심볼
                - side: buy/sell/long/short
                - amount: 수량
                - order_type: market/limit
                - price: 가격 (지정가시)
        
        Returns:
            실행된 주문
        """
        try:
            order = await self.engine.place_order(
                exchange=signal['exchange'],
                symbol=signal['symbol'],
                side=OrderSide(signal['side']),
                amount=signal['amount'],
                order_type=OrderType(signal.get('order_type', 'market')),
                price=signal.get('price')
            )
            
            return order
            
        except Exception as e:
            logger.error(f"Failed to execute signal: {e}")
            return None


def create_paper_trading_manager(config: Dict[str, Any]) -> PaperTradingManager:
    """Paper Trading Manager 생성"""
    return PaperTradingManager(config)


if __name__ == "__main__":
    # Paper Trading 테스트
    async def test_paper_trading():
        config = {
            'initial_balance': {
                'upbit': {'KRW': 20000000},
                'binance': {'USDT': 14000}
            }
        }
        
        manager = create_paper_trading_manager(config)
        await manager.start()
        
        # 가격 업데이트
        manager.on_price_update('upbit', 'BTC/KRW', 100000000)
        manager.on_price_update('binance', 'BTC/USDT', 70000)
        
        # 테스트 주문
        signal = {
            'exchange': 'upbit',
            'symbol': 'BTC/KRW',
            'side': 'buy',
            'amount': 0.01,
            'order_type': 'market'
        }
        
        order = await manager.execute_signal(signal)
        print(f"Order executed: {order}")
        
        # 통계 확인
        stats = manager.engine.get_statistics()
        print(f"Statistics: {stats}")
        
        await manager.stop()
    
    asyncio.run(test_paper_trading())