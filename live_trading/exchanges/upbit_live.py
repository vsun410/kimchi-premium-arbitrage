"""
업비트 실거래 API 구현
pyupbit 라이브러리를 사용한 실제 거래 기능
"""

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
import pyupbit
import pandas as pd

from .base_exchange import (
    BaseExchangeAPI, Order, Balance, Ticker,
    OrderSide, OrderType, OrderStatus
)

logger = logging.getLogger(__name__)


class UpbitLiveAPI(BaseExchangeAPI):
    """
    업비트 실거래 API
    
    주의사항:
    - 실제 자금이 사용되므로 매우 신중히 사용
    - 모든 주문은 로깅됨
    - 테스트넷이 없으므로 소액으로 테스트 필요
    """
    
    def __init__(self, api_key: str, secret_key: str, testnet: bool = False):
        """
        업비트 API 초기화
        
        Args:
            api_key: 업비트 API 키
            secret_key: 업비트 Secret 키
            testnet: 미지원 (업비트는 테스트넷 없음)
        """
        super().__init__(api_key, secret_key, testnet)
        
        if testnet:
            logger.warning("Upbit does not support testnet. Using real API with caution.")
        
        # pyupbit 객체 생성
        self.upbit = pyupbit.Upbit(api_key, secret_key)
        self.connected = False
        
        # 최소 주문 금액 (KRW)
        self.MIN_ORDER_AMOUNT_KRW = 5000
        
        # 최소 주문 수량 (BTC)
        self.MIN_ORDER_AMOUNT_BTC = 0.0001
        
        logger.info("UpbitLiveAPI initialized")
    
    async def connect(self) -> bool:
        """
        업비트 연결 확인
        
        Returns:
            연결 성공 여부
        """
        try:
            # 잔고 조회로 연결 테스트
            balances = self.upbit.get_balances()
            if balances is not None:
                self.connected = True
                logger.info("Successfully connected to Upbit")
                return True
            else:
                logger.error("Failed to connect to Upbit - invalid credentials")
                return False
                
        except Exception as e:
            logger.error(f"Failed to connect to Upbit: {e}")
            return False
    
    async def disconnect(self):
        """연결 해제"""
        self.connected = False
        logger.info("Disconnected from Upbit")
    
    async def get_balance(self, currency: Optional[str] = None) -> Dict[str, Balance]:
        """
        잔고 조회
        
        Args:
            currency: 특정 통화만 조회 (예: 'KRW', 'BTC')
            
        Returns:
            통화별 잔고 딕셔너리
        """
        try:
            balances_data = self.upbit.get_balances()
            
            if not balances_data:
                logger.warning("No balance data received")
                return {}
            
            balances = {}
            
            for item in balances_data:
                curr = item['currency']
                
                # 특정 통화만 요청한 경우 필터링
                if currency and curr != currency:
                    continue
                
                # Balance 객체 생성
                balance = Balance(
                    currency=curr,
                    free=float(item['balance']),
                    locked=float(item['locked']),
                    total=float(item['balance']) + float(item['locked'])
                )
                
                balances[curr] = balance
                
                logger.debug(f"Balance {curr}: free={balance.free}, locked={balance.locked}")
            
            return balances
            
        except Exception as e:
            logger.error(f"Failed to get balance: {e}")
            return {}
    
    async def get_ticker(self, symbol: str) -> Ticker:
        """
        현재가 조회
        
        Args:
            symbol: 심볼 (예: 'BTC/KRW' → 'KRW-BTC'로 변환)
            
        Returns:
            현재가 정보
        """
        try:
            # 심볼 변환 (BTC/KRW → KRW-BTC)
            upbit_symbol = self._convert_symbol(symbol)
            
            # 현재가 조회
            ticker_data = pyupbit.get_current_price(upbit_symbol)
            
            # 오더북 조회 (bid/ask 가격)
            orderbook = pyupbit.get_orderbook(upbit_symbol)
            
            if not ticker_data or not orderbook:
                raise ValueError(f"Failed to get ticker for {symbol}")
            
            # 첫 번째 오더북 데이터
            ob = orderbook[0] if isinstance(orderbook, list) else orderbook
            
            ticker = Ticker(
                symbol=symbol,
                bid=float(ob['orderbook_units'][0]['bid_price']),
                ask=float(ob['orderbook_units'][0]['ask_price']),
                last=float(ticker_data),
                volume=0.0,  # 별도 조회 필요
                timestamp=datetime.now()
            )
            
            return ticker
            
        except Exception as e:
            logger.error(f"Failed to get ticker for {symbol}: {e}")
            raise
    
    async def place_order(
        self,
        symbol: str,
        side: OrderSide,
        type: OrderType,
        amount: float,
        price: Optional[float] = None
    ) -> Order:
        """
        주문 실행
        
        Args:
            symbol: 거래 심볼 (예: 'BTC/KRW')
            side: 매수/매도
            type: 주문 타입
            amount: 수량 (BTC)
            price: 가격 (KRW)
            
        Returns:
            주문 정보
        """
        try:
            # 파라미터 검증
            if not self.validate_order_params(symbol, side, type, amount, price):
                raise ValueError("Invalid order parameters")
            
            # 심볼 변환
            upbit_symbol = self._convert_symbol(symbol)
            
            # 최소 주문 금액/수량 체크
            if 'BTC' in symbol:
                if amount < self.MIN_ORDER_AMOUNT_BTC:
                    raise ValueError(f"Amount too small: {amount} < {self.MIN_ORDER_AMOUNT_BTC} BTC")
                
                if type == OrderType.LIMIT and price:
                    krw_amount = amount * price
                    if krw_amount < self.MIN_ORDER_AMOUNT_KRW:
                        raise ValueError(f"Order value too small: {krw_amount} < {self.MIN_ORDER_AMOUNT_KRW} KRW")
            
            # 주문 실행
            order_result = None
            
            if side == OrderSide.BUY:
                if type == OrderType.LIMIT:
                    # 지정가 매수
                    logger.info(f"Placing limit buy order: {amount} BTC @ {price} KRW")
                    order_result = self.upbit.buy_limit_order(upbit_symbol, price, amount)
                else:
                    # 시장가 매수 (KRW 금액으로)
                    krw_amount = amount * price if price else amount
                    logger.info(f"Placing market buy order: {krw_amount} KRW")
                    order_result = self.upbit.buy_market_order(upbit_symbol, krw_amount)
                    
            else:  # SELL
                if type == OrderType.LIMIT:
                    # 지정가 매도
                    logger.info(f"Placing limit sell order: {amount} BTC @ {price} KRW")
                    order_result = self.upbit.sell_limit_order(upbit_symbol, price, amount)
                else:
                    # 시장가 매도
                    logger.info(f"Placing market sell order: {amount} BTC")
                    order_result = self.upbit.sell_market_order(upbit_symbol, amount)
            
            if not order_result or 'uuid' not in order_result:
                raise ValueError(f"Order failed: {order_result}")
            
            # Order 객체 생성
            order = Order(
                order_id=order_result['uuid'],
                symbol=symbol,
                side=side,
                type=type,
                price=price,
                amount=amount,
                status=OrderStatus.OPEN,
                timestamp=datetime.now()
            )
            
            logger.info(f"Order placed successfully: {order.order_id}")
            return order
            
        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            raise
    
    async def cancel_order(self, order_id: str, symbol: str = None) -> bool:
        """
        주문 취소
        
        Args:
            order_id: 주문 UUID
            symbol: 미사용 (호환성을 위해 유지)
            
        Returns:
            취소 성공 여부
        """
        try:
            result = self.upbit.cancel_order(order_id)
            
            if result and 'uuid' in result:
                logger.info(f"Order cancelled: {order_id}")
                return True
            else:
                logger.error(f"Failed to cancel order: {result}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False
    
    async def get_order(self, order_id: str, symbol: str = None) -> Order:
        """
        주문 조회
        
        Args:
            order_id: 주문 UUID
            symbol: 미사용
            
        Returns:
            주문 정보
        """
        try:
            order_data = self.upbit.get_order(order_id)
            
            if not order_data:
                raise ValueError(f"Order not found: {order_id}")
            
            # 상태 변환
            status_map = {
                'wait': OrderStatus.OPEN,
                'done': OrderStatus.FILLED,
                'cancel': OrderStatus.CANCELLED
            }
            
            # Order 객체 생성
            order = Order(
                order_id=order_data['uuid'],
                symbol=self._revert_symbol(order_data['market']),
                side=OrderSide.BUY if order_data['side'] == 'bid' else OrderSide.SELL,
                type=OrderType.LIMIT if order_data['ord_type'] == 'limit' else OrderType.MARKET,
                price=float(order_data.get('price', 0)),
                amount=float(order_data['volume']),
                status=status_map.get(order_data['state'], OrderStatus.PENDING),
                filled_amount=float(order_data.get('executed_volume', 0)),
                fee=float(order_data.get('paid_fee', 0)),
                timestamp=datetime.fromisoformat(order_data['created_at'])
            )
            
            return order
            
        except Exception as e:
            logger.error(f"Failed to get order {order_id}: {e}")
            raise
    
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """
        미체결 주문 조회
        
        Args:
            symbol: 특정 심볼만 조회
            
        Returns:
            미체결 주문 리스트
        """
        try:
            # 미체결 주문 조회
            if symbol:
                upbit_symbol = self._convert_symbol(symbol)
                orders_data = self.upbit.get_order(upbit_symbol, state='wait')
            else:
                orders_data = self.upbit.get_order(state='wait')
            
            if not orders_data:
                return []
            
            # 리스트가 아닌 경우 리스트로 변환
            if not isinstance(orders_data, list):
                orders_data = [orders_data]
            
            orders = []
            for order_data in orders_data:
                order = Order(
                    order_id=order_data['uuid'],
                    symbol=self._revert_symbol(order_data['market']),
                    side=OrderSide.BUY if order_data['side'] == 'bid' else OrderSide.SELL,
                    type=OrderType.LIMIT if order_data['ord_type'] == 'limit' else OrderType.MARKET,
                    price=float(order_data.get('price', 0)),
                    amount=float(order_data['volume']),
                    status=OrderStatus.OPEN,
                    filled_amount=float(order_data.get('executed_volume', 0)),
                    fee=float(order_data.get('paid_fee', 0)),
                    timestamp=datetime.fromisoformat(order_data['created_at'])
                )
                orders.append(order)
            
            return orders
            
        except Exception as e:
            logger.error(f"Failed to get open orders: {e}")
            return []
    
    async def get_order_history(
        self,
        symbol: Optional[str] = None,
        limit: int = 100
    ) -> List[Order]:
        """
        주문 히스토리 조회
        
        Args:
            symbol: 특정 심볼만 조회
            limit: 조회 개수
            
        Returns:
            주문 리스트
        """
        try:
            # 완료된 주문 조회
            if symbol:
                upbit_symbol = self._convert_symbol(symbol)
                orders_data = self.upbit.get_order(upbit_symbol, state='done')
            else:
                orders_data = self.upbit.get_order(state='done')
            
            if not orders_data:
                return []
            
            # 리스트가 아닌 경우 리스트로 변환
            if not isinstance(orders_data, list):
                orders_data = [orders_data]
            
            # limit 적용
            orders_data = orders_data[:limit]
            
            orders = []
            for order_data in orders_data:
                order = Order(
                    order_id=order_data['uuid'],
                    symbol=self._revert_symbol(order_data['market']),
                    side=OrderSide.BUY if order_data['side'] == 'bid' else OrderSide.SELL,
                    type=OrderType.LIMIT if order_data['ord_type'] == 'limit' else OrderType.MARKET,
                    price=float(order_data.get('price', 0)),
                    amount=float(order_data['volume']),
                    status=OrderStatus.FILLED,
                    filled_amount=float(order_data.get('executed_volume', 0)),
                    fee=float(order_data.get('paid_fee', 0)),
                    timestamp=datetime.fromisoformat(order_data['created_at'])
                )
                orders.append(order)
            
            return orders
            
        except Exception as e:
            logger.error(f"Failed to get order history: {e}")
            return []
    
    def _convert_symbol(self, symbol: str) -> str:
        """
        심볼 변환 (BTC/KRW → KRW-BTC)
        
        Args:
            symbol: 표준 심볼
            
        Returns:
            업비트 심볼
        """
        if '/' in symbol:
            base, quote = symbol.split('/')
            return f"{quote}-{base}"
        return symbol
    
    def _revert_symbol(self, upbit_symbol: str) -> str:
        """
        심볼 역변환 (KRW-BTC → BTC/KRW)
        
        Args:
            upbit_symbol: 업비트 심볼
            
        Returns:
            표준 심볼
        """
        if '-' in upbit_symbol:
            quote, base = upbit_symbol.split('-')
            return f"{base}/{quote}"
        return upbit_symbol