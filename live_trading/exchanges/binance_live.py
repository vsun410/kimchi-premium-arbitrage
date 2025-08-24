"""
바이낸스 선물 실거래 API 구현
ccxt 라이브러리를 사용한 실제 거래 기능
"""

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
import ccxt.pro as ccxtpro
from decimal import Decimal

from .base_exchange import (
    BaseExchangeAPI, Order, Balance, Ticker,
    OrderSide, OrderType, OrderStatus
)

logger = logging.getLogger(__name__)


class BinanceLiveAPI(BaseExchangeAPI):
    """
    바이낸스 선물 실거래 API
    
    주의사항:
    - 실제 자금이 사용되므로 매우 신중히 사용
    - 모든 주문은 로깅됨
    - Testnet 사용 가능 (testnet=True)
    """
    
    def __init__(self, api_key: str, secret_key: str, testnet: bool = False):
        """
        바이낸스 API 초기화
        
        Args:
            api_key: 바이낸스 API 키
            secret_key: 바이낸스 Secret 키
            testnet: 테스트넷 사용 여부
        """
        super().__init__(api_key, secret_key, testnet)
        
        # CCXT Pro 설정
        config = {
            'apiKey': api_key,
            'secret': secret_key,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',  # 선물 거래
                'adjustForTimeDifference': True
            }
        }
        
        if testnet:
            config['urls'] = {
                'api': {
                    'public': 'https://testnet.binance.vision',
                    'private': 'https://testnet.binance.vision',
                    'futures': 'https://testnet.binancefuture.com'
                }
            }
            logger.info("Using Binance testnet")
        
        self.exchange = ccxtpro.binance(config)
        self.connected = False
        
        # 최소 주문 금액 (USDT)
        self.MIN_ORDER_AMOUNT_USDT = 10.0
        
        # 최소 주문 수량 (BTC)
        self.MIN_ORDER_AMOUNT_BTC = 0.001
        
        logger.info(f"BinanceLiveAPI initialized (testnet={testnet})")
    
    async def connect(self) -> bool:
        """
        바이낸스 연결 확인
        
        Returns:
            연결 성공 여부
        """
        try:
            # 시장 정보 로드
            await self.exchange.load_markets()
            
            # 잔고 조회로 연결 테스트
            balance = await self.exchange.fetch_balance()
            if balance:
                self.connected = True
                logger.info("Successfully connected to Binance")
                return True
            else:
                logger.error("Failed to connect to Binance - invalid credentials")
                return False
                
        except Exception as e:
            logger.error(f"Failed to connect to Binance: {e}")
            return False
    
    async def disconnect(self):
        """연결 해제"""
        try:
            await self.exchange.close()
            self.connected = False
            logger.info("Disconnected from Binance")
        except Exception as e:
            logger.error(f"Error disconnecting from Binance: {e}")
    
    async def get_balance(self, currency: Optional[str] = None) -> Dict[str, Balance]:
        """
        잔고 조회
        
        Args:
            currency: 특정 통화만 조회 (예: 'USDT', 'BTC')
            
        Returns:
            통화별 잔고 딕셔너리
        """
        try:
            balance_data = await self.exchange.fetch_balance()
            
            if not balance_data or 'info' not in balance_data:
                logger.warning("No balance data received")
                return {}
            
            balances = {}
            
            # CCXT 표준 형식으로 파싱
            for curr, data in balance_data.items():
                if curr in ['info', 'free', 'used', 'total']:
                    continue
                
                # 특정 통화만 요청한 경우 필터링
                if currency and curr != currency:
                    continue
                
                if isinstance(data, dict):
                    # Balance 객체 생성
                    balance = Balance(
                        currency=curr,
                        free=float(data.get('free', 0)),
                        locked=float(data.get('used', 0)),
                        total=float(data.get('total', 0))
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
            symbol: 심볼 (예: 'BTC/USDT')
            
        Returns:
            현재가 정보
        """
        try:
            # Ticker 조회
            ticker_data = await self.exchange.fetch_ticker(symbol)
            
            if not ticker_data:
                raise ValueError(f"Failed to get ticker for {symbol}")
            
            ticker = Ticker(
                symbol=symbol,
                bid=float(ticker_data['bid']),
                ask=float(ticker_data['ask']),
                last=float(ticker_data['last']),
                volume=float(ticker_data['volume']),
                timestamp=datetime.fromtimestamp(ticker_data['timestamp'] / 1000)
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
            symbol: 거래 심볼 (예: 'BTC/USDT')
            side: 매수/매도
            type: 주문 타입
            amount: 수량 (BTC)
            price: 가격 (USDT)
            
        Returns:
            주문 정보
        """
        try:
            # 파라미터 검증
            if not self.validate_order_params(symbol, side, type, amount, price):
                raise ValueError("Invalid order parameters")
            
            # 최소 주문 수량 체크
            if 'BTC' in symbol:
                if amount < self.MIN_ORDER_AMOUNT_BTC:
                    raise ValueError(f"Amount too small: {amount} < {self.MIN_ORDER_AMOUNT_BTC} BTC")
                
                if type == OrderType.LIMIT and price:
                    usdt_amount = amount * price
                    if usdt_amount < self.MIN_ORDER_AMOUNT_USDT:
                        raise ValueError(f"Order value too small: {usdt_amount} < {self.MIN_ORDER_AMOUNT_USDT} USDT")
            
            # CCXT 파라미터 준비
            order_side = side.value
            order_type = type.value
            
            # 주문 실행
            if type == OrderType.LIMIT:
                logger.info(f"Placing limit {order_side} order: {amount} {symbol} @ {price}")
                order_result = await self.exchange.create_order(
                    symbol=symbol,
                    type=order_type,
                    side=order_side,
                    amount=amount,
                    price=price
                )
            else:
                logger.info(f"Placing market {order_side} order: {amount} {symbol}")
                order_result = await self.exchange.create_order(
                    symbol=symbol,
                    type=order_type,
                    side=order_side,
                    amount=amount
                )
            
            if not order_result or 'id' not in order_result:
                raise ValueError(f"Order failed: {order_result}")
            
            # Order 객체 생성
            order = Order(
                order_id=str(order_result['id']),
                symbol=symbol,
                side=side,
                type=type,
                price=price if price else order_result.get('price'),
                amount=amount,
                status=self._convert_order_status(order_result.get('status')),
                filled_amount=float(order_result.get('filled', 0)),
                filled_price=float(order_result.get('average', 0)),
                fee=float(order_result.get('fee', {}).get('cost', 0)),
                timestamp=datetime.fromtimestamp(order_result['timestamp'] / 1000)
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
            order_id: 주문 ID
            symbol: 거래 심볼 (바이낸스는 필수)
            
        Returns:
            취소 성공 여부
        """
        try:
            if not symbol:
                raise ValueError("Symbol is required for Binance order cancellation")
            
            result = await self.exchange.cancel_order(order_id, symbol)
            
            if result and result.get('status') == 'canceled':
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
            order_id: 주문 ID
            symbol: 거래 심볼 (바이낸스는 필수)
            
        Returns:
            주문 정보
        """
        try:
            if not symbol:
                raise ValueError("Symbol is required for Binance order query")
            
            order_data = await self.exchange.fetch_order(order_id, symbol)
            
            if not order_data:
                raise ValueError(f"Order not found: {order_id}")
            
            # Order 객체 생성
            order = Order(
                order_id=str(order_data['id']),
                symbol=order_data['symbol'],
                side=OrderSide.BUY if order_data['side'] == 'buy' else OrderSide.SELL,
                type=OrderType.LIMIT if order_data['type'] == 'limit' else OrderType.MARKET,
                price=float(order_data.get('price', 0)),
                amount=float(order_data['amount']),
                status=self._convert_order_status(order_data['status']),
                filled_amount=float(order_data.get('filled', 0)),
                filled_price=float(order_data.get('average', 0)),
                fee=float(order_data.get('fee', {}).get('cost', 0)),
                timestamp=datetime.fromtimestamp(order_data['timestamp'] / 1000)
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
            orders_data = await self.exchange.fetch_open_orders(symbol)
            
            if not orders_data:
                return []
            
            orders = []
            for order_data in orders_data:
                order = Order(
                    order_id=str(order_data['id']),
                    symbol=order_data['symbol'],
                    side=OrderSide.BUY if order_data['side'] == 'buy' else OrderSide.SELL,
                    type=OrderType.LIMIT if order_data['type'] == 'limit' else OrderType.MARKET,
                    price=float(order_data.get('price', 0)),
                    amount=float(order_data['amount']),
                    status=self._convert_order_status(order_data['status']),
                    filled_amount=float(order_data.get('filled', 0)),
                    filled_price=float(order_data.get('average', 0)),
                    fee=float(order_data.get('fee', {}).get('cost', 0)),
                    timestamp=datetime.fromtimestamp(order_data['timestamp'] / 1000)
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
            orders_data = await self.exchange.fetch_closed_orders(symbol, limit=limit)
            
            if not orders_data:
                return []
            
            orders = []
            for order_data in orders_data:
                order = Order(
                    order_id=str(order_data['id']),
                    symbol=order_data['symbol'],
                    side=OrderSide.BUY if order_data['side'] == 'buy' else OrderSide.SELL,
                    type=OrderType.LIMIT if order_data['type'] == 'limit' else OrderType.MARKET,
                    price=float(order_data.get('price', 0)),
                    amount=float(order_data['amount']),
                    status=self._convert_order_status(order_data['status']),
                    filled_amount=float(order_data.get('filled', 0)),
                    filled_price=float(order_data.get('average', 0)),
                    fee=float(order_data.get('fee', {}).get('cost', 0)),
                    timestamp=datetime.fromtimestamp(order_data['timestamp'] / 1000)
                )
                orders.append(order)
            
            return orders
            
        except Exception as e:
            logger.error(f"Failed to get order history: {e}")
            return []
    
    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        """
        레버리지 설정 (선물 전용)
        
        Args:
            symbol: 거래 심볼
            leverage: 레버리지 배수 (1-125)
            
        Returns:
            설정 성공 여부
        """
        try:
            result = await self.exchange.set_leverage(leverage, symbol)
            logger.info(f"Leverage set to {leverage}x for {symbol}")
            return True
        except Exception as e:
            logger.error(f"Failed to set leverage: {e}")
            return False
    
    async def get_position(self, symbol: str) -> Dict[str, Any]:
        """
        포지션 조회 (선물 전용)
        
        Args:
            symbol: 거래 심볼
            
        Returns:
            포지션 정보
        """
        try:
            positions = await self.exchange.fetch_positions([symbol])
            
            if positions and len(positions) > 0:
                position = positions[0]
                return {
                    'symbol': position['symbol'],
                    'side': position['side'],
                    'contracts': position['contracts'],
                    'contractSize': position['contractSize'],
                    'unrealizedPnl': position['unrealizedPnl'],
                    'percentage': position['percentage'],
                    'markPrice': position['markPrice'],
                    'entryPrice': position.get('entryPrice'),
                    'timestamp': position['timestamp']
                }
            
            return {}
            
        except Exception as e:
            logger.error(f"Failed to get position: {e}")
            return {}
    
    def _convert_order_status(self, status: str) -> OrderStatus:
        """
        CCXT 주문 상태를 내부 상태로 변환
        
        Args:
            status: CCXT 상태
            
        Returns:
            내부 주문 상태
        """
        status_map = {
            'open': OrderStatus.OPEN,
            'closed': OrderStatus.FILLED,
            'canceled': OrderStatus.CANCELLED,
            'expired': OrderStatus.CANCELLED,
            'rejected': OrderStatus.FAILED,
            'partially_filled': OrderStatus.PARTIAL
        }
        
        return status_map.get(status, OrderStatus.PENDING)