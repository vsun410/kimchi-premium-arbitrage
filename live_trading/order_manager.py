"""
주문 관리자 (Order Manager)
모든 주문의 실행, 추적, 관리를 담당

목적: 안전하고 정확한 주문 실행
결과: 모든 주문 추적 가능, 실패 시 자동 복구
평가: 주문 성공률, 슬리피지, 실행 시간 측정
"""

import asyncio
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
from decimal import Decimal, ROUND_DOWN

from .exchanges.base_exchange import (
    BaseExchangeAPI, Order, OrderSide, OrderType, OrderStatus
)
from .exchanges.upbit_live import UpbitLiveAPI
from .exchanges.binance_live import BinanceLiveAPI
from .price_validator import RealTimePriceValidator
from .exchange_rate_fetcher import get_current_exchange_rate

logger = logging.getLogger(__name__)


class ExecutionStrategy(Enum):
    """주문 실행 전략"""
    AGGRESSIVE = "aggressive"    # 즉시 시장가
    PASSIVE = "passive"         # 지정가 대기
    ADAPTIVE = "adaptive"       # 상황에 따라 조정


@dataclass
class OrderRequest:
    """주문 요청"""
    exchange: str               # 'upbit' or 'binance'
    symbol: str                # 'BTC/KRW' or 'BTC/USDT'
    side: OrderSide            # BUY or SELL
    amount: float              # 수량
    strategy: ExecutionStrategy = ExecutionStrategy.PASSIVE
    max_slippage_pct: float = 0.1  # 최대 슬리피지 0.1%
    timeout_seconds: int = 30      # 주문 타임아웃
    
    def __post_init__(self):
        """유효성 검증"""
        if self.amount <= 0:
            raise ValueError(f"Invalid amount: {self.amount}")
        if self.max_slippage_pct < 0 or self.max_slippage_pct > 1:
            raise ValueError(f"Invalid slippage: {self.max_slippage_pct}")


@dataclass
class OrderResult:
    """주문 결과"""
    request: OrderRequest
    order: Optional[Order]
    success: bool
    error_message: Optional[str] = None
    execution_time_ms: float = 0
    slippage_pct: float = 0
    
    def to_dict(self) -> Dict:
        """딕셔너리 변환"""
        return {
            'exchange': self.request.exchange,
            'symbol': self.request.symbol,
            'side': self.request.side.value,
            'amount': self.request.amount,
            'success': self.success,
            'order_id': self.order.order_id if self.order else None,
            'filled_amount': self.order.filled_amount if self.order else 0,
            'filled_price': self.order.filled_price if self.order else 0,
            'error': self.error_message,
            'execution_time_ms': self.execution_time_ms,
            'slippage_pct': self.slippage_pct
        }


class OrderManager:
    """
    주문 관리자
    
    핵심 기능:
    1. 주문 실행 및 관리
    2. 주문 상태 추적
    3. 실패 시 재시도
    4. 슬리피지 관리
    5. 주문 이력 관리
    """
    
    def __init__(
        self,
        upbit_api: Optional[UpbitLiveAPI] = None,
        binance_api: Optional[BinanceLiveAPI] = None
    ):
        """
        초기화
        
        Args:
            upbit_api: 업비트 API
            binance_api: 바이낸스 API
        """
        self.exchanges: Dict[str, BaseExchangeAPI] = {}
        
        if upbit_api:
            self.exchanges['upbit'] = upbit_api
        if binance_api:
            self.exchanges['binance'] = binance_api
            
        # 주문 추적
        self.active_orders: Dict[str, Order] = {}  # order_id -> Order
        self.order_history: List[OrderResult] = []
        
        # 가격 검증기
        self.price_validator = RealTimePriceValidator()
        
        # 통계
        self.stats = {
            'total_orders': 0,
            'successful_orders': 0,
            'failed_orders': 0,
            'total_volume_krw': 0,
            'total_volume_usdt': 0,
            'avg_slippage_pct': 0,
            'avg_execution_time_ms': 0
        }
        
        logger.info(f"OrderManager initialized with exchanges: {list(self.exchanges.keys())}")
    
    async def connect_all(self) -> bool:
        """
        모든 거래소 연결
        
        Returns:
            모든 연결 성공 여부
        """
        results = []
        
        for name, exchange in self.exchanges.items():
            try:
                connected = await exchange.connect()
                results.append(connected)
                logger.info(f"{name} connection: {'success' if connected else 'failed'}")
            except Exception as e:
                logger.error(f"Failed to connect {name}: {e}")
                results.append(False)
        
        return all(results) if results else False
    
    async def execute_order(self, request: OrderRequest) -> OrderResult:
        """
        주문 실행
        
        Args:
            request: 주문 요청
            
        Returns:
            주문 결과
        """
        start_time = datetime.now()
        
        try:
            # 거래소 확인
            if request.exchange not in self.exchanges:
                raise ValueError(f"Exchange not available: {request.exchange}")
            
            exchange = self.exchanges[request.exchange]
            
            # 현재가 조회
            ticker = await exchange.get_ticker(request.symbol)
            
            # 가격 검증 (업비트인 경우)
            if request.exchange == 'upbit':
                await self._validate_price(ticker.last, request.symbol)
            
            # 실행 전략에 따라 주문
            if request.strategy == ExecutionStrategy.AGGRESSIVE:
                order = await self._execute_aggressive(exchange, request, ticker)
            elif request.strategy == ExecutionStrategy.PASSIVE:
                order = await self._execute_passive(exchange, request, ticker)
            else:  # ADAPTIVE
                order = await self._execute_adaptive(exchange, request, ticker)
            
            # 실행 시간 계산
            execution_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            # 슬리피지 계산
            slippage_pct = self._calculate_slippage(
                request.side,
                ticker.last,
                order.filled_price if order.filled_price else order.price
            )
            
            # 결과 생성
            result = OrderResult(
                request=request,
                order=order,
                success=True,
                execution_time_ms=execution_time_ms,
                slippage_pct=slippage_pct
            )
            
            # 통계 업데이트
            self._update_statistics(result)
            
            # 이력 저장
            self.order_history.append(result)
            
            logger.info(
                f"Order executed: {request.exchange} {request.symbol} "
                f"{request.side.value} {request.amount} "
                f"(slippage: {slippage_pct:.3f}%, time: {execution_time_ms:.0f}ms)"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Order execution failed: {e}")
            
            execution_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            result = OrderResult(
                request=request,
                order=None,
                success=False,
                error_message=str(e),
                execution_time_ms=execution_time_ms
            )
            
            self.order_history.append(result)
            self.stats['failed_orders'] += 1
            
            return result
    
    async def _execute_aggressive(
        self,
        exchange: BaseExchangeAPI,
        request: OrderRequest,
        ticker: Any
    ) -> Order:
        """
        공격적 실행 (시장가)
        
        Args:
            exchange: 거래소 API
            request: 주문 요청
            ticker: 현재가 정보
            
        Returns:
            주문 객체
        """
        logger.info("Executing aggressive (market) order")
        
        # 시장가 주문
        order = await exchange.place_market_order(
            symbol=request.symbol,
            side=request.side,
            amount=request.amount
        )
        
        # 주문 추적
        self.active_orders[order.order_id] = order
        
        # 체결 대기
        await self._wait_for_fill(exchange, order, request.timeout_seconds)
        
        return order
    
    async def _execute_passive(
        self,
        exchange: BaseExchangeAPI,
        request: OrderRequest,
        ticker: Any
    ) -> Order:
        """
        수동적 실행 (지정가)
        
        Args:
            exchange: 거래소 API
            request: 주문 요청
            ticker: 현재가 정보
            
        Returns:
            주문 객체
        """
        logger.info("Executing passive (limit) order")
        
        # 지정가 계산 (메이커 주문)
        if request.side == OrderSide.BUY:
            # 매수: 현재 매수호가보다 낮게
            price = ticker.bid * 0.9999  # 0.01% 낮게
        else:
            # 매도: 현재 매도호가보다 높게
            price = ticker.ask * 1.0001  # 0.01% 높게
        
        # 가격 라운딩
        price = self._round_price(price, request.symbol)
        
        # 지정가 주문
        order = await exchange.place_limit_order(
            symbol=request.symbol,
            side=request.side,
            amount=request.amount,
            price=price
        )
        
        # 주문 추적
        self.active_orders[order.order_id] = order
        
        # 체결 대기 (타임아웃 있음)
        filled = await self._wait_for_fill(exchange, order, request.timeout_seconds)
        
        # 타임아웃 시 취소하고 시장가로 전환
        if not filled:
            logger.warning("Limit order timeout, converting to market order")
            
            # 주문 취소
            await exchange.cancel_order(order.order_id, request.symbol)
            
            # 시장가로 재실행
            order = await exchange.place_market_order(
                symbol=request.symbol,
                side=request.side,
                amount=request.amount
            )
            
            self.active_orders[order.order_id] = order
        
        return order
    
    async def _execute_adaptive(
        self,
        exchange: BaseExchangeAPI,
        request: OrderRequest,
        ticker: Any
    ) -> Order:
        """
        적응형 실행 (상황에 따라 조정)
        
        Args:
            exchange: 거래소 API
            request: 주문 요청
            ticker: 현재가 정보
            
        Returns:
            주문 객체
        """
        logger.info("Executing adaptive order")
        
        # 스프레드 계산
        spread_pct = (ticker.ask - ticker.bid) / ticker.bid * 100
        
        # 스프레드가 작으면 지정가, 크면 시장가
        if spread_pct < 0.05:  # 0.05% 미만
            logger.info(f"Small spread ({spread_pct:.3f}%), using limit order")
            return await self._execute_passive(exchange, request, ticker)
        else:
            logger.info(f"Large spread ({spread_pct:.3f}%), using market order")
            return await self._execute_aggressive(exchange, request, ticker)
    
    async def _wait_for_fill(
        self,
        exchange: BaseExchangeAPI,
        order: Order,
        timeout_seconds: int
    ) -> bool:
        """
        주문 체결 대기
        
        Args:
            exchange: 거래소 API
            order: 주문 객체
            timeout_seconds: 타임아웃 (초)
            
        Returns:
            체결 여부
        """
        start_time = datetime.now()
        
        while (datetime.now() - start_time).total_seconds() < timeout_seconds:
            # 주문 상태 조회
            updated_order = await exchange.get_order(order.order_id, order.symbol)
            
            # 체결 확인
            if updated_order.status == OrderStatus.FILLED:
                self.active_orders[order.order_id] = updated_order
                return True
            
            # 부분 체결
            if updated_order.status == OrderStatus.PARTIAL:
                self.active_orders[order.order_id] = updated_order
            
            # 실패 또는 취소
            if updated_order.status in [OrderStatus.CANCELLED, OrderStatus.FAILED]:
                if order.order_id in self.active_orders:
                    del self.active_orders[order.order_id]
                return False
            
            await asyncio.sleep(0.5)  # 0.5초 대기
        
        return False
    
    async def _validate_price(self, price: float, symbol: str):
        """
        가격 검증
        
        Args:
            price: 가격
            symbol: 심볼
        """
        if 'KRW' in symbol:
            # 업비트 가격 검증
            validation = await self.price_validator.validate_btc_price(
                our_upbit_price=price,
                our_binance_price=0  # 바이낸스는 검증 안함
            )
            
            if not validation[0].is_valid:
                logger.warning(f"Price validation failed: {validation[0].difference_pct:.2f}% difference")
    
    def _calculate_slippage(
        self,
        side: OrderSide,
        expected_price: float,
        actual_price: float
    ) -> float:
        """
        슬리피지 계산
        
        Args:
            side: 매수/매도
            expected_price: 예상 가격
            actual_price: 실제 체결 가격
            
        Returns:
            슬리피지 (%)
        """
        if not actual_price or not expected_price:
            return 0
        
        if side == OrderSide.BUY:
            # 매수: 실제가 더 높으면 손실
            slippage = (actual_price - expected_price) / expected_price * 100
        else:
            # 매도: 실제가 더 낮으면 손실
            slippage = (expected_price - actual_price) / expected_price * 100
        
        return slippage
    
    def _round_price(self, price: float, symbol: str) -> float:
        """
        가격 라운딩
        
        Args:
            price: 원본 가격
            symbol: 심볼
            
        Returns:
            라운딩된 가격
        """
        if 'KRW' in symbol:
            # 원화: 정수 단위
            return round(price)
        elif 'USDT' in symbol or 'USD' in symbol:
            # 달러: 소수점 2자리
            return round(price, 2)
        else:
            return price
    
    def _update_statistics(self, result: OrderResult):
        """
        통계 업데이트
        
        Args:
            result: 주문 결과
        """
        self.stats['total_orders'] += 1
        
        if result.success:
            self.stats['successful_orders'] += 1
            
            # 거래량 업데이트
            if result.order:
                if 'KRW' in result.request.symbol:
                    volume = result.order.filled_amount * result.order.filled_price
                    self.stats['total_volume_krw'] += volume
                else:
                    volume = result.order.filled_amount * result.order.filled_price
                    self.stats['total_volume_usdt'] += volume
            
            # 평균 계산
            n = self.stats['successful_orders']
            
            # 슬리피지 평균
            prev_avg_slip = self.stats['avg_slippage_pct']
            self.stats['avg_slippage_pct'] = (
                (prev_avg_slip * (n - 1) + abs(result.slippage_pct)) / n
            )
            
            # 실행 시간 평균
            prev_avg_time = self.stats['avg_execution_time_ms']
            self.stats['avg_execution_time_ms'] = (
                (prev_avg_time * (n - 1) + result.execution_time_ms) / n
            )
        else:
            self.stats['failed_orders'] += 1
    
    async def cancel_all_orders(self, exchange_name: Optional[str] = None):
        """
        모든 주문 취소
        
        Args:
            exchange_name: 특정 거래소만 (None이면 전체)
        """
        exchanges_to_cancel = (
            {exchange_name: self.exchanges[exchange_name]}
            if exchange_name and exchange_name in self.exchanges
            else self.exchanges
        )
        
        for name, exchange in exchanges_to_cancel.items():
            try:
                open_orders = await exchange.get_open_orders()
                
                for order in open_orders:
                    await exchange.cancel_order(order.order_id, order.symbol)
                    logger.info(f"Cancelled order {order.order_id} on {name}")
                    
                    if order.order_id in self.active_orders:
                        del self.active_orders[order.order_id]
                        
            except Exception as e:
                logger.error(f"Failed to cancel orders on {name}: {e}")
    
    def get_statistics(self) -> Dict:
        """
        통계 조회
        
        Returns:
            통계 정보
        """
        success_rate = (
            self.stats['successful_orders'] / self.stats['total_orders'] * 100
            if self.stats['total_orders'] > 0
            else 0
        )
        
        return {
            **self.stats,
            'success_rate': success_rate,
            'active_orders_count': len(self.active_orders)
        }
    
    def get_recent_orders(self, limit: int = 10) -> List[Dict]:
        """
        최근 주문 조회
        
        Args:
            limit: 조회 개수
            
        Returns:
            최근 주문 리스트
        """
        recent = self.order_history[-limit:] if len(self.order_history) > limit else self.order_history
        return [order.to_dict() for order in reversed(recent)]