"""
거래소 API 베이스 클래스
모든 거래소 구현이 따라야 할 인터페이스 정의
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class OrderSide(Enum):
    """주문 방향"""
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """주문 타입"""
    MARKET = "market"
    LIMIT = "limit"


class OrderStatus(Enum):
    """주문 상태"""
    PENDING = "pending"      # 대기중
    OPEN = "open"           # 미체결
    PARTIAL = "partial"     # 부분체결
    FILLED = "filled"       # 완전체결
    CANCELLED = "cancelled" # 취소됨
    FAILED = "failed"       # 실패


@dataclass
class Order:
    """주문 정보"""
    order_id: str
    symbol: str
    side: OrderSide
    type: OrderType
    price: Optional[float]
    amount: float
    status: OrderStatus
    filled_amount: float = 0.0
    filled_price: float = 0.0
    fee: float = 0.0
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class Balance:
    """잔고 정보"""
    currency: str
    free: float      # 사용 가능
    locked: float    # 주문 중
    total: float     # 총액
    
    @property
    def available(self) -> float:
        """사용 가능한 잔고"""
        return self.free


@dataclass
class Ticker:
    """현재가 정보"""
    symbol: str
    bid: float       # 매수 호가
    ask: float       # 매도 호가
    last: float      # 최종 거래가
    volume: float    # 거래량
    timestamp: datetime


class BaseExchangeAPI(ABC):
    """
    거래소 API 베이스 클래스
    실거래 API의 공통 인터페이스를 정의
    """
    
    def __init__(self, api_key: str, secret_key: str, testnet: bool = False):
        """
        거래소 API 초기화
        
        Args:
            api_key: API 키
            secret_key: Secret 키
            testnet: 테스트넷 사용 여부
        """
        self.api_key = api_key
        self.secret_key = secret_key
        self.testnet = testnet
        self.exchange_name = self.__class__.__name__
        
        logger.info(f"{self.exchange_name} initialized (testnet={testnet})")
    
    @abstractmethod
    async def connect(self) -> bool:
        """
        거래소 연결
        
        Returns:
            연결 성공 여부
        """
        pass
    
    @abstractmethod
    async def disconnect(self):
        """거래소 연결 해제"""
        pass
    
    @abstractmethod
    async def get_balance(self, currency: Optional[str] = None) -> Dict[str, Balance]:
        """
        잔고 조회
        
        Args:
            currency: 특정 통화만 조회 (None이면 전체)
            
        Returns:
            통화별 잔고 딕셔너리
        """
        pass
    
    @abstractmethod
    async def get_ticker(self, symbol: str) -> Ticker:
        """
        현재가 조회
        
        Args:
            symbol: 심볼 (예: BTC/KRW)
            
        Returns:
            현재가 정보
        """
        pass
    
    @abstractmethod
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
            symbol: 거래 심볼
            side: 매수/매도
            type: 주문 타입
            amount: 수량
            price: 가격 (지정가인 경우)
            
        Returns:
            주문 정보
        """
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str, symbol: str = None) -> bool:
        """
        주문 취소
        
        Args:
            order_id: 주문 ID
            symbol: 거래 심볼 (일부 거래소에서 필요)
            
        Returns:
            취소 성공 여부
        """
        pass
    
    @abstractmethod
    async def get_order(self, order_id: str, symbol: str = None) -> Order:
        """
        주문 조회
        
        Args:
            order_id: 주문 ID
            symbol: 거래 심볼 (일부 거래소에서 필요)
            
        Returns:
            주문 정보
        """
        pass
    
    @abstractmethod
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """
        미체결 주문 조회
        
        Args:
            symbol: 특정 심볼만 조회 (None이면 전체)
            
        Returns:
            미체결 주문 리스트
        """
        pass
    
    @abstractmethod
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
        pass
    
    async def place_limit_order(
        self,
        symbol: str,
        side: OrderSide,
        amount: float,
        price: float
    ) -> Order:
        """
        지정가 주문 (편의 메서드)
        
        Args:
            symbol: 거래 심볼
            side: 매수/매도
            amount: 수량
            price: 가격
            
        Returns:
            주문 정보
        """
        return await self.place_order(
            symbol=symbol,
            side=side,
            type=OrderType.LIMIT,
            amount=amount,
            price=price
        )
    
    async def place_market_order(
        self,
        symbol: str,
        side: OrderSide,
        amount: float
    ) -> Order:
        """
        시장가 주문 (편의 메서드)
        
        Args:
            symbol: 거래 심볼
            side: 매수/매도
            amount: 수량
            
        Returns:
            주문 정보
        """
        return await self.place_order(
            symbol=symbol,
            side=side,
            type=OrderType.MARKET,
            amount=amount,
            price=None
        )
    
    async def get_available_balance(self, currency: str) -> float:
        """
        사용 가능한 잔고 조회 (편의 메서드)
        
        Args:
            currency: 통화
            
        Returns:
            사용 가능한 잔고
        """
        balances = await self.get_balance(currency)
        if currency in balances:
            return balances[currency].available
        return 0.0
    
    def validate_order_params(
        self,
        symbol: str,
        side: OrderSide,
        type: OrderType,
        amount: float,
        price: Optional[float] = None
    ) -> bool:
        """
        주문 파라미터 검증
        
        Args:
            symbol: 거래 심볼
            side: 매수/매도
            type: 주문 타입
            amount: 수량
            price: 가격
            
        Returns:
            유효성 여부
        """
        # 기본 검증
        if amount <= 0:
            logger.error(f"Invalid amount: {amount}")
            return False
        
        if type == OrderType.LIMIT and (price is None or price <= 0):
            logger.error(f"Invalid limit price: {price}")
            return False
        
        # 심볼 형식 검증
        if "/" not in symbol:
            logger.error(f"Invalid symbol format: {symbol}")
            return False
        
        return True
    
    def __str__(self) -> str:
        """문자열 표현"""
        return f"{self.exchange_name}(testnet={self.testnet})"
    
    def __repr__(self) -> str:
        """개발자용 표현"""
        return f"{self.exchange_name}(api_key=***{self.api_key[-4:]}, testnet={self.testnet})"