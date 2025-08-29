"""
Paper Trading schemas
"""
from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum


class OrderSide(str, Enum):
    """주문 방향"""
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    """주문 유형"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(str, Enum):
    """주문 상태"""
    PENDING = "pending"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class SessionStatus(str, Enum):
    """세션 상태"""
    ACTIVE = "active"
    PAUSED = "paused"
    STOPPED = "stopped"
    COMPLETED = "completed"


class PaperTradingSessionCreate(BaseModel):
    """Paper Trading 세션 생성"""
    name: str = Field(..., description="세션 이름")
    strategy_id: Optional[int] = Field(None, description="전략 ID")
    initial_balance_krw: float = Field(20000000, gt=0, description="초기 KRW 잔고")
    initial_balance_usdt: float = Field(15000, gt=0, description="초기 USDT 잔고")
    initial_btc: float = Field(0, ge=0, description="초기 BTC 보유량")
    description: Optional[str] = Field(None, description="세션 설명")


class PaperTradingSessionUpdate(BaseModel):
    """Paper Trading 세션 업데이트"""
    name: Optional[str] = Field(None, description="세션 이름")
    description: Optional[str] = Field(None, description="세션 설명")
    is_active: Optional[bool] = Field(None, description="활성 상태")

class PaperSessionUpdate(PaperTradingSessionUpdate):
    """Alias for PaperTradingSessionUpdate"""
    pass

class PaperTradingSessionResponse(BaseModel):
    """Paper Trading 세션 응답"""
    id: int
    name: str
    strategy_id: Optional[int]
    status: SessionStatus
    initial_balance_krw: float
    initial_balance_usdt: float
    current_balance_krw: float
    current_balance_usdt: float
    btc_balance: float
    total_pnl: float
    total_pnl_percentage: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    created_at: datetime
    last_trade_at: Optional[datetime]


class PaperOrderCreate(BaseModel):
    """Paper 주문 생성"""
    session_id: int = Field(..., description="세션 ID")
    exchange: str = Field(..., description="거래소 (upbit/binance)")
    symbol: str = Field(..., description="심볼 (BTC/KRW, BTC/USDT)")
    side: OrderSide = Field(..., description="매수/매도")
    order_type: OrderType = Field(..., description="주문 유형")
    amount: float = Field(..., gt=0, description="수량")
    price: Optional[float] = Field(None, gt=0, description="가격 (limit 주문)")
    stop_price: Optional[float] = Field(None, gt=0, description="스탑 가격")


class PaperOrderResponse(BaseModel):
    """Paper 주문 응답"""
    id: str
    session_id: int
    exchange: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    status: OrderStatus
    amount: float
    price: Optional[float]
    executed_price: Optional[float]
    executed_amount: float
    fee: float
    created_at: datetime
    executed_at: Optional[datetime]


class PaperPositionResponse(BaseModel):
    """Paper 포지션 응답"""
    session_id: int
    exchange: str
    symbol: str
    side: str  # long/short
    amount: float
    entry_price: float
    current_price: float
    pnl: float
    pnl_percentage: float
    created_at: datetime


class PaperBalanceResponse(BaseModel):
    """Paper 잔고 응답"""
    session_id: int
    upbit: Dict[str, float]  # {"KRW": 20000000, "BTC": 0.001}
    binance: Dict[str, float]  # {"USDT": 15000, "BTC": 0.001}
    total_value_krw: float
    initial_value_krw: float
    pnl_krw: float
    pnl_percentage: float
    timestamp: datetime


class PaperTradeHistory(BaseModel):
    """Paper 거래 내역"""
    id: str
    session_id: int
    timestamp: datetime
    exchange: str
    symbol: str
    side: OrderSide
    price: float
    amount: float
    fee: float
    pnl: Optional[float]
    balance_after: Dict[str, float]


class PaperSessionMetrics(BaseModel):
    """Paper Trading 세션 성과 지표"""
    session_id: int
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    total_trades: int
    total_volume: float
    total_fees: float