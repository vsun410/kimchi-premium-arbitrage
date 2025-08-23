"""
데이터 모델 스키마 정의
Pydantic을 사용한 데이터 검증 및 직렬화
"""

from pydantic import BaseModel, Field, validator
from datetime import datetime
from typing import Optional, List, Tuple, Dict
from enum import Enum


class Exchange(str, Enum):
    """거래소 종류"""
    UPBIT = "upbit"
    BINANCE = "binance"


class Symbol(str, Enum):
    """거래 심볼"""
    BTC = "BTC"
    ETH = "ETH"
    XRP = "XRP"


class OrderSide(str, Enum):
    """주문 방향"""
    BUY = "buy"
    SELL = "sell"


class SignalAction(str, Enum):
    """시그널 액션"""
    ENTER = "enter"
    EXIT = "exit"
    HOLD = "hold"
    REHEDGE = "rehedge"


class PriceData(BaseModel):
    """가격 데이터 모델"""
    timestamp: datetime
    exchange: Exchange
    symbol: Symbol
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    volume: float = Field(ge=0)
    kimchi_premium_rate: Optional[float] = None
    exchange_rate: Optional[float] = Field(None, gt=0)
    
    @validator('high')
    def high_gte_low(cls, v, values):
        if 'low' in values and v < values['low']:
            raise ValueError('high must be >= low')
        return v
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class OrderBookLevel(BaseModel):
    """오더북 레벨"""
    price: float = Field(gt=0)
    size: float = Field(gt=0)


class OrderBookData(BaseModel):
    """오더북 데이터 모델"""
    timestamp: datetime
    exchange: Exchange
    symbol: Symbol
    bids: List[OrderBookLevel]  # 매수 호가
    asks: List[OrderBookLevel]  # 매도 호가
    liquidity_score: float = Field(ge=0, le=100)
    spread_percentage: float = Field(ge=0)
    bid_ask_imbalance: Optional[float] = None
    
    @validator('spread_percentage')
    def calculate_spread(cls, v, values):
        if 'bids' in values and 'asks' in values:
            if values['bids'] and values['asks']:
                best_bid = values['bids'][0].price
                best_ask = values['asks'][0].price
                return ((best_ask - best_bid) / best_ask) * 100
        return v


class Signal(BaseModel):
    """거래 시그널 모델"""
    timestamp: datetime
    action: SignalAction
    confidence: float = Field(ge=0, le=1)  # 0~1 신뢰도
    position_size: float = Field(gt=0)  # 포지션 크기
    expected_return: float  # 예상 수익률 (%)
    risk_score: float = Field(ge=0, le=100)  # 리스크 점수
    reason: str  # 시그널 생성 이유
    metadata: Optional[Dict] = None


class Position(BaseModel):
    """포지션 모델"""
    entry_time: datetime
    symbol: Symbol
    spot_entry_price: float = Field(gt=0)  # 현물 진입가
    futures_entry_price: float = Field(gt=0)  # 선물 진입가
    size: float = Field(gt=0)  # 포지션 크기
    current_pnl: float = 0.0  # 현재 손익
    status: str = "open"  # open, closed, liquidated
    hedge_ratio: float = Field(default=1.0, ge=0)  # 헤지 비율
    kimchi_premium_at_entry: float  # 진입 시 김프
    exit_time: Optional[datetime] = None
    exit_reason: Optional[str] = None
    final_pnl: Optional[float] = None
    
    @property
    def duration(self) -> Optional[float]:
        """포지션 보유 시간 (시간 단위)"""
        if self.exit_time:
            return (self.exit_time - self.entry_time).total_seconds() / 3600
        return None


class ExchangeRate(BaseModel):
    """환율 데이터 모델"""
    timestamp: datetime
    usd_krw: float = Field(gt=0)
    source: str  # API 소스
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class TradingMetrics(BaseModel):
    """거래 성과 지표"""
    total_trades: int = Field(ge=0)
    winning_trades: int = Field(ge=0)
    losing_trades: int = Field(ge=0)
    win_rate: float = Field(ge=0, le=100)
    average_return: float  # 평균 수익률 (%)
    sharpe_ratio: float
    calmar_ratio: float
    max_drawdown: float  # 최대 낙폭 (%)
    total_pnl: float  # 총 손익
    roi: float  # ROI (%)
    
    @validator('win_rate')
    def calculate_win_rate(cls, v, values):
        if 'total_trades' in values and values['total_trades'] > 0:
            return (values.get('winning_trades', 0) / values['total_trades']) * 100
        return 0.0