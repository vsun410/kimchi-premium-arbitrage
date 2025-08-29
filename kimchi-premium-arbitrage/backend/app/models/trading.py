"""
Trading related models
"""
from sqlalchemy import Column, String, Float, Integer, Boolean, DateTime, ForeignKey, Enum as SQLEnum
from sqlalchemy.orm import relationship
from app.models.base import BaseModel
import enum


class OrderStatus(str, enum.Enum):
    """Order status enum"""
    PENDING = "pending"
    OPEN = "open"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    FAILED = "failed"


class OrderType(str, enum.Enum):
    """Order type enum"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(str, enum.Enum):
    """Order side enum"""
    BUY = "buy"
    SELL = "sell"


class Position(BaseModel):
    """Position model"""
    __tablename__ = "positions"
    
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    session_id = Column(Integer, ForeignKey("trading_sessions.id"))
    exchange = Column(String(50), nullable=False)
    symbol = Column(String(20), nullable=False)
    side = Column(SQLEnum(OrderSide), nullable=False)
    quantity = Column(Float, nullable=False)
    entry_price = Column(Float, nullable=False)
    current_price = Column(Float)
    realized_pnl = Column(Float, default=0)
    unrealized_pnl = Column(Float, default=0)
    is_open = Column(Boolean, default=True)
    opened_at = Column(DateTime)
    closed_at = Column(DateTime)
    
    # Hedge position info
    is_hedge = Column(Boolean, default=False)
    hedge_pair_id = Column(Integer, ForeignKey("positions.id"))
    
    # Relationships
    orders = relationship("Order", back_populates="position")
    hedge_pair = relationship("Position", remote_side=[id], uselist=False)


class Order(BaseModel):
    """Order model"""
    __tablename__ = "orders"
    
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    position_id = Column(Integer, ForeignKey("positions.id"))
    session_id = Column(Integer, ForeignKey("trading_sessions.id"))
    exchange = Column(String(50), nullable=False)
    exchange_order_id = Column(String(100))
    symbol = Column(String(20), nullable=False)
    order_type = Column(SQLEnum(OrderType), nullable=False)
    side = Column(SQLEnum(OrderSide), nullable=False)
    price = Column(Float)
    quantity = Column(Float, nullable=False)
    filled_quantity = Column(Float, default=0)
    status = Column(SQLEnum(OrderStatus), nullable=False)
    fee = Column(Float, default=0)
    fee_currency = Column(String(10))
    executed_at = Column(DateTime)
    
    # Relationships
    position = relationship("Position", back_populates="orders")


class TradingSession(BaseModel):
    """Trading session model"""
    __tablename__ = "trading_sessions"
    
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    strategy_id = Column(Integer, ForeignKey("strategies.id"))
    name = Column(String(100), nullable=False)
    is_paper = Column(Boolean, default=True)
    is_active = Column(Boolean, default=True)
    starting_balance = Column(Float, nullable=False)
    current_balance = Column(Float)
    total_pnl = Column(Float, default=0)
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    losing_trades = Column(Integer, default=0)
    max_drawdown = Column(Float, default=0)
    sharpe_ratio = Column(Float)
    started_at = Column(DateTime)
    ended_at = Column(DateTime)
    
    # Relationships
    user = relationship("User", back_populates="trading_sessions")
    strategy = relationship("Strategy", back_populates="sessions")