"""
Paper trading models
"""
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, JSON, Enum as SQLEnum
from sqlalchemy.orm import relationship
from datetime import datetime
import enum

from app.models.base import Base


class OrderStatus(str, enum.Enum):
    """Order status enum"""
    PENDING = "pending"
    EXECUTED = "executed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class OrderType(str, enum.Enum):
    """Order type enum"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class PaperTradingSession(Base):
    """Paper trading session model"""
    __tablename__ = "paper_trading_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    description = Column(String)
    
    # Initial balances
    initial_balance_krw = Column(Float, default=20000000)  # 20M KRW
    initial_balance_usd = Column(Float, default=15000)     # 15K USD
    
    # Current balances
    current_balance_krw = Column(Float)
    current_balance_usd = Column(Float)
    
    # Trading statistics
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    losing_trades = Column(Integer, default=0)
    total_pnl = Column(Float, default=0)
    total_fees = Column(Float, default=0)
    
    # Session status
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    orders = relationship("PaperOrder", back_populates="session", cascade="all, delete-orphan")
    positions = relationship("PaperPosition", back_populates="session", cascade="all, delete-orphan")


class PaperOrder(Base):
    """Paper trading order model"""
    __tablename__ = "paper_orders"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("paper_trading_sessions.id"), nullable=False)
    
    # Order details
    symbol = Column(String, nullable=False, index=True)
    side = Column(String, nullable=False)  # buy/sell
    order_type = Column(SQLEnum(OrderType), default=OrderType.MARKET)
    quantity = Column(Float, nullable=False)
    price = Column(Float)  # For limit orders
    stop_price = Column(Float)  # For stop orders
    
    # Execution details
    executed_price = Column(Float)
    executed_quantity = Column(Float)
    fees = Column(Float, default=0)
    
    # Status
    status = Column(SQLEnum(OrderStatus), default=OrderStatus.PENDING)
    exchange = Column(String)  # upbit/binance
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    executed_at = Column(DateTime)
    cancelled_at = Column(DateTime)
    
    # Metadata
    order_metadata = Column(JSON)
    
    # Relationships
    session = relationship("PaperTradingSession", back_populates="orders")


class PaperPosition(Base):
    """Paper trading position model"""
    __tablename__ = "paper_positions"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("paper_trading_sessions.id"), nullable=False)
    
    # Position details
    symbol = Column(String, nullable=False, index=True)
    side = Column(String, nullable=False)  # long/short
    quantity = Column(Float, nullable=False)
    entry_price = Column(Float, nullable=False)
    
    # Current state
    current_price = Column(Float)
    pnl = Column(Float, default=0)
    pnl_percentage = Column(Float, default=0)
    
    # Risk metrics
    max_profit = Column(Float, default=0)
    max_loss = Column(Float, default=0)
    
    # Status
    is_open = Column(Boolean, default=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    closed_at = Column(DateTime)
    
    # Relationships
    session = relationship("PaperTradingSession", back_populates="positions")