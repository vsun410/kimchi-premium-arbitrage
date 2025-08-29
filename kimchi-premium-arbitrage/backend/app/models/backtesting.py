"""
Backtesting models
"""
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, JSON, Text
from sqlalchemy.orm import relationship
from datetime import datetime

from app.models.base import Base


class Backtest(Base):
    """Backtest model"""
    __tablename__ = "backtests"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    strategy_id = Column(Integer, ForeignKey("strategies.id"), nullable=False)
    
    # Backtest configuration
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    initial_capital = Column(Float, nullable=False)
    parameters = Column(JSON)  # Strategy parameters
    
    # Execution details
    status = Column(String, default="pending")  # pending, running, completed, failed, cancelled
    progress = Column(Integer, default=0)  # Progress percentage
    
    # Results (stored as JSON for flexibility)
    results = Column(JSON)
    metrics = Column(JSON)
    
    # Error handling
    error_message = Column(Text)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    
    # Relationships
    strategy = relationship("Strategy", backref="backtests")
    backtest_results = relationship("BacktestResult", back_populates="backtest", cascade="all, delete-orphan")
    backtest_trades = relationship("BacktestTrade", back_populates="backtest", cascade="all, delete-orphan")


class BacktestResult(Base):
    """Backtest result model"""
    __tablename__ = "backtest_results"
    
    id = Column(Integer, primary_key=True, index=True)
    backtest_id = Column(Integer, ForeignKey("backtests.id"), nullable=False)
    
    # Performance metrics
    total_return = Column(Float)
    total_pnl = Column(Float)
    sharpe_ratio = Column(Float)
    calmar_ratio = Column(Float)
    max_drawdown = Column(Float)
    win_rate = Column(Float)
    
    # Trade statistics
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    losing_trades = Column(Integer, default=0)
    avg_win = Column(Float)
    avg_loss = Column(Float)
    profit_factor = Column(Float)
    
    # Risk metrics
    var_95 = Column(Float)  # Value at Risk 95%
    cvar_95 = Column(Float)  # Conditional Value at Risk 95%
    
    # Additional metrics (stored as JSON)
    additional_metrics = Column(JSON)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    backtest = relationship("Backtest", back_populates="backtest_results")


class BacktestTrade(Base):
    """Backtest trade model"""
    __tablename__ = "backtest_trades"
    
    id = Column(Integer, primary_key=True, index=True)
    backtest_id = Column(Integer, ForeignKey("backtests.id"), nullable=False)
    
    # Trade details
    symbol = Column(String, nullable=False)
    side = Column(String, nullable=False)  # buy/sell
    quantity = Column(Float, nullable=False)
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float)
    
    # Timestamps
    entry_time = Column(DateTime, nullable=False)
    exit_time = Column(DateTime)
    
    # Performance
    pnl = Column(Float, default=0)
    pnl_percentage = Column(Float, default=0)
    fees = Column(Float, default=0)
    
    # Trade metadata
    trade_metadata = Column(JSON)
    
    # Relationships
    backtest = relationship("Backtest", back_populates="backtest_trades")