"""
Backtest model
"""
from sqlalchemy import Column, String, Float, Integer, Boolean, DateTime, Text, JSON, ForeignKey
from sqlalchemy.orm import relationship
from app.models.base import BaseModel


class Backtest(BaseModel):
    """Backtest model"""
    __tablename__ = "backtests"
    
    strategy_id = Column(Integer, ForeignKey("strategies.id"), nullable=False)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    initial_capital = Column(Float, nullable=False)
    final_capital = Column(Float)
    
    # Performance metrics
    total_return = Column(Float)
    annual_return = Column(Float)
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    losing_trades = Column(Integer, default=0)
    win_rate = Column(Float)
    profit_factor = Column(Float)
    max_drawdown = Column(Float)
    max_drawdown_duration = Column(Integer)  # in days
    sharpe_ratio = Column(Float)
    sortino_ratio = Column(Float)
    calmar_ratio = Column(Float)
    
    # Trade statistics
    avg_win = Column(Float)
    avg_loss = Column(Float)
    largest_win = Column(Float)
    largest_loss = Column(Float)
    avg_trade_duration = Column(Float)  # in hours
    
    # Risk metrics
    value_at_risk = Column(Float)  # VaR
    conditional_value_at_risk = Column(Float)  # CVaR
    
    # Configuration
    parameters = Column(JSON)  # Backtest parameters
    fee_rate = Column(Float)
    slippage = Column(Float)
    
    # Results
    trades = Column(JSON)  # List of all trades
    equity_curve = Column(JSON)  # Time series of portfolio value
    daily_returns = Column(JSON)  # Daily return series
    
    # Execution info
    executed_at = Column(DateTime)
    execution_time = Column(Float)  # in seconds
    
    # Relationships
    strategy = relationship("Strategy", back_populates="backtests")