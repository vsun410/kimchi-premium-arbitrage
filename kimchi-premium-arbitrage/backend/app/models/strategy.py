"""
Strategy model
"""
from sqlalchemy import Column, String, Float, Integer, Boolean, DateTime, Text, JSON
from sqlalchemy.orm import relationship
from app.models.base import BaseModel


class Strategy(BaseModel):
    """Strategy model"""
    __tablename__ = "strategies"
    
    name = Column(String(100), unique=True, nullable=False)
    description = Column(Text)
    strategy_type = Column(String(50), nullable=False)  # 'kimchi_premium', 'triangular_arb', etc.
    version = Column(String(20), nullable=False)
    is_active = Column(Boolean, default=True)
    
    # Strategy parameters (stored as JSON)
    parameters = Column(JSON)
    
    # Performance metrics
    total_trades = Column(Integer, default=0)
    win_rate = Column(Float, default=0)
    avg_profit = Column(Float, default=0)
    max_drawdown = Column(Float, default=0)
    sharpe_ratio = Column(Float)
    calmar_ratio = Column(Float)
    
    # ML model info
    model_type = Column(String(50))  # 'lstm', 'xgboost', 'ensemble'
    model_path = Column(String(255))
    model_version = Column(String(20))
    last_trained = Column(DateTime)
    training_metrics = Column(JSON)
    
    # Risk parameters
    max_position_size = Column(Float)
    stop_loss = Column(Float)
    take_profit = Column(Float)
    max_daily_loss = Column(Float)
    
    # Relationships
    sessions = relationship("TradingSession", back_populates="strategy")
    backtests = relationship("Backtest", back_populates="strategy")