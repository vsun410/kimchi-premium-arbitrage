"""
Alert model
"""
from sqlalchemy import Column, String, Float, Integer, Boolean, DateTime, Text, ForeignKey, Enum as SQLEnum
from sqlalchemy.orm import relationship
from app.models.base import BaseModel
import enum


class AlertType(str, enum.Enum):
    """Alert type enum"""
    PREMIUM_HIGH = "premium_high"
    PREMIUM_LOW = "premium_low"
    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class AlertPriority(str, enum.Enum):
    """Alert priority enum"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Alert(BaseModel):
    """Alert model"""
    __tablename__ = "alerts"
    
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    alert_type = Column(SQLEnum(AlertType), nullable=False)
    priority = Column(SQLEnum(AlertPriority), nullable=False)
    title = Column(String(200), nullable=False)
    message = Column(Text, nullable=False)
    alert_metadata = Column(Text)  # JSON string of additional data
    is_read = Column(Boolean, default=False)
    is_sent = Column(Boolean, default=False)
    sent_at = Column(DateTime)
    read_at = Column(DateTime)
    
    # Relationships
    user = relationship("User", back_populates="alerts")