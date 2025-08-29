"""
System models for alerts and notifications
"""
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, JSON, Text
from sqlalchemy.orm import relationship
from datetime import datetime

from app.models.base import Base


class Alert(Base):
    """Alert model"""
    __tablename__ = "alerts"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    message = Column(Text, nullable=False)
    
    # Alert configuration
    alert_type = Column(String, nullable=False)  # price, premium, position, system
    condition = Column(JSON, nullable=False)  # Alert condition configuration
    priority = Column(String, default="medium")  # low, medium, high, critical
    
    # Notification settings
    notification_channels = Column(JSON)  # email, slack, telegram, etc.
    is_recurring = Column(Boolean, default=True)
    
    # Status
    is_active = Column(Boolean, default=True)
    last_triggered = Column(DateTime)
    trigger_count = Column(Integer, default=0)
    
    # Metadata
    alert_metadata = Column(JSON)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    notifications = relationship("Notification", back_populates="alert", cascade="all, delete-orphan")


class Notification(Base):
    """Notification model"""
    __tablename__ = "notifications"
    
    id = Column(Integer, primary_key=True, index=True)
    alert_id = Column(Integer, ForeignKey("alerts.id"), nullable=True)
    
    # Notification details
    title = Column(String, nullable=False)
    message = Column(Text, nullable=False)
    channel = Column(String, nullable=False)  # email, slack, telegram, webhook
    recipient = Column(String)  # Email address, chat ID, etc.
    
    # Status
    status = Column(String, default="pending")  # pending, sent, failed
    error_message = Column(Text)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    sent_at = Column(DateTime)
    
    # Metadata
    notification_metadata = Column(JSON)
    
    # Relationships
    alert = relationship("Alert", back_populates="notifications")