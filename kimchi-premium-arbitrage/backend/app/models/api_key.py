"""
API Key model for exchange connections
"""
from sqlalchemy import Column, String, Boolean, Integer, ForeignKey, Text
from sqlalchemy.orm import relationship
from app.models.base import BaseModel


class APIKey(BaseModel):
    """API Key model for managing exchange API keys"""
    __tablename__ = "api_keys"
    
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    exchange = Column(String(50), nullable=False)  # 'binance', 'upbit'
    name = Column(String(100), nullable=False)
    api_key = Column(Text, nullable=False)  # Encrypted
    api_secret = Column(Text, nullable=False)  # Encrypted
    passphrase = Column(Text)  # For exchanges that require it (encrypted)
    is_testnet = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)
    permissions = Column(Text)  # JSON string of permissions
    
    # Relationships
    user = relationship("User", back_populates="api_keys")