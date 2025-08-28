"""
Market data models
"""
from sqlalchemy import Column, String, Float, Integer, DateTime, Index, UniqueConstraint
from app.models.base import Base


class PriceData(Base):
    """Price data model for storing OHLCV data"""
    __tablename__ = "price_data"
    __table_args__ = (
        UniqueConstraint('exchange', 'symbol', 'timestamp', name='_exchange_symbol_timestamp_uc'),
        Index('idx_price_data_lookup', 'exchange', 'symbol', 'timestamp'),
        {'timescaledb_hypertable': {'time_column_name': 'timestamp'}}
    )
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, nullable=False)
    exchange = Column(String(50), nullable=False)
    symbol = Column(String(20), nullable=False)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    quote_volume = Column(Float)
    trades_count = Column(Integer)


class OrderBookSnapshot(Base):
    """Order book snapshot model"""
    __tablename__ = "orderbook_snapshots"
    __table_args__ = (
        Index('idx_orderbook_lookup', 'exchange', 'symbol', 'timestamp'),
        {'timescaledb_hypertable': {'time_column_name': 'timestamp'}}
    )
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, nullable=False)
    exchange = Column(String(50), nullable=False)
    symbol = Column(String(20), nullable=False)
    bids = Column(String)  # JSON string of bid levels
    asks = Column(String)  # JSON string of ask levels
    bid_volume = Column(Float)
    ask_volume = Column(Float)
    spread = Column(Float)
    mid_price = Column(Float)


class PremiumData(Base):
    """Kimchi premium data model"""
    __tablename__ = "premium_data"
    __table_args__ = (
        Index('idx_premium_lookup', 'symbol', 'timestamp'),
        {'timescaledb_hypertable': {'time_column_name': 'timestamp'}}
    )
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, nullable=False)
    symbol = Column(String(20), nullable=False)
    upbit_price = Column(Float, nullable=False)
    binance_price = Column(Float, nullable=False)
    usd_krw_rate = Column(Float, nullable=False)
    premium_rate = Column(Float, nullable=False)
    premium_amount = Column(Float, nullable=False)
    volume_upbit = Column(Float)
    volume_binance = Column(Float)
    spread_upbit = Column(Float)
    spread_binance = Column(Float)