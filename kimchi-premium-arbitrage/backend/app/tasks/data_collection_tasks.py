"""
Celery tasks for data collection operations
"""
from celery import Task
from typing import Dict, Any, List, Optional
import asyncio
from datetime import datetime, timedelta
import json
import aiohttp

from app.core.celery_app import celery_app
from app.core.database import AsyncSessionLocal
from app.core.config import settings
from app.models.market_data import MarketData, OrderBook, KimchiPremium
from sqlalchemy import select, and_, delete
import ccxt.pro as ccxtpro

class DataCollectionTask(Task):
    """Base task for data collection"""
    def __init__(self):
        self.exchanges = {}
        
    async def get_exchange(self, exchange_name: str):
        """Get or create exchange instance"""
        if exchange_name not in self.exchanges:
            if exchange_name == "upbit":
                self.exchanges[exchange_name] = ccxtpro.upbit({
                    'enableRateLimit': True,
                    'options': {'defaultType': 'spot'}
                })
            elif exchange_name == "binance":
                self.exchanges[exchange_name] = ccxtpro.binance({
                    'enableRateLimit': True,
                    'options': {'defaultType': 'future'}
                })
        return self.exchanges[exchange_name]

@celery_app.task(bind=True, base=DataCollectionTask, name='collect_market_data')
def collect_market_data(self, symbols: List[str] = None) -> Dict[str, Any]:
    """
    Collect market data for specified symbols
    
    Args:
        symbols: List of symbols to collect data for (default: BTC, ETH, XRP)
        
    Returns:
        Dict with collection status
    """
    if symbols is None:
        symbols = ['BTC/USDT', 'ETH/USDT', 'XRP/USDT']
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        result = loop.run_until_complete(_collect_market_data_async(self, symbols))
        return result
    finally:
        loop.close()

async def _collect_market_data_async(task: DataCollectionTask, symbols: List[str]) -> Dict[str, Any]:
    """Async implementation of market data collection"""
    collected_data = []
    errors = []
    
    async with AsyncSessionLocal() as db:
        for symbol in symbols:
            try:
                # Collect from Upbit
                upbit = await task.get_exchange('upbit')
                upbit_ticker = await upbit.fetch_ticker(symbol.replace('/USDT', '/KRW'))
                
                # Collect from Binance
                binance = await task.get_exchange('binance')
                binance_ticker = await binance.fetch_ticker(symbol)
                
                # Store in database
                market_data = MarketData(
                    exchange="upbit",
                    symbol=symbol,
                    price=upbit_ticker['last'],
                    volume=upbit_ticker['quoteVolume'],
                    bid=upbit_ticker['bid'],
                    ask=upbit_ticker['ask'],
                    timestamp=datetime.utcnow()
                )
                db.add(market_data)
                
                market_data = MarketData(
                    exchange="binance",
                    symbol=symbol,
                    price=binance_ticker['last'],
                    volume=binance_ticker['quoteVolume'],
                    bid=binance_ticker['bid'],
                    ask=binance_ticker['ask'],
                    timestamp=datetime.utcnow()
                )
                db.add(market_data)
                
                collected_data.append(symbol)
                
            except Exception as e:
                errors.append({
                    "symbol": symbol,
                    "error": str(e)
                })
        
        await db.commit()
    
    return {
        "status": "completed",
        "collected": collected_data,
        "errors": errors,
        "timestamp": datetime.utcnow().isoformat()
    }

@celery_app.task(bind=True, base=DataCollectionTask, name='collect_orderbook_data')
def collect_orderbook_data(self, symbols: List[str] = None, depth: int = 20) -> Dict[str, Any]:
    """
    Collect orderbook data for specified symbols
    
    Args:
        symbols: List of symbols to collect data for
        depth: Orderbook depth to collect
        
    Returns:
        Dict with collection status
    """
    if symbols is None:
        symbols = ['BTC/USDT', 'ETH/USDT', 'XRP/USDT']
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        result = loop.run_until_complete(_collect_orderbook_async(self, symbols, depth))
        return result
    finally:
        loop.close()

async def _collect_orderbook_async(
    task: DataCollectionTask,
    symbols: List[str],
    depth: int
) -> Dict[str, Any]:
    """Async implementation of orderbook collection"""
    collected_data = []
    
    async with AsyncSessionLocal() as db:
        for symbol in symbols:
            try:
                # Collect from both exchanges
                for exchange_name in ['upbit', 'binance']:
                    exchange = await task.get_exchange(exchange_name)
                    
                    # Adjust symbol for Upbit
                    exchange_symbol = symbol
                    if exchange_name == 'upbit':
                        exchange_symbol = symbol.replace('/USDT', '/KRW')
                    
                    orderbook = await exchange.fetch_order_book(exchange_symbol, depth)
                    
                    # Store in database
                    orderbook_data = OrderBook(
                        exchange=exchange_name,
                        symbol=symbol,
                        bids=json.dumps(orderbook['bids'][:depth]),
                        asks=json.dumps(orderbook['asks'][:depth]),
                        timestamp=datetime.utcnow()
                    )
                    db.add(orderbook_data)
                
                collected_data.append(symbol)
                
            except Exception as e:
                print(f"Error collecting orderbook for {symbol}: {e}")
        
        await db.commit()
    
    return {
        "status": "completed",
        "collected": collected_data,
        "depth": depth,
        "timestamp": datetime.utcnow().isoformat()
    }

@celery_app.task(name='update_kimchi_premium')
def update_kimchi_premium() -> Dict[str, Any]:
    """
    Calculate and update Kimchi premium for all active pairs
    
    Returns:
        Dict with update status
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        result = loop.run_until_complete(_update_kimchi_premium_async())
        return result
    finally:
        loop.close()

async def _update_kimchi_premium_async() -> Dict[str, Any]:
    """Async implementation of Kimchi premium calculation"""
    updated_pairs = []
    
    async with AsyncSessionLocal() as db:
        # Get latest prices for each symbol
        symbols = ['BTC/USDT', 'ETH/USDT', 'XRP/USDT']
        
        for symbol in symbols:
            try:
                # Get latest Upbit price (in KRW)
                upbit_result = await db.execute(
                    select(MarketData)
                    .where(and_(
                        MarketData.exchange == "upbit",
                        MarketData.symbol == symbol
                    ))
                    .order_by(MarketData.timestamp.desc())
                    .limit(1)
                )
                upbit_data = upbit_result.scalar_one_or_none()
                
                # Get latest Binance price (in USDT)
                binance_result = await db.execute(
                    select(MarketData)
                    .where(and_(
                        MarketData.exchange == "binance",
                        MarketData.symbol == symbol
                    ))
                    .order_by(MarketData.timestamp.desc())
                    .limit(1)
                )
                binance_data = binance_result.scalar_one_or_none()
                
                if upbit_data and binance_data:
                    # Get current USD/KRW rate (simplified - should use real rate)
                    usd_krw_rate = 1300.0  # Placeholder
                    
                    # Calculate Kimchi premium
                    upbit_price_usd = upbit_data.price / usd_krw_rate
                    binance_price_usd = binance_data.price
                    premium_percentage = ((upbit_price_usd - binance_price_usd) / binance_price_usd) * 100
                    
                    # Store Kimchi premium
                    kimchi_premium = KimchiPremium(
                        symbol=symbol.replace('/USDT', ''),
                        upbit_price_krw=upbit_data.price,
                        binance_price_usdt=binance_data.price,
                        usd_krw_rate=usd_krw_rate,
                        premium_percentage=premium_percentage,
                        timestamp=datetime.utcnow()
                    )
                    db.add(kimchi_premium)
                    
                    updated_pairs.append({
                        "symbol": symbol,
                        "premium": premium_percentage
                    })
                    
            except Exception as e:
                print(f"Error calculating Kimchi premium for {symbol}: {e}")
        
        await db.commit()
    
    return {
        "status": "completed",
        "updated_pairs": updated_pairs,
        "timestamp": datetime.utcnow().isoformat()
    }

@celery_app.task(name='cleanup_old_data')
def cleanup_old_data(days_to_keep: int = 30) -> Dict[str, Any]:
    """
    Clean up old market data older than specified days
    
    Args:
        days_to_keep: Number of days of data to keep
        
    Returns:
        Dict with cleanup status
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        result = loop.run_until_complete(_cleanup_old_data_async(days_to_keep))
        return result
    finally:
        loop.close()

async def _cleanup_old_data_async(days_to_keep: int) -> Dict[str, Any]:
    """Async implementation of data cleanup"""
    cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
    deleted_counts = {}
    
    async with AsyncSessionLocal() as db:
        # Clean up MarketData
        result = await db.execute(
            delete(MarketData).where(MarketData.timestamp < cutoff_date)
        )
        deleted_counts['market_data'] = result.rowcount
        
        # Clean up OrderBook
        result = await db.execute(
            delete(OrderBook).where(OrderBook.timestamp < cutoff_date)
        )
        deleted_counts['orderbook'] = result.rowcount
        
        # Clean up KimchiPremium
        result = await db.execute(
            delete(KimchiPremium).where(KimchiPremium.timestamp < cutoff_date)
        )
        deleted_counts['kimchi_premium'] = result.rowcount
        
        await db.commit()
    
    return {
        "status": "completed",
        "deleted_counts": deleted_counts,
        "cutoff_date": cutoff_date.isoformat(),
        "timestamp": datetime.utcnow().isoformat()
    }

@celery_app.task(name='fetch_exchange_rate')
def fetch_exchange_rate() -> Dict[str, Any]:
    """
    Fetch current USD/KRW exchange rate from external API
    
    Returns:
        Dict with exchange rate data
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        result = loop.run_until_complete(_fetch_exchange_rate_async())
        return result
    finally:
        loop.close()

async def _fetch_exchange_rate_async() -> Dict[str, Any]:
    """Async implementation of exchange rate fetching"""
    try:
        # In production, use a real FX API like Fixer.io or Exchange Rates API
        # This is a placeholder implementation
        async with aiohttp.ClientSession() as session:
            # Example API endpoint (replace with actual)
            url = "https://api.exchangerate-api.com/v4/latest/USD"
            async with session.get(url) as response:
                data = await response.json()
                krw_rate = data.get('rates', {}).get('KRW', 1300.0)
                
                # Store in cache or database
                return {
                    "status": "success",
                    "usd_krw": krw_rate,
                    "timestamp": datetime.utcnow().isoformat()
                }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "fallback_rate": 1300.0,
            "timestamp": datetime.utcnow().isoformat()
        }