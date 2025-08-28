"""
Celery tasks for analysis operations
"""
from celery import Task
from typing import Dict, Any, List, Optional
import asyncio
from datetime import datetime, timedelta
import json
import pandas as pd
import numpy as np

from app.core.celery_app import celery_app
from app.core.database import AsyncSessionLocal
from app.models.market_data import MarketData, KimchiPremium
from app.models.trading import Trade, Position
from sqlalchemy import select, and_, func

@celery_app.task(name='generate_daily_report')
def generate_daily_report(date: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate daily trading report
    
    Args:
        date: Date for report (YYYY-MM-DD format), defaults to yesterday
        
    Returns:
        Dict with report data
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        result = loop.run_until_complete(_generate_daily_report_async(date))
        return result
    finally:
        loop.close()

async def _generate_daily_report_async(date: Optional[str]) -> Dict[str, Any]:
    """Async implementation of daily report generation"""
    if date:
        report_date = datetime.strptime(date, "%Y-%m-%d")
    else:
        report_date = datetime.utcnow() - timedelta(days=1)
    
    start_time = report_date.replace(hour=0, minute=0, second=0)
    end_time = report_date.replace(hour=23, minute=59, second=59)
    
    async with AsyncSessionLocal() as db:
        # Get trading statistics
        trades_result = await db.execute(
            select(
                func.count(Trade.id).label('total_trades'),
                func.sum(Trade.pnl).label('total_pnl'),
                func.avg(Trade.pnl).label('avg_pnl')
            ).where(
                and_(
                    Trade.created_at >= start_time,
                    Trade.created_at <= end_time
                )
            )
        )
        trade_stats = trades_result.first()
        
        # Get Kimchi premium statistics
        premium_result = await db.execute(
            select(
                func.avg(KimchiPremium.premium_percentage).label('avg_premium'),
                func.max(KimchiPremium.premium_percentage).label('max_premium'),
                func.min(KimchiPremium.premium_percentage).label('min_premium')
            ).where(
                and_(
                    KimchiPremium.timestamp >= start_time,
                    KimchiPremium.timestamp <= end_time
                )
            )
        )
        premium_stats = premium_result.first()
        
        # Get position statistics
        positions_result = await db.execute(
            select(
                func.count(Position.id).label('total_positions'),
                func.sum(Position.unrealized_pnl).label('total_unrealized_pnl')
            ).where(
                Position.is_open == True
            )
        )
        position_stats = positions_result.first()
        
        report = {
            "date": report_date.strftime("%Y-%m-%d"),
            "trading": {
                "total_trades": trade_stats.total_trades or 0,
                "total_pnl": float(trade_stats.total_pnl or 0),
                "avg_pnl": float(trade_stats.avg_pnl or 0)
            },
            "kimchi_premium": {
                "avg_premium": float(premium_stats.avg_premium or 0),
                "max_premium": float(premium_stats.max_premium or 0),
                "min_premium": float(premium_stats.min_premium or 0)
            },
            "positions": {
                "open_positions": position_stats.total_positions or 0,
                "unrealized_pnl": float(position_stats.total_unrealized_pnl or 0)
            },
            "generated_at": datetime.utcnow().isoformat()
        }
        
        # TODO: Send report via email or save to storage
        
        return report

@celery_app.task(name='calculate_portfolio_metrics')
def calculate_portfolio_metrics() -> Dict[str, Any]:
    """
    Calculate portfolio performance metrics
    
    Returns:
        Dict with portfolio metrics
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        result = loop.run_until_complete(_calculate_portfolio_metrics_async())
        return result
    finally:
        loop.close()

async def _calculate_portfolio_metrics_async() -> Dict[str, Any]:
    """Async implementation of portfolio metrics calculation"""
    async with AsyncSessionLocal() as db:
        # Get all trades
        trades_result = await db.execute(
            select(Trade).order_by(Trade.created_at)
        )
        trades = trades_result.scalars().all()
        
        if not trades:
            return {
                "status": "no_trades",
                "metrics": {}
            }
        
        # Convert to DataFrame for easier calculation
        trades_data = [
            {
                "date": t.created_at,
                "pnl": float(t.pnl),
                "symbol": t.symbol
            }
            for t in trades
        ]
        df = pd.DataFrame(trades_data)
        df.set_index('date', inplace=True)
        
        # Calculate daily returns
        daily_pnl = df.groupby(pd.Grouper(freq='D'))['pnl'].sum()
        returns = daily_pnl.pct_change().dropna()
        
        # Calculate metrics
        metrics = {}
        
        # Sharpe Ratio (assuming 0% risk-free rate)
        if len(returns) > 0:
            metrics['sharpe_ratio'] = (returns.mean() / returns.std()) * np.sqrt(252)
        else:
            metrics['sharpe_ratio'] = 0
        
        # Maximum Drawdown
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        metrics['max_drawdown'] = float(drawdown.min() * 100)
        
        # Win Rate
        winning_trades = len([t for t in trades if t.pnl > 0])
        total_trades = len(trades)
        metrics['win_rate'] = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # Profit Factor
        gross_profit = sum([float(t.pnl) for t in trades if t.pnl > 0])
        gross_loss = abs(sum([float(t.pnl) for t in trades if t.pnl < 0]))
        metrics['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Average Trade
        metrics['avg_trade_pnl'] = float(df['pnl'].mean())
        
        # Total PnL
        metrics['total_pnl'] = float(df['pnl'].sum())
        
        return {
            "status": "success",
            "metrics": metrics,
            "period": {
                "start": trades[0].created_at.isoformat(),
                "end": trades[-1].created_at.isoformat(),
                "total_days": (trades[-1].created_at - trades[0].created_at).days
            },
            "calculated_at": datetime.utcnow().isoformat()
        }

@celery_app.task(name='detect_arbitrage_opportunities')
def detect_arbitrage_opportunities(
    min_premium: float = 2.0,
    min_volume: float = 10000.0
) -> Dict[str, Any]:
    """
    Detect arbitrage opportunities based on Kimchi premium
    
    Args:
        min_premium: Minimum premium percentage to consider
        min_volume: Minimum volume requirement
        
    Returns:
        Dict with detected opportunities
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        result = loop.run_until_complete(
            _detect_arbitrage_opportunities_async(min_premium, min_volume)
        )
        return result
    finally:
        loop.close()

async def _detect_arbitrage_opportunities_async(
    min_premium: float,
    min_volume: float
) -> Dict[str, Any]:
    """Async implementation of arbitrage detection"""
    opportunities = []
    
    async with AsyncSessionLocal() as db:
        # Get latest Kimchi premium data
        result = await db.execute(
            select(KimchiPremium)
            .where(KimchiPremium.premium_percentage >= min_premium)
            .order_by(KimchiPremium.timestamp.desc())
            .limit(10)
        )
        high_premium_data = result.scalars().all()
        
        for premium_data in high_premium_data:
            # Check volume requirements
            market_result = await db.execute(
                select(MarketData)
                .where(
                    and_(
                        MarketData.symbol == f"{premium_data.symbol}/USDT",
                        MarketData.timestamp >= premium_data.timestamp - timedelta(minutes=5)
                    )
                )
                .order_by(MarketData.timestamp.desc())
                .limit(2)
            )
            market_data = market_result.scalars().all()
            
            if market_data:
                avg_volume = sum([m.volume for m in market_data]) / len(market_data)
                
                if avg_volume >= min_volume:
                    opportunities.append({
                        "symbol": premium_data.symbol,
                        "premium": float(premium_data.premium_percentage),
                        "upbit_price": float(premium_data.upbit_price_krw),
                        "binance_price": float(premium_data.binance_price_usdt),
                        "volume": float(avg_volume),
                        "timestamp": premium_data.timestamp.isoformat()
                    })
        
        return {
            "status": "success",
            "opportunities": opportunities,
            "filters": {
                "min_premium": min_premium,
                "min_volume": min_volume
            },
            "detected_at": datetime.utcnow().isoformat()
        }

@celery_app.task(name='analyze_market_correlation')
def analyze_market_correlation(
    symbols: List[str] = None,
    period_days: int = 30
) -> Dict[str, Any]:
    """
    Analyze correlation between different market pairs
    
    Args:
        symbols: List of symbols to analyze
        period_days: Period for correlation analysis
        
    Returns:
        Dict with correlation matrix
    """
    if symbols is None:
        symbols = ['BTC/USDT', 'ETH/USDT', 'XRP/USDT']
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        result = loop.run_until_complete(
            _analyze_market_correlation_async(symbols, period_days)
        )
        return result
    finally:
        loop.close()

async def _analyze_market_correlation_async(
    symbols: List[str],
    period_days: int
) -> Dict[str, Any]:
    """Async implementation of market correlation analysis"""
    start_time = datetime.utcnow() - timedelta(days=period_days)
    
    async with AsyncSessionLocal() as db:
        price_data = {}
        
        for symbol in symbols:
            result = await db.execute(
                select(MarketData)
                .where(
                    and_(
                        MarketData.symbol == symbol,
                        MarketData.exchange == "binance",
                        MarketData.timestamp >= start_time
                    )
                )
                .order_by(MarketData.timestamp)
            )
            market_data = result.scalars().all()
            
            if market_data:
                prices = [float(m.price) for m in market_data]
                price_data[symbol] = prices
        
        if len(price_data) < 2:
            return {
                "status": "insufficient_data",
                "message": "Not enough data for correlation analysis"
            }
        
        # Create DataFrame and calculate correlation
        df = pd.DataFrame(price_data)
        correlation_matrix = df.corr()
        
        # Convert to serializable format
        correlation_dict = {}
        for symbol1 in correlation_matrix.index:
            correlation_dict[symbol1] = {}
            for symbol2 in correlation_matrix.columns:
                correlation_dict[symbol1][symbol2] = float(correlation_matrix.loc[symbol1, symbol2])
        
        return {
            "status": "success",
            "correlation_matrix": correlation_dict,
            "period": {
                "days": period_days,
                "start": start_time.isoformat(),
                "end": datetime.utcnow().isoformat()
            },
            "analyzed_at": datetime.utcnow().isoformat()
        }