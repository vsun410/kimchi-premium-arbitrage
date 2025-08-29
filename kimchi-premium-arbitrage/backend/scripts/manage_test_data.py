"""
CLI script to manage test data
Usage: python scripts/manage_test_data.py [command] [options]
"""
import asyncio
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import click
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text

from app.core.config import settings
from app.models import Base
from app.fixtures.data_generator import TestDataGenerator

# Create async engine
engine = create_async_engine(settings.DATABASE_URL, echo=False)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

@click.group()
def cli():
    """Test data management CLI"""
    pass

@cli.command()
@click.option('--drop-existing', is_flag=True, help='Drop existing tables first')
async def init_db(drop_existing):
    """Initialize database schema"""
    async with engine.begin() as conn:
        if drop_existing:
            click.echo("Dropping existing tables...")
            await conn.run_sync(Base.metadata.drop_all)
        
        click.echo("Creating tables...")
        await conn.run_sync(Base.metadata.create_all)
        
        # Enable TimescaleDB extensions if using PostgreSQL
        try:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb"))
            click.echo("TimescaleDB extension enabled")
        except Exception as e:
            click.echo(f"Could not enable TimescaleDB: {e}")
    
    click.echo("Database initialized successfully!")

@cli.command()
@click.option('--seed', default=42, help='Random seed for reproducibility')
@click.option('--market-data/--no-market-data', default=True, help='Include market data')
@click.option('--trades/--no-trades', default=True, help='Include trade history')
@click.option('--backtests/--no-backtests', default=True, help='Include backtest results')
@click.option('--paper/--no-paper', default=True, help='Include paper trading sessions')
@click.option('--strategies/--no-strategies', default=True, help='Include strategies')
async def populate(seed, market_data, trades, backtests, paper, strategies):
    """Populate database with test data"""
    generator = TestDataGenerator(seed=seed)
    
    async with AsyncSessionLocal() as db:
        click.echo("Populating database with test data...")
        
        await generator.populate_database(
            db,
            include_market_data=market_data,
            include_trades=trades,
            include_backtests=backtests,
            include_paper_trading=paper,
            include_strategies=strategies
        )
        
        click.echo("Test data populated successfully!")

@cli.command()
@click.option('--table', help='Specific table to clear')
async def clear(table):
    """Clear test data from database"""
    async with AsyncSessionLocal() as db:
        if table:
            click.echo(f"Clearing table: {table}")
            await db.execute(text(f"TRUNCATE TABLE {table} CASCADE"))
        else:
            click.echo("Clearing all test data...")
            # Clear in order to respect foreign key constraints
            tables = [
                'backtest_trades', 'backtest_results', 'backtests',
                'paper_positions', 'paper_orders', 'paper_trading_sessions',
                'strategy_executions', 'strategy_parameters', 'strategies',
                'positions', 'orders', 'trades',
                'kimchi_premium', 'orderbooks', 'market_data',
                'notifications', 'alerts'
            ]
            
            for table_name in tables:
                try:
                    await db.execute(text(f"TRUNCATE TABLE {table_name} CASCADE"))
                    click.echo(f"  Cleared: {table_name}")
                except Exception as e:
                    click.echo(f"  Skipped {table_name}: {e}")
        
        await db.commit()
        click.echo("Data cleared successfully!")

@cli.command()
@click.option('--symbol', default='BTC/USDT', help='Trading symbol')
@click.option('--hours', default=1, help='Hours of data to generate')
async def generate_live_data(symbol, hours):
    """Generate live-like streaming data for testing"""
    generator = TestDataGenerator()
    
    click.echo(f"Generating live data for {symbol} over {hours} hour(s)...")
    
    async with AsyncSessionLocal() as db:
        # Generate data points every 5 seconds
        interval_seconds = 5
        num_points = (hours * 3600) // interval_seconds
        
        for i in range(num_points):
            # Generate single data point
            market_data = generator.generate_market_data(
                symbol=symbol,
                exchange='binance',
                hours=0.001,  # Very short timeframe
                interval_seconds=1
            )[0]
            
            db.add(market_data)
            
            if i % 12 == 0:  # Every minute
                await db.commit()
                click.echo(f"  Generated {i+1}/{num_points} data points")
            
            await asyncio.sleep(interval_seconds)
        
        await db.commit()
        click.echo("Live data generation completed!")

@cli.command()
async def stats():
    """Show database statistics"""
    async with AsyncSessionLocal() as db:
        stats_queries = {
            'Strategies': "SELECT COUNT(*) FROM strategies",
            'Trades': "SELECT COUNT(*) FROM trades",
            'Market Data': "SELECT COUNT(*) FROM market_data",
            'Kimchi Premium': "SELECT COUNT(*) FROM kimchi_premium",
            'Backtests': "SELECT COUNT(*) FROM backtests",
            'Paper Sessions': "SELECT COUNT(*) FROM paper_trading_sessions",
            'Paper Orders': "SELECT COUNT(*) FROM paper_orders",
            'Alerts': "SELECT COUNT(*) FROM alerts"
        }
        
        click.echo("\nDatabase Statistics:")
        click.echo("-" * 40)
        
        for table_name, query in stats_queries.items():
            try:
                result = await db.execute(text(query))
                count = result.scalar()
                click.echo(f"{table_name:20} {count:>10} records")
            except Exception as e:
                click.echo(f"{table_name:20} {'N/A':>10} (table not found)")
        
        click.echo("-" * 40)

@cli.command()
@click.option('--num-trades', default=50, help='Number of trades to generate')
@click.option('--session-id', default=1, help='Paper trading session ID')
async def generate_paper_trades(num_trades, session_id):
    """Generate paper trading activity for a session"""
    generator = TestDataGenerator()
    
    async with AsyncSessionLocal() as db:
        click.echo(f"Generating {num_trades} paper trades for session {session_id}...")
        
        from app.models.paper_trading import PaperOrder
        import random
        from datetime import datetime
        
        symbols = ['BTC/USDT', 'ETH/USDT', 'XRP/USDT']
        
        for i in range(num_trades):
            symbol = random.choice(symbols)
            side = random.choice(['buy', 'sell'])
            
            order = PaperOrder(
                session_id=session_id,
                symbol=symbol,
                side=side,
                order_type=random.choice(['market', 'limit']),
                quantity=random.uniform(0.001, 1.0),
                price=random.uniform(40000, 60000) if 'BTC' in symbol else random.uniform(2000, 4000),
                status='executed',
                exchange=random.choice(['upbit', 'binance']),
                executed_at=datetime.utcnow()
            )
            
            db.add(order)
            
            if i % 10 == 0:
                await db.commit()
                click.echo(f"  Generated {i+1}/{num_trades} orders")
        
        await db.commit()
        click.echo("Paper trades generated successfully!")

def main():
    """Main entry point for async CLI"""
    # Handle async commands
    if len(sys.argv) > 1:
        command = sys.argv[1]
        async_commands = [
            'init-db', 'populate', 'clear', 'generate-live-data',
            'stats', 'generate-paper-trades'
        ]
        
        if command in async_commands:
            # Run async command
            asyncio.run(cli())
        else:
            # Run sync command
            cli()
    else:
        cli()

if __name__ == '__main__':
    main()