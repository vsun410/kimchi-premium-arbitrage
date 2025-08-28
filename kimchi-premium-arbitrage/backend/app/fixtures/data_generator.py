"""
Test data generator for development and testing
"""
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import random
import json
from faker import Faker
import numpy as np
import pandas as pd

from app.models.market_data import MarketData, OrderBook, KimchiPremium
from app.models.trading import Trade, Order, Position
from app.models.backtesting import Backtest, BacktestResult, BacktestTrade
from app.models.paper_trading import PaperTradingSession, PaperOrder, PaperPosition
from app.models.strategy import Strategy, StrategyParameter, StrategyExecution
from sqlalchemy.ext.asyncio import AsyncSession

fake = Faker()

class TestDataGenerator:
    """Generate realistic test data for the system"""
    
    def __init__(self, seed: int = 42):
        """Initialize with optional random seed for reproducibility"""
        random.seed(seed)
        np.random.seed(seed)
        Faker.seed(seed)
        self.fake = fake
        
        # Common symbols
        self.symbols = ['BTC/USDT', 'ETH/USDT', 'XRP/USDT']
        self.exchanges = ['upbit', 'binance']
    
    def generate_price_series(
        self,
        start_price: float,
        num_points: int,
        volatility: float = 0.02,
        trend: float = 0.0001
    ) -> List[float]:
        """
        Generate realistic price series with random walk
        
        Args:
            start_price: Initial price
            num_points: Number of price points
            volatility: Price volatility (default 2%)
            trend: Trend direction and strength
            
        Returns:
            List of prices
        """
        prices = [start_price]
        
        for _ in range(num_points - 1):
            change = np.random.normal(trend, volatility)
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, start_price * 0.5))  # Prevent negative prices
        
        return prices
    
    def generate_market_data(
        self,
        symbol: str = 'BTC/USDT',
        exchange: str = 'binance',
        hours: int = 24,
        interval_seconds: int = 60
    ) -> List[MarketData]:
        """
        Generate market data for testing
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            hours: Number of hours of data
            interval_seconds: Interval between data points
            
        Returns:
            List of MarketData objects
        """
        num_points = (hours * 3600) // interval_seconds
        start_time = datetime.utcnow() - timedelta(hours=hours)
        
        # Generate base prices
        base_price = {
            'BTC/USDT': 50000.0,
            'ETH/USDT': 3000.0,
            'XRP/USDT': 0.5
        }.get(symbol, 100.0)
        
        prices = self.generate_price_series(base_price, num_points)
        
        market_data_list = []
        for i, price in enumerate(prices):
            timestamp = start_time + timedelta(seconds=i * interval_seconds)
            
            # Generate volume with some randomness
            base_volume = 1000000 if 'BTC' in symbol else 100000
            volume = base_volume * np.random.uniform(0.5, 2.0)
            
            # Generate bid/ask spread
            spread = price * 0.0005  # 0.05% spread
            
            market_data = MarketData(
                exchange=exchange,
                symbol=symbol,
                price=price,
                volume=volume,
                bid=price - spread / 2,
                ask=price + spread / 2,
                timestamp=timestamp
            )
            market_data_list.append(market_data)
        
        return market_data_list
    
    def generate_orderbook(
        self,
        symbol: str = 'BTC/USDT',
        exchange: str = 'binance',
        mid_price: float = 50000.0,
        depth: int = 20,
        spread_percent: float = 0.001
    ) -> OrderBook:
        """
        Generate realistic orderbook data
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            mid_price: Middle price
            depth: Number of levels
            spread_percent: Spread as percentage of price
            
        Returns:
            OrderBook object
        """
        spread = mid_price * spread_percent
        
        # Generate bids (buy orders)
        bids = []
        bid_price = mid_price - spread / 2
        for i in range(depth):
            price = bid_price - (i * mid_price * 0.0001)
            quantity = np.random.exponential(10) * (depth - i) / depth
            bids.append([price, quantity])
        
        # Generate asks (sell orders)
        asks = []
        ask_price = mid_price + spread / 2
        for i in range(depth):
            price = ask_price + (i * mid_price * 0.0001)
            quantity = np.random.exponential(10) * (depth - i) / depth
            asks.append([price, quantity])
        
        return OrderBook(
            exchange=exchange,
            symbol=symbol,
            bids=json.dumps(bids),
            asks=json.dumps(asks),
            timestamp=datetime.utcnow()
        )
    
    def generate_kimchi_premium_data(
        self,
        hours: int = 24,
        interval_minutes: int = 5
    ) -> List[KimchiPremium]:
        """
        Generate Kimchi premium data
        
        Args:
            hours: Number of hours of data
            interval_minutes: Interval between data points
            
        Returns:
            List of KimchiPremium objects
        """
        num_points = (hours * 60) // interval_minutes
        start_time = datetime.utcnow() - timedelta(hours=hours)
        
        premium_data_list = []
        
        for symbol in ['BTC', 'ETH', 'XRP']:
            # Base prices
            base_prices = {
                'BTC': {'upbit': 65000000, 'binance': 50000},
                'ETH': {'upbit': 3900000, 'binance': 3000},
                'XRP': {'upbit': 650, 'binance': 0.5}
            }
            
            for i in range(num_points):
                timestamp = start_time + timedelta(minutes=i * interval_minutes)
                
                # Generate realistic premium (usually 1-5%)
                base_premium = np.random.uniform(1, 5)
                premium_variation = np.sin(i / 10) * 2  # Add cyclical pattern
                premium_percentage = base_premium + premium_variation
                
                # USD/KRW rate with small variations
                usd_krw_rate = 1300 + np.random.uniform(-10, 10)
                
                # Calculate prices
                binance_price = base_prices[symbol]['binance'] * np.random.uniform(0.98, 1.02)
                upbit_price_usd = binance_price * (1 + premium_percentage / 100)
                upbit_price_krw = upbit_price_usd * usd_krw_rate
                
                premium_data = KimchiPremium(
                    symbol=symbol,
                    upbit_price_krw=upbit_price_krw,
                    binance_price_usdt=binance_price,
                    usd_krw_rate=usd_krw_rate,
                    premium_percentage=premium_percentage,
                    timestamp=timestamp
                )
                premium_data_list.append(premium_data)
        
        return premium_data_list
    
    def generate_trades(
        self,
        num_trades: int = 100,
        days_back: int = 7
    ) -> List[Trade]:
        """
        Generate trade history
        
        Args:
            num_trades: Number of trades to generate
            days_back: How many days of history
            
        Returns:
            List of Trade objects
        """
        trades = []
        start_time = datetime.utcnow() - timedelta(days=days_back)
        
        for i in range(num_trades):
            timestamp = start_time + timedelta(
                seconds=random.randint(0, days_back * 86400)
            )
            
            symbol = random.choice(self.symbols)
            side = random.choice(['buy', 'sell'])
            
            # Generate realistic trade data
            base_prices = {
                'BTC/USDT': 50000.0,
                'ETH/USDT': 3000.0,
                'XRP/USDT': 0.5
            }
            
            price = base_prices[symbol] * np.random.uniform(0.95, 1.05)
            quantity = np.random.exponential(1.0)
            
            # Calculate PnL (simplified)
            pnl = np.random.normal(0, price * quantity * 0.01)
            
            trade = Trade(
                strategy_id=random.randint(1, 5),
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=price,
                exchange=random.choice(self.exchanges),
                order_id=f"ORDER-{fake.uuid4()[:8]}",
                pnl=pnl,
                fees=price * quantity * 0.001,  # 0.1% fee
                created_at=timestamp
            )
            trades.append(trade)
        
        return trades
    
    def generate_backtest(
        self,
        name: Optional[str] = None,
        status: str = 'completed'
    ) -> Backtest:
        """
        Generate backtest data
        
        Args:
            name: Backtest name
            status: Backtest status
            
        Returns:
            Backtest object
        """
        if not name:
            name = f"Backtest {fake.word().capitalize()} Strategy"
        
        start_date = fake.date_time_between(start_date='-30d', end_date='-7d')
        end_date = fake.date_time_between(start_date='-6d', end_date='now')
        
        # Generate realistic metrics
        total_trades = random.randint(50, 500)
        win_rate = random.uniform(0.4, 0.7)
        
        backtest = Backtest(
            name=name,
            strategy_id=random.randint(1, 10),
            start_date=start_date,
            end_date=end_date,
            initial_capital=10000.0,
            parameters=json.dumps({
                'stop_loss': random.uniform(0.01, 0.05),
                'take_profit': random.uniform(0.02, 0.10),
                'position_size': random.uniform(0.1, 0.5)
            }),
            status=status,
            created_at=datetime.utcnow()
        )
        
        if status == 'completed':
            backtest.started_at = start_date
            backtest.completed_at = end_date
            
            # Add results
            total_pnl = random.uniform(-2000, 5000)
            backtest.results = json.dumps({
                'total_trades': total_trades,
                'winning_trades': int(total_trades * win_rate),
                'losing_trades': int(total_trades * (1 - win_rate)),
                'total_pnl': total_pnl,
                'roi': (total_pnl / 10000) * 100,
                'sharpe_ratio': random.uniform(-1, 3),
                'max_drawdown': random.uniform(-0.3, -0.05),
                'win_rate': win_rate * 100
            })
        
        return backtest
    
    def generate_paper_session(
        self,
        name: Optional[str] = None,
        active: bool = True
    ) -> PaperTradingSession:
        """
        Generate paper trading session
        
        Args:
            name: Session name
            active: Whether session is active
            
        Returns:
            PaperTradingSession object
        """
        if not name:
            name = f"Paper Trading {fake.company()}"
        
        initial_balance_krw = 20000000.0  # 20M KRW
        initial_balance_usd = 15000.0
        
        # Generate some trading activity
        total_trades = random.randint(0, 100)
        winning_trades = int(total_trades * random.uniform(0.4, 0.7))
        
        session = PaperTradingSession(
            name=name,
            description=fake.sentence(),
            initial_balance_krw=initial_balance_krw,
            initial_balance_usd=initial_balance_usd,
            current_balance_krw=initial_balance_krw * random.uniform(0.8, 1.3),
            current_balance_usd=initial_balance_usd * random.uniform(0.8, 1.3),
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=total_trades - winning_trades,
            total_pnl=random.uniform(-5000000, 10000000),
            total_fees=random.uniform(10000, 100000),
            is_active=active,
            created_at=fake.date_time_between(start_date='-30d', end_date='now')
        )
        
        return session
    
    def generate_strategy(
        self,
        name: Optional[str] = None,
        strategy_type: Optional[str] = None
    ) -> Strategy:
        """
        Generate strategy data
        
        Args:
            name: Strategy name
            strategy_type: Type of strategy
            
        Returns:
            Strategy object
        """
        if not name:
            name = f"{fake.word().capitalize()} {fake.word().capitalize()} Strategy"
        
        if not strategy_type:
            strategy_type = random.choice(['kimchi_arbitrage', 'momentum', 'mean_reversion'])
        
        strategy = Strategy(
            name=name,
            strategy_type=strategy_type,
            description=fake.paragraph(),
            parameters=json.dumps({
                'entry_threshold': random.uniform(2, 5),
                'exit_threshold': random.uniform(0.5, 2),
                'position_size': random.uniform(0.1, 0.5),
                'stop_loss': random.uniform(0.02, 0.05),
                'take_profit': random.uniform(0.05, 0.15),
                'max_positions': random.randint(1, 5)
            }),
            is_active=random.choice([True, False]),
            created_at=fake.date_time_between(start_date='-60d', end_date='now')
        )
        
        return strategy
    
    async def populate_database(
        self,
        db: AsyncSession,
        include_market_data: bool = True,
        include_trades: bool = True,
        include_backtests: bool = True,
        include_paper_trading: bool = True,
        include_strategies: bool = True
    ):
        """
        Populate database with test data
        
        Args:
            db: Database session
            include_market_data: Include market data
            include_trades: Include trade history
            include_backtests: Include backtest results
            include_paper_trading: Include paper trading sessions
            include_strategies: Include strategies
        """
        try:
            # Generate and add strategies
            if include_strategies:
                strategies = [
                    self.generate_strategy() for _ in range(10)
                ]
                for strategy in strategies:
                    db.add(strategy)
                print(f"Added {len(strategies)} strategies")
            
            # Generate and add market data
            if include_market_data:
                for symbol in self.symbols:
                    for exchange in self.exchanges:
                        market_data = self.generate_market_data(
                            symbol=symbol,
                            exchange=exchange,
                            hours=24
                        )
                        for data in market_data[:100]:  # Limit to 100 per combination
                            db.add(data)
                print(f"Added market data for {len(self.symbols)} symbols")
                
                # Add Kimchi premium data
                premium_data = self.generate_kimchi_premium_data(hours=24)
                for data in premium_data[:100]:
                    db.add(data)
                print(f"Added {len(premium_data[:100])} Kimchi premium records")
            
            # Generate and add trades
            if include_trades:
                trades = self.generate_trades(num_trades=100)
                for trade in trades:
                    db.add(trade)
                print(f"Added {len(trades)} trades")
            
            # Generate and add backtests
            if include_backtests:
                backtests = [
                    self.generate_backtest(status=status)
                    for status in ['completed', 'completed', 'failed', 'running', 'pending']
                ]
                for backtest in backtests:
                    db.add(backtest)
                print(f"Added {len(backtests)} backtests")
            
            # Generate and add paper trading sessions
            if include_paper_trading:
                sessions = [
                    self.generate_paper_session(active=active)
                    for active in [True, True, False]
                ]
                for session in sessions:
                    db.add(session)
                print(f"Added {len(sessions)} paper trading sessions")
            
            # Commit all changes
            await db.commit()
            print("Successfully populated database with test data")
            
        except Exception as e:
            await db.rollback()
            print(f"Error populating database: {e}")
            raise