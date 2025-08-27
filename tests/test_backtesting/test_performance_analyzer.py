"""
Tests for PerformanceAnalyzer
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from backtesting.performance_analyzer import PerformanceAnalyzer
from backtesting.backtest_engine import Trade, OrderSide, PositionSide


class TestPerformanceAnalyzer:
    """Test suite for PerformanceAnalyzer"""
    
    @pytest.fixture
    def sample_portfolio_history(self):
        """Create sample portfolio history"""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1H')
        values = [40000000]
        
        # Simulate portfolio value changes
        for i in range(1, 100):
            change = np.random.randn() * 100000
            values.append(values[-1] + change)
        
        return pd.DataFrame({
            'timestamp': dates,
            'value': values,
            'cash_krw': [20000000] * 100,
            'cash_usd': [15000] * 100,
            'positions': [0] * 100
        })
    
    @pytest.fixture
    def sample_trades(self):
        """Create sample trades"""
        trades = []
        base_time = datetime(2024, 1, 1)
        
        # Generate some winning and losing trades
        for i in range(20):
            timestamp = base_time + timedelta(hours=i*5)
            
            if i % 3 == 0:  # Losing trade
                trades.append(Trade(
                    timestamp=timestamp,
                    exchange='upbit',
                    symbol='BTC',
                    side=OrderSide.BUY,
                    position_side=PositionSide.LONG,
                    amount=0.1,
                    price=100000000,
                    fee=10000
                ))
            else:  # Winning trade
                trades.append(Trade(
                    timestamp=timestamp,
                    exchange='upbit',
                    symbol='BTC',
                    side=OrderSide.SELL,
                    position_side=PositionSide.LONG,
                    amount=0.1,
                    price=101000000,
                    fee=10100
                ))
        
        return trades
    
    @pytest.fixture
    def analyzer(self, sample_portfolio_history, sample_trades):
        """Create PerformanceAnalyzer instance"""
        return PerformanceAnalyzer(sample_portfolio_history, sample_trades)
    
    def test_initialization(self, analyzer):
        """Test analyzer initialization"""
        assert analyzer.portfolio_history is not None
        assert analyzer.trades is not None
        assert len(analyzer.portfolio_history) == 100
        assert len(analyzer.trades) == 20
    
    def test_calculate_returns(self, analyzer):
        """Test returns calculation"""
        returns = analyzer.calculate_returns()
        
        assert isinstance(returns, pd.Series)
        assert len(returns) == 99  # One less than portfolio history
        assert returns.dtype == float
        
        # Check first return calculation
        initial_value = analyzer.portfolio_history.iloc[0]['value']
        next_value = analyzer.portfolio_history.iloc[1]['value']
        expected_return = (next_value - initial_value) / initial_value
        assert abs(returns.iloc[0] - expected_return) < 1e-10
    
    def test_calculate_sharpe_ratio(self, analyzer):
        """Test Sharpe ratio calculation"""
        sharpe = analyzer.calculate_sharpe_ratio()
        
        assert isinstance(sharpe, float)
        # Sharpe ratio should be finite
        assert not np.isnan(sharpe)
        assert not np.isinf(sharpe)
    
    def test_calculate_sharpe_ratio_zero_std(self):
        """Test Sharpe ratio with zero standard deviation"""
        # Create portfolio with no changes
        dates = pd.date_range(start='2024-01-01', periods=10, freq='1H')
        portfolio = pd.DataFrame({
            'timestamp': dates,
            'value': [40000000] * 10
        })
        
        analyzer = PerformanceAnalyzer(portfolio, [])
        sharpe = analyzer.calculate_sharpe_ratio()
        
        assert sharpe == 0.0
    
    def test_calculate_calmar_ratio(self, analyzer):
        """Test Calmar ratio calculation"""
        calmar = analyzer.calculate_calmar_ratio()
        
        assert isinstance(calmar, float)
        # Calmar should be finite
        assert not np.isnan(calmar)
        assert not np.isinf(calmar)
    
    def test_calculate_max_drawdown(self, analyzer):
        """Test max drawdown calculation"""
        max_dd = analyzer.calculate_max_drawdown()
        
        assert isinstance(max_dd, float)
        assert max_dd <= 0  # Drawdown should be negative or zero
        assert max_dd >= -100  # Cannot exceed -100%
    
    def test_calculate_max_drawdown_no_loss(self):
        """Test max drawdown with no losses"""
        # Create monotonically increasing portfolio
        dates = pd.date_range(start='2024-01-01', periods=10, freq='1H')
        values = [40000000 + i * 100000 for i in range(10)]
        
        portfolio = pd.DataFrame({
            'timestamp': dates,
            'value': values
        })
        
        analyzer = PerformanceAnalyzer(portfolio, [])
        max_dd = analyzer.calculate_max_drawdown()
        
        assert max_dd == 0.0
    
    def test_calculate_win_rate(self, analyzer):
        """Test win rate calculation"""
        win_rate = analyzer.calculate_win_rate()
        
        assert isinstance(win_rate, float)
        assert 0 <= win_rate <= 100
        
        # Win rate should be a valid percentage
        # Not checking exact value as trades don't have pnl tracking yet
    
    def test_calculate_win_rate_no_trades(self):
        """Test win rate with no trades"""
        analyzer = PerformanceAnalyzer(pd.DataFrame(), [])
        win_rate = analyzer.calculate_win_rate()
        
        assert win_rate == 0.0
    
    def test_calculate_profit_factor(self, analyzer):
        """Test profit factor calculation"""
        profit_factor = analyzer.calculate_profit_factor()
        
        assert isinstance(profit_factor, float)
        assert profit_factor >= 0
        
        # Profit factor should be valid
        # Not checking exact value as trades don't have pnl tracking yet
    
    def test_calculate_profit_factor_no_losses(self):
        """Test profit factor with no losses"""
        trades = [
            Trade(
                timestamp=datetime.now(),
                exchange='upbit',
                symbol='BTC',
                side=OrderSide.SELL,
                position_side=PositionSide.LONG,
                amount=0.1,
                price=100000000,
                fee=10000
            )
        ]
        
        analyzer = PerformanceAnalyzer(pd.DataFrame(), trades)
        profit_factor = analyzer.calculate_profit_factor()
        
        assert profit_factor == float('inf')
    
    def test_get_performance_summary(self, analyzer):
        """Test performance summary generation"""
        summary = analyzer.get_performance_summary()
        
        assert isinstance(summary, dict)
        
        # Check required fields
        required_fields = [
            'total_return', 'total_return_krw', 'sharpe_ratio',
            'calmar_ratio', 'max_drawdown', 'win_rate',
            'profit_factor', 'total_trades', 'total_fees',
            'best_trade', 'worst_trade', 'avg_trade',
            'monthly_return', 'monthly_return_krw',
            'initial_value', 'final_value'
        ]
        
        for field in required_fields:
            assert field in summary
            assert summary[field] is not None
    
    def test_get_monthly_returns(self, analyzer):
        """Test monthly returns calculation"""
        monthly_returns = analyzer.get_monthly_returns()
        
        assert isinstance(monthly_returns, pd.Series)
        assert monthly_returns.index.freq in ['M', 'ME', 'MS']
        
        # All values should be percentages
        assert all(-100 <= r <= 1000 for r in monthly_returns)
    
    def test_get_trade_analysis(self, analyzer):
        """Test trade analysis"""
        analysis = analyzer.get_trade_analysis()
        
        assert isinstance(analysis, dict)
        
        # Check structure
        assert 'by_exchange' in analysis
        assert 'by_side' in analysis
        assert 'by_hour' in analysis
        
        # Check exchange analysis
        assert 'upbit' in analysis['by_exchange']
        assert 'count' in analysis['by_exchange']['upbit']
        assert 'total_pnl' in analysis['by_exchange']['upbit']
        
        # Check side analysis
        assert 'BUY' in analysis['by_side']
        assert 'SELL' in analysis['by_side']
    
    def test_get_risk_metrics(self, analyzer):
        """Test risk metrics calculation"""
        metrics = analyzer.get_risk_metrics()
        
        assert isinstance(metrics, dict)
        
        # Check required risk metrics
        required_metrics = [
            'value_at_risk_95', 'value_at_risk_99',
            'conditional_var_95', 'conditional_var_99',
            'sortino_ratio', 'information_ratio',
            'downside_deviation', 'upside_potential_ratio'
        ]
        
        for metric in required_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))
    
    def test_generate_report_data(self, analyzer):
        """Test report data generation"""
        report_data = analyzer.generate_report_data()
        
        assert isinstance(report_data, dict)
        
        # Check main sections
        assert 'summary' in report_data
        assert 'risk_metrics' in report_data
        assert 'trade_analysis' in report_data
        assert 'monthly_returns' in report_data
        
        # Verify summary matches get_performance_summary
        summary = analyzer.get_performance_summary()
        assert report_data['summary']['total_return'] == summary['total_return']
        assert report_data['summary']['sharpe_ratio'] == summary['sharpe_ratio']