"""
Performance Analyzer for Backtesting
백테스팅 성과 분석
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class PerformanceAnalyzer:
    """
    성과 분석기
    - Sharpe Ratio, Calmar Ratio 계산
    - Maximum Drawdown 분석
    - 거래별 통계
    """
    
    def __init__(self, portfolio_history: List, trades: List):
        """
        Args:
            portfolio_history: 포트폴리오 히스토리
            trades: 거래 기록
        """
        self.portfolio_history = portfolio_history
        self.trades = trades
        
    def calculate_returns(self) -> pd.Series:
        """수익률 계산"""
        if not self.portfolio_history:
            return pd.Series()
        
        # 포트폴리오 가치 시계열
        values = pd.Series(
            [p.total_value for p in self.portfolio_history],
            index=[p.timestamp for p in self.portfolio_history]
        )
        
        # 일일 수익률
        returns = values.pct_change().dropna()
        return returns
    
    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """
        Sharpe Ratio 계산
        
        Args:
            risk_free_rate: 무위험 수익률 (연율)
            
        Returns:
            Sharpe Ratio
        """
        returns = self.calculate_returns()
        
        if returns.empty:
            return 0
        
        # 일일 무위험 수익률
        daily_rf = risk_free_rate / 252
        
        # 초과 수익률
        excess_returns = returns - daily_rf
        
        # Sharpe Ratio (연율화)
        if excess_returns.std() > 0:
            sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
        else:
            sharpe = 0
            
        return sharpe
    
    def calculate_calmar_ratio(self) -> float:
        """
        Calmar Ratio 계산 (연수익률 / 최대낙폭)
        
        Returns:
            Calmar Ratio
        """
        if not self.portfolio_history:
            return 0
        
        # 총 수익률
        initial_value = self.portfolio_history[0].total_value
        final_value = self.portfolio_history[-1].total_value
        total_return = (final_value - initial_value) / initial_value
        
        # 기간 (일)
        days = (self.portfolio_history[-1].timestamp - 
                self.portfolio_history[0].timestamp).days
        
        # 연율화 수익률
        if days > 0:
            annual_return = (1 + total_return) ** (365 / days) - 1
        else:
            annual_return = 0
        
        # 최대 낙폭
        max_dd = self.calculate_max_drawdown()
        
        # Calmar Ratio
        if max_dd > 0:
            calmar = annual_return / max_dd
        else:
            calmar = 0 if annual_return <= 0 else float('inf')
            
        return calmar
    
    def calculate_max_drawdown(self) -> float:
        """
        최대 낙폭 계산
        
        Returns:
            최대 낙폭 (비율)
        """
        if not self.portfolio_history:
            return 0
        
        values = [p.total_value for p in self.portfolio_history]
        
        peak = values[0]
        max_dd = 0
        
        for value in values:
            if value > peak:
                peak = value
            else:
                dd = (peak - value) / peak
                max_dd = max(max_dd, dd)
        
        return max_dd
    
    def calculate_win_rate(self) -> float:
        """승률 계산"""
        if not self.trades:
            return 0
        
        # 매수/매도 쌍 찾기
        winning_trades = 0
        total_trades = 0
        
        # 거래를 포지션별로 그룹화
        positions = {}
        for trade in self.trades:
            key = f"{trade.exchange}_{trade.symbol}"
            if key not in positions:
                positions[key] = []
            positions[key].append(trade)
        
        # 각 포지션의 손익 계산
        for key, trades in positions.items():
            # 진입과 청산 찾기
            entries = [t for t in trades if t.side.value == 'buy']
            exits = [t for t in trades if t.side.value == 'sell']
            
            for exit_trade in exits:
                # 해당 청산에 대한 평균 진입가 계산
                if entries:
                    avg_entry = sum(e.price * e.amount for e in entries) / sum(e.amount for e in entries)
                    pnl = (exit_trade.price - avg_entry) * exit_trade.amount
                    
                    if pnl > 0:
                        winning_trades += 1
                    total_trades += 1
        
        return winning_trades / total_trades if total_trades > 0 else 0
    
    def calculate_profit_factor(self) -> float:
        """
        Profit Factor 계산 (총 이익 / 총 손실)
        
        Returns:
            Profit Factor
        """
        if not self.portfolio_history:
            return 0
        
        gross_profit = 0
        gross_loss = 0
        
        for i in range(1, len(self.portfolio_history)):
            pnl = (self.portfolio_history[i].total_value - 
                   self.portfolio_history[i-1].total_value)
            
            if pnl > 0:
                gross_profit += pnl
            else:
                gross_loss += abs(pnl)
        
        return gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    def calculate_monthly_returns(self) -> pd.Series:
        """월별 수익률 계산"""
        returns = self.calculate_returns()
        
        if returns.empty:
            return pd.Series()
        
        # 월별 그룹화
        monthly = returns.resample('M').apply(
            lambda x: (1 + x).prod() - 1
        )
        
        return monthly
    
    def get_trade_statistics(self) -> Dict:
        """거래 통계"""
        if not self.trades:
            return {}
        
        stats = {
            'total_trades': len(self.trades),
            'buy_trades': sum(1 for t in self.trades if t.side.value == 'buy'),
            'sell_trades': sum(1 for t in self.trades if t.side.value == 'sell'),
            'total_fees': sum(t.fee for t in self.trades),
            'avg_trade_size': np.mean([t.amount for t in self.trades]),
            'total_volume': sum(t.value for t in self.trades)
        }
        
        # 거래소별 통계
        upbit_trades = [t for t in self.trades if t.exchange == 'upbit']
        binance_trades = [t for t in self.trades if t.exchange == 'binance']
        
        stats['upbit_trades'] = len(upbit_trades)
        stats['binance_trades'] = len(binance_trades)
        
        return stats
    
    def get_performance_summary(self) -> Dict:
        """전체 성과 요약"""
        if not self.portfolio_history:
            return {}
        
        initial_value = self.portfolio_history[0].total_value
        final_value = self.portfolio_history[-1].total_value
        total_return = (final_value - initial_value) / initial_value
        
        # 기간
        start_date = self.portfolio_history[0].timestamp
        end_date = self.portfolio_history[-1].timestamp
        days = (end_date - start_date).days
        
        summary = {
            'initial_value': initial_value,
            'final_value': final_value,
            'total_return': total_return * 100,  # %
            'total_return_krw': final_value - initial_value,
            'sharpe_ratio': self.calculate_sharpe_ratio(),
            'calmar_ratio': self.calculate_calmar_ratio(),
            'max_drawdown': self.calculate_max_drawdown() * 100,  # %
            'win_rate': self.calculate_win_rate() * 100,  # %
            'profit_factor': self.calculate_profit_factor(),
            'trading_days': days,
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d')
        }
        
        # 월평균 수익률
        if days > 0:
            monthly_return = total_return * 30 / days
            summary['monthly_return'] = monthly_return * 100  # %
            summary['monthly_return_krw'] = initial_value * monthly_return
        
        # 거래 통계 추가
        trade_stats = self.get_trade_statistics()
        summary.update(trade_stats)
        
        return summary