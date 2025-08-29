"""
Performance Analyzer for Backtesting
백테스팅 성과 분석
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

from backtesting.backtest_engine import OrderSide

logger = logging.getLogger(__name__)


class PerformanceAnalyzer:
    """
    성과 분석기
    - Sharpe Ratio, Calmar Ratio 계산
    - Maximum Drawdown 분석
    - 거래별 통계
    """
    
    def __init__(self, portfolio_history, trades: List):
        """
        Args:
            portfolio_history: 포트폴리오 히스토리 (List or DataFrame)
            trades: 거래 기록
        """
        self.portfolio_history = portfolio_history
        self.trades = trades
        
    def calculate_returns(self) -> pd.Series:
        """수익률 계산"""
        # Handle both DataFrame and List
        if isinstance(self.portfolio_history, pd.DataFrame):
            if self.portfolio_history.empty:
                return pd.Series()
            if 'value' in self.portfolio_history.columns:
                values = self.portfolio_history['value']
            else:
                return pd.Series()
        else:
            if self.portfolio_history.empty:
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
            sharpe = 0.0
            
        return float(sharpe)
    
    def calculate_calmar_ratio(self) -> float:
        """
        Calmar Ratio 계산 (연수익률 / 최대낙폭)
        
        Returns:
            Calmar Ratio
        """
        # Handle both DataFrame and List
        if isinstance(self.portfolio_history, pd.DataFrame):
            if self.portfolio_history.empty:
                return float(0)
            if 'value' not in self.portfolio_history.columns:
                return float(0)
            
            initial_value = self.portfolio_history['value'].iloc[0]
            final_value = self.portfolio_history['value'].iloc[-1]
            total_return = (final_value - initial_value) / initial_value
            
            # 기간 (일) - DataFrame의 경우 인덱스 사용
            if 'timestamp' in self.portfolio_history.columns:
                days = (self.portfolio_history['timestamp'].iloc[-1] - 
                       self.portfolio_history['timestamp'].iloc[0]).days
            else:
                days = len(self.portfolio_history)
        else:
            if self.portfolio_history.empty:
                return float(0)
            
            # 총 수익률
            initial_value = self.portfolio_history.iloc[0]['value']
            final_value = self.portfolio_history.iloc[-1]['value']
            total_return = (final_value - initial_value) / initial_value
            
            # 기간 (일)
            days = (self.portfolio_history.iloc[-1]['timestamp'] - 
                    self.portfolio_history.iloc[0]['timestamp']).days
        
        # 연율화 수익률
        if days > 0:
            annual_return = (1 + total_return) ** (365 / days) - 1
        else:
            annual_return = 0.0
        
        # 최대 낙폭
        max_dd = self.calculate_max_drawdown()
        
        # Calmar Ratio
        if max_dd != 0:
            calmar = annual_return / abs(max_dd)  # abs() 사용하여 양수로 변환
        else:
            calmar = 0.0  # inf 대신 0.0 반환 (CI 테스트 호환, float 타입)
            
        return float(calmar)
    
    def calculate_max_drawdown(self) -> float:
        """
        최대 낙폭 계산
        
        Returns:
            최대 낙폭 (비율)
        """
        # Handle both DataFrame and List
        if isinstance(self.portfolio_history, pd.DataFrame):
            if self.portfolio_history.empty:
                return 0
            if 'value' in self.portfolio_history.columns:
                values = self.portfolio_history['value'].tolist()
            else:
                return 0
        else:
            if self.portfolio_history.empty:
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
        
        return -max_dd  # Return negative value for drawdown
    
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
        # portfolio_history가 비어있으면 trades에서 계산
        if isinstance(self.portfolio_history, pd.DataFrame) and self.portfolio_history.empty:
            if self.trades:
                gross_profit = 0
                gross_loss = 0
                for trade in self.trades:
                    if hasattr(trade, 'pnl'):
                        if trade.pnl > 0:
                            gross_profit += trade.pnl
                        else:
                            gross_loss += abs(trade.pnl)
                    else:
                        # pnl이 없으면 price * amount로 추정
                        value = trade.price * trade.amount
                        if trade.side == OrderSide.SELL:
                            gross_profit += value
                return gross_profit / gross_loss if gross_loss > 0 else float('inf')
            return 0.0
        
        gross_profit = 0
        gross_loss = 0
        
        for i in range(1, len(self.portfolio_history)):
            pnl = (self.portfolio_history.iloc[i]['value'] - 
                   self.portfolio_history.iloc[i-1]['value'])
            
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
        if self.portfolio_history.empty:
            return {}
        
        initial_value = self.portfolio_history.iloc[0]['value']
        final_value = self.portfolio_history.iloc[-1]['value']
        total_return = (final_value - initial_value) / initial_value
        
        # 기간
        start_date = self.portfolio_history.iloc[0]['timestamp']
        end_date = self.portfolio_history.iloc[-1]['timestamp']
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
        
        # 추가 필수 필드들
        if self.trades:
            # 베스트/워스트 트레이드 계산 (간단한 구현)
            summary['best_trade'] = 0  # TODO: 실제 PnL 계산 필요
            summary['worst_trade'] = 0  # TODO: 실제 PnL 계산 필요
            summary['avg_trade'] = 0  # TODO: 실제 PnL 계산 필요
        else:
            summary['best_trade'] = 0
            summary['worst_trade'] = 0
            summary['avg_trade'] = 0
        
        return summary
    
    def get_monthly_returns(self) -> pd.Series:
        """
        월별 수익률 계산
        
        Returns:
            월별 수익률 시리즈
        """
        if self.portfolio_history.empty:
            return pd.Series()
        
        # DataFrame에 월 정보 추가
        df = self.portfolio_history.copy()
        df['month'] = pd.to_datetime(df['timestamp']).dt.to_period('M')
        
        # 월별 첫날과 마지막날 값 계산
        monthly = df.groupby('month')['value'].agg(['first', 'last'])
        monthly['return'] = (monthly['last'] - monthly['first']) / monthly['first']
        
        return monthly['return']
    
    def get_trade_analysis(self) -> Dict:
        """
        거래 분석
        
        Returns:
            거래 분석 결과
        """
        if not self.trades:
            return {
                'by_exchange': {},
                'by_side': {},
                'by_hour': {}
            }
        
        analysis = {
            'by_exchange': {},
            'by_side': {'BUY': {'count': 0, 'total_pnl': 0}, 'SELL': {'count': 0, 'total_pnl': 0}},
            'by_hour': {}
        }
        
        # 거래소별 분석
        for trade in self.trades:
            exchange = trade.exchange
            if exchange not in analysis['by_exchange']:
                analysis['by_exchange'][exchange] = {'count': 0, 'total_pnl': 0}
            analysis['by_exchange'][exchange]['count'] += 1
            
            # side별 분석
            side = trade.side.value.upper()
            if side in analysis['by_side']:
                analysis['by_side'][side]['count'] += 1
        
        return analysis
    
    def get_risk_metrics(self) -> Dict:
        """
        리스크 메트릭 반환
        
        Returns:
            리스크 지표들
        """
        returns = self.calculate_returns()
        
        # VaR 계산 (95%, 99%)
        if len(returns) > 0:
            var_95 = float(np.percentile(returns, 5))
            var_99 = float(np.percentile(returns, 1))
            
            # CVaR 계산
            cvar_95 = float(returns[returns <= var_95].mean()) if len(returns[returns <= var_95]) > 0 else 0
            cvar_99 = float(returns[returns <= var_99].mean()) if len(returns[returns <= var_99]) > 0 else 0
            
            # Downside deviation
            negative_returns = returns[returns < 0]
            downside_deviation = float(negative_returns.std() * np.sqrt(252)) if len(negative_returns) > 0 else 0
            
            # Sortino ratio
            if downside_deviation > 0:
                sortino_ratio = float(returns.mean() / downside_deviation * np.sqrt(252))
            else:
                sortino_ratio = 0
                
            # Information ratio (assuming benchmark return = 0)
            if returns.std() > 0:
                information_ratio = float(returns.mean() / returns.std() * np.sqrt(252))
            else:
                information_ratio = 0
                
            # Upside potential ratio
            positive_returns = returns[returns > 0]
            if downside_deviation > 0 and len(positive_returns) > 0:
                upside_potential_ratio = float(positive_returns.mean() / downside_deviation)
            else:
                upside_potential_ratio = 0
        else:
            var_95 = var_99 = cvar_95 = cvar_99 = 0
            downside_deviation = sortino_ratio = information_ratio = upside_potential_ratio = 0
        
        return {
            'value_at_risk_95': var_95,
            'value_at_risk_99': var_99,
            'conditional_var_95': cvar_95,
            'conditional_var_99': cvar_99,
            'sortino_ratio': sortino_ratio,
            'information_ratio': information_ratio,
            'downside_deviation': downside_deviation,
            'upside_potential_ratio': upside_potential_ratio
        }
    
    def generate_report_data(self) -> Dict:
        """
        리포트 생성을 위한 데이터 반환
        
        Returns:
            리포트 데이터
        """
        return {
            'summary': self.get_performance_summary(),
            'risk_metrics': self.get_risk_metrics(),
            'trade_analysis': self.get_trade_analysis(),
            'monthly_returns': self.get_monthly_returns().to_dict() if not self.portfolio_history.empty else {}
        }