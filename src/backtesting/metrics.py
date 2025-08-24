"""
Performance Metrics for Backtesting
Phase 3: 성과 평가 지표
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from src.utils.logger import logger


class PerformanceMetrics:
    """
    백테스트 성과 평가
    
    Metrics:
    - Sharpe Ratio
    - Calmar Ratio
    - Maximum Drawdown
    - Win Rate
    - Profit Factor
    """
    
    def __init__(self, results: Dict):
        """
        초기화
        
        Args:
            results: 백테스트 결과
        """
        self.results = results
        self.trades = pd.DataFrame(results.get('trades', []))
        self.equity_curve = pd.DataFrame(results.get('equity_curve', []))
        
    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """
        Sharpe Ratio 계산
        
        Args:
            risk_free_rate: 무위험 수익률 (연율)
            
        Returns:
            Sharpe Ratio
        """
        if len(self.equity_curve) < 2:
            return 0.0
        
        # 일일 수익률
        returns = self.equity_curve['return'].pct_change().dropna()
        
        if returns.std() == 0:
            return 0.0
        
        # 연율화
        excess_return = returns.mean() - risk_free_rate / 252
        sharpe = excess_return / returns.std() * np.sqrt(252)
        
        return sharpe
    
    def calculate_calmar_ratio(self) -> float:
        """
        Calmar Ratio 계산 (연수익률 / 최대낙폭)
        
        Returns:
            Calmar Ratio
        """
        if len(self.equity_curve) < 2:
            return 0.0
        
        # 연수익률
        total_return = self.results.get('total_return', 0)
        days = len(self.equity_curve)
        annual_return = total_return * (365 / days) if days > 0 else 0
        
        # 최대 낙폭
        max_dd = abs(self.results.get('max_drawdown', 0))
        
        if max_dd == 0:
            return 0.0
        
        return annual_return / max_dd
    
    def calculate_profit_factor(self) -> float:
        """
        Profit Factor 계산 (총이익 / 총손실)
        
        Returns:
            Profit Factor
        """
        if len(self.trades) == 0:
            return 0.0
        
        profits = self.trades[self.trades['pnl'] > 0]['pnl'].sum()
        losses = abs(self.trades[self.trades['pnl'] < 0]['pnl'].sum())
        
        if losses == 0:
            return float('inf') if profits > 0 else 0.0
        
        return profits / losses
    
    def calculate_recovery_factor(self) -> float:
        """
        Recovery Factor 계산 (순이익 / 최대낙폭)
        
        Returns:
            Recovery Factor
        """
        net_profit = self.results.get('final_capital', 0) - 40_000_000  # 초기자본
        max_dd_amount = abs(self.results.get('max_drawdown', 0)) * 40_000_000 / 100
        
        if max_dd_amount == 0:
            return 0.0
        
        return net_profit / max_dd_amount
    
    def calculate_win_statistics(self) -> Dict:
        """
        승률 통계 계산
        
        Returns:
            승률 관련 통계
        """
        if len(self.trades) == 0:
            return {
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'avg_win_loss_ratio': 0,
                'max_consecutive_wins': 0,
                'max_consecutive_losses': 0
            }
        
        wins = self.trades[self.trades['pnl'] > 0]
        losses = self.trades[self.trades['pnl'] < 0]
        
        # 연속 승/패 계산
        is_win = (self.trades['pnl'] > 0).astype(int)
        consecutive_wins = (is_win.groupby((is_win != is_win.shift()).cumsum()).cumsum() * is_win)
        consecutive_losses = ((1-is_win).groupby(((1-is_win) != (1-is_win).shift()).cumsum()).cumsum() * (1-is_win))
        
        avg_loss = losses['pnl'].mean() if len(losses) > 0 else 0
        
        return {
            'win_rate': len(wins) / len(self.trades) * 100,
            'avg_win': wins['pnl'].mean() if len(wins) > 0 else 0,
            'avg_loss': avg_loss,
            'avg_win_loss_ratio': (wins['pnl'].mean() / abs(avg_loss)) if avg_loss != 0 else 0,
            'max_consecutive_wins': consecutive_wins.max(),
            'max_consecutive_losses': consecutive_losses.max()
        }
    
    def calculate_risk_metrics(self) -> Dict:
        """
        리스크 지표 계산
        
        Returns:
            리스크 관련 지표
        """
        if len(self.equity_curve) < 2:
            return {
                'volatility': 0,
                'downside_deviation': 0,
                'var_95': 0,
                'cvar_95': 0
            }
        
        returns = self.equity_curve['return'].pct_change().dropna()
        
        # 하방 변동성
        downside_returns = returns[returns < 0]
        downside_dev = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        
        # VaR (95%)
        var_95 = np.percentile(returns, 5) if len(returns) > 0 else 0
        
        # CVaR (95%)
        cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else 0
        
        return {
            'volatility': returns.std() * np.sqrt(252),
            'downside_deviation': downside_dev,
            'var_95': var_95,
            'cvar_95': cvar_95
        }
    
    def generate_report(self) -> Dict:
        """
        종합 성과 리포트 생성
        
        Returns:
            성과 리포트
        """
        report = {
            'summary': {
                'total_trades': self.results.get('total_trades', 0),
                'total_return': self.results.get('total_return', 0),
                'final_capital': self.results.get('final_capital', 0),
                'max_drawdown': self.results.get('max_drawdown', 0)
            },
            'returns': {
                'sharpe_ratio': self.calculate_sharpe_ratio(),
                'calmar_ratio': self.calculate_calmar_ratio(),
                'profit_factor': self.calculate_profit_factor(),
                'recovery_factor': self.calculate_recovery_factor()
            },
            'win_statistics': self.calculate_win_statistics(),
            'risk_metrics': self.calculate_risk_metrics()
        }
        
        return report
    
    def print_report(self):
        """
        리포트 출력
        """
        report = self.generate_report()
        
        print("\n" + "=" * 60)
        print("  BACKTEST PERFORMANCE REPORT")
        print("=" * 60)
        
        print("\n[Summary]")
        for key, value in report['summary'].items():
            if 'capital' in key:
                print(f"  {key}: {value:,.0f} KRW")
            elif 'return' in key or 'drawdown' in key:
                print(f"  {key}: {value:.2f}%")
            else:
                print(f"  {key}: {value}")
        
        print("\n[Returns]")
        for key, value in report['returns'].items():
            print(f"  {key}: {value:.2f}")
        
        print("\n[Win Statistics]")
        for key, value in report['win_statistics'].items():
            if 'rate' in key:
                print(f"  {key}: {value:.2f}%")
            elif 'avg' in key and 'ratio' not in key:
                print(f"  {key}: {value:,.0f} KRW")
            else:
                print(f"  {key}: {value:.2f}")
        
        print("\n[Risk Metrics]")
        for key, value in report['risk_metrics'].items():
            print(f"  {key}: {value:.4f}")
        
        print("\n" + "=" * 60)
    
    def plot_results(self, save_path: Optional[str] = None):
        """
        결과 시각화
        
        Args:
            save_path: 저장 경로
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Equity Curve
        if len(self.equity_curve) > 0:
            axes[0, 0].plot(self.equity_curve.index, self.equity_curve['equity'])
            axes[0, 0].set_title('Equity Curve')
            axes[0, 0].set_xlabel('Time')
            axes[0, 0].set_ylabel('Capital (KRW)')
            axes[0, 0].grid(True)
        
        # 2. Drawdown
        if len(self.equity_curve) > 0:
            cummax = self.equity_curve['equity'].cummax()
            drawdown = (self.equity_curve['equity'] - cummax) / cummax * 100
            axes[0, 1].fill_between(self.equity_curve.index, drawdown, 0, alpha=0.3, color='red')
            axes[0, 1].set_title('Drawdown')
            axes[0, 1].set_xlabel('Time')
            axes[0, 1].set_ylabel('Drawdown (%)')
            axes[0, 1].grid(True)
        
        # 3. Trade Distribution
        if len(self.trades) > 0:
            axes[1, 0].hist(self.trades['pnl'], bins=30, edgecolor='black')
            axes[1, 0].axvline(x=0, color='red', linestyle='--')
            axes[1, 0].set_title('Trade P&L Distribution')
            axes[1, 0].set_xlabel('P&L (KRW)')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].grid(True)
        
        # 4. Cumulative Returns
        if len(self.trades) > 0:
            cumulative_pnl = self.trades['pnl'].cumsum()
            axes[1, 1].plot(range(len(cumulative_pnl)), cumulative_pnl)
            axes[1, 1].set_title('Cumulative P&L')
            axes[1, 1].set_xlabel('Trade Number')
            axes[1, 1].set_ylabel('Cumulative P&L (KRW)')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Results plot saved to {save_path}")
        
        plt.show()
    
    def save_results(self, filepath: str):
        """
        결과 저장
        
        Args:
            filepath: 저장 경로
        """
        report = self.generate_report()
        
        # DataFrame으로 변환
        report_df = pd.DataFrame([report['summary']])
        report_df = pd.concat([
            report_df,
            pd.DataFrame([report['returns']]),
            pd.DataFrame([report['win_statistics']]),
            pd.DataFrame([report['risk_metrics']])
        ], axis=1)
        
        # CSV 저장
        report_df.to_csv(filepath, index=False)
        logger.info(f"Results saved to {filepath}")
        
        # 거래 내역 저장
        if len(self.trades) > 0:
            trades_file = filepath.replace('.csv', '_trades.csv')
            self.trades.to_csv(trades_file, index=False)
            logger.info(f"Trades saved to {trades_file}")