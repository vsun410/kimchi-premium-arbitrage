"""
성과 메트릭 계산기
모든 평가 지표를 계산하는 통합 클래스
"""

import numpy as np
import pandas as pd
from typing import List, Union, Optional, Tuple
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class PerformanceMetrics:
    """
    포괄적인 성과 메트릭 계산기
    
    Sharpe, Sortino, Calmar, Information Ratio 등
    모든 주요 성과 지표 계산
    """
    
    def __init__(
        self,
        portfolio_values: Union[List[float], np.ndarray, pd.Series],
        benchmark_values: Optional[Union[List[float], np.ndarray]] = None,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252
    ):
        """
        메트릭 계산기 초기화
        
        Args:
            portfolio_values: 포트폴리오 가치 시계열
            benchmark_values: 벤치마크 가치 시계열 (optional)
            risk_free_rate: 연간 무위험 수익률
            periods_per_year: 연간 거래일 수
        """
        self.portfolio_values = np.array(portfolio_values)
        self.benchmark_values = np.array(benchmark_values) if benchmark_values else None
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year
        
        # 수익률 계산
        self.returns = self._calculate_returns()
        self.benchmark_returns = self._calculate_benchmark_returns()
        
    def _calculate_returns(self) -> np.ndarray:
        """수익률 계산"""
        if len(self.portfolio_values) < 2:
            return np.array([])
        
        returns = np.diff(self.portfolio_values) / self.portfolio_values[:-1]
        return returns[~np.isnan(returns)]
    
    def _calculate_benchmark_returns(self) -> Optional[np.ndarray]:
        """벤치마크 수익률 계산"""
        if self.benchmark_values is None or len(self.benchmark_values) < 2:
            return None
        
        returns = np.diff(self.benchmark_values) / self.benchmark_values[:-1]
        return returns[~np.isnan(returns)]
    
    # 수익성 메트릭
    def total_return(self) -> float:
        """총 수익률 (%)"""
        if len(self.portfolio_values) < 2:
            return 0.0
        
        return ((self.portfolio_values[-1] / self.portfolio_values[0]) - 1) * 100
    
    def annual_return(self) -> float:
        """연간 수익률 (%)"""
        if len(self.returns) == 0:
            return 0.0
        
        total_ret = self.total_return() / 100
        years = len(self.returns) / self.periods_per_year
        
        if years <= 0:
            return 0.0
        
        annual_ret = (1 + total_ret) ** (1 / years) - 1
        return annual_ret * 100
    
    def monthly_return(self) -> float:
        """월간 수익률 (%)"""
        annual_ret = self.annual_return() / 100
        monthly_ret = (1 + annual_ret) ** (1/12) - 1
        return monthly_ret * 100
    
    def daily_return(self) -> float:
        """일간 평균 수익률 (%)"""
        if len(self.returns) == 0:
            return 0.0
        return np.mean(self.returns) * 100
    
    # 리스크 메트릭
    def volatility(self) -> float:
        """변동성 (연간화, %)"""
        if len(self.returns) == 0:
            return 0.0
        
        daily_vol = np.std(self.returns)
        annual_vol = daily_vol * np.sqrt(self.periods_per_year)
        return annual_vol * 100
    
    def max_drawdown(self) -> float:
        """최대 낙폭 (%)"""
        if len(self.portfolio_values) < 2:
            return 0.0
        
        cumulative = np.maximum.accumulate(self.portfolio_values)
        drawdown = (self.portfolio_values - cumulative) / cumulative
        return np.min(drawdown) * 100
    
    def value_at_risk(self, confidence: float = 0.95) -> float:
        """Value at Risk (%)"""
        if len(self.returns) == 0:
            return 0.0
        
        var = np.percentile(self.returns, (1 - confidence) * 100)
        return var * 100
    
    def conditional_var(self, confidence: float = 0.95) -> float:
        """Conditional Value at Risk (CVaR) or Expected Shortfall (%)"""
        if len(self.returns) == 0:
            return 0.0
        
        var = self.value_at_risk(confidence) / 100
        cvar = np.mean(self.returns[self.returns <= var])
        return cvar * 100 if not np.isnan(cvar) else 0.0
    
    # 리스크 조정 메트릭
    def sharpe_ratio(self) -> float:
        """샤프 비율"""
        if len(self.returns) == 0 or np.std(self.returns) == 0:
            return 0.0
        
        excess_returns = self.returns - (self.risk_free_rate / self.periods_per_year)
        sharpe = np.mean(excess_returns) / np.std(excess_returns)
        return sharpe * np.sqrt(self.periods_per_year)
    
    def sortino_ratio(self, target_return: float = 0.0) -> float:
        """소르티노 비율"""
        if len(self.returns) == 0:
            return 0.0
        
        excess_returns = self.returns - (target_return / self.periods_per_year)
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            return float('inf') if np.mean(excess_returns) > 0 else 0.0
        
        downside_std = np.std(downside_returns)
        if downside_std == 0:
            return float('inf') if np.mean(excess_returns) > 0 else 0.0
        
        sortino = np.mean(excess_returns) / downside_std
        return sortino * np.sqrt(self.periods_per_year)
    
    def calmar_ratio(self) -> float:
        """칼마 비율 (연간 수익률 / 최대 낙폭)"""
        max_dd = abs(self.max_drawdown())
        
        if max_dd == 0:
            return float('inf') if self.annual_return() > 0 else 0.0
        
        return self.annual_return() / max_dd
    
    def information_ratio(self) -> float:
        """정보 비율 (벤치마크 대비)"""
        if self.benchmark_returns is None or len(self.benchmark_returns) == 0:
            return 0.0
        
        if len(self.returns) != len(self.benchmark_returns):
            # 길이 맞추기
            min_len = min(len(self.returns), len(self.benchmark_returns))
            returns = self.returns[:min_len]
            bench_returns = self.benchmark_returns[:min_len]
        else:
            returns = self.returns
            bench_returns = self.benchmark_returns
        
        active_returns = returns - bench_returns
        
        if np.std(active_returns) == 0:
            return 0.0
        
        ir = np.mean(active_returns) / np.std(active_returns)
        return ir * np.sqrt(self.periods_per_year)
    
    # 거래 효율 메트릭
    def win_rate(self) -> float:
        """승률 (%)"""
        if len(self.returns) == 0:
            return 0.0
        
        winning_trades = self.returns > 0
        return np.mean(winning_trades) * 100
    
    def profit_factor(self) -> float:
        """이익 계수 (총 이익 / 총 손실)"""
        if len(self.returns) == 0:
            return 0.0
        
        gains = self.returns[self.returns > 0]
        losses = abs(self.returns[self.returns < 0])
        
        total_gains = np.sum(gains)
        total_losses = np.sum(losses)
        
        if total_losses == 0:
            return float('inf') if total_gains > 0 else 0.0
        
        return total_gains / total_losses
    
    def avg_win(self) -> float:
        """평균 수익 거래 (%)"""
        if len(self.returns) == 0:
            return 0.0
        
        wins = self.returns[self.returns > 0]
        return np.mean(wins) * 100 if len(wins) > 0 else 0.0
    
    def avg_loss(self) -> float:
        """평균 손실 거래 (%)"""
        if len(self.returns) == 0:
            return 0.0
        
        losses = self.returns[self.returns < 0]
        return np.mean(losses) * 100 if len(losses) > 0 else 0.0
    
    def max_consecutive_wins(self) -> int:
        """최대 연속 수익 거래"""
        if len(self.returns) == 0:
            return 0
        
        wins = self.returns > 0
        max_streak = 0
        current_streak = 0
        
        for win in wins:
            if win:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        
        return max_streak
    
    def max_consecutive_losses(self) -> int:
        """최대 연속 손실 거래"""
        if len(self.returns) == 0:
            return 0
        
        losses = self.returns < 0
        max_streak = 0
        current_streak = 0
        
        for loss in losses:
            if loss:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        
        return max_streak
    
    def total_trades(self) -> int:
        """총 거래 수"""
        # 수익률이 0이 아닌 경우를 거래로 간주
        return np.sum(self.returns != 0)
    
    # 추가 메트릭
    def omega_ratio(self, threshold: float = 0.0) -> float:
        """오메가 비율"""
        if len(self.returns) == 0:
            return 0.0
        
        excess = self.returns - threshold
        gains = excess[excess > 0]
        losses = abs(excess[excess < 0])
        
        if np.sum(losses) == 0:
            return float('inf') if np.sum(gains) > 0 else 0.0
        
        return np.sum(gains) / np.sum(losses)
    
    def skewness(self) -> float:
        """왜도 (분포의 비대칭성)"""
        if len(self.returns) < 3:
            return 0.0
        
        return stats.skew(self.returns)
    
    def kurtosis(self) -> float:
        """첨도 (분포의 꼬리 두께)"""
        if len(self.returns) < 4:
            return 0.0
        
        return stats.kurtosis(self.returns)
    
    def recovery_factor(self) -> float:
        """회복 계수 (총 수익 / 최대 낙폭)"""
        max_dd = abs(self.max_drawdown())
        
        if max_dd == 0:
            return float('inf') if self.total_return() > 0 else 0.0
        
        return self.total_return() / max_dd
    
    def get_all_metrics(self) -> dict:
        """모든 메트릭 반환"""
        return {
            # 수익성
            'total_return': self.total_return(),
            'annual_return': self.annual_return(),
            'monthly_return': self.monthly_return(),
            'daily_return': self.daily_return(),
            
            # 리스크
            'volatility': self.volatility(),
            'max_drawdown': self.max_drawdown(),
            'var_95': self.value_at_risk(0.95),
            'cvar_95': self.conditional_var(0.95),
            
            # 리스크 조정
            'sharpe_ratio': self.sharpe_ratio(),
            'sortino_ratio': self.sortino_ratio(),
            'calmar_ratio': self.calmar_ratio(),
            'information_ratio': self.information_ratio(),
            
            # 거래 효율
            'win_rate': self.win_rate(),
            'profit_factor': self.profit_factor(),
            'avg_win': self.avg_win(),
            'avg_loss': self.avg_loss(),
            'max_consecutive_wins': self.max_consecutive_wins(),
            'max_consecutive_losses': self.max_consecutive_losses(),
            
            # 추가
            'omega_ratio': self.omega_ratio(),
            'skewness': self.skewness(),
            'kurtosis': self.kurtosis(),
            'recovery_factor': self.recovery_factor(),
            'total_trades': self.total_trades()
        }