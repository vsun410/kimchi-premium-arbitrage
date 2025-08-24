"""
성과 분석 모듈
백테스트 결과를 상세히 분석하고 시각화
"""

import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import json
import logging
from dataclasses import asdict

# 프로젝트 루트 경로 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from backtesting.engine import BacktestResult, Trade

logger = logging.getLogger(__name__)


class PerformanceAnalyzer:
    """
    백테스트 성과 분석기
    다양한 메트릭과 시각화 제공
    """
    
    def __init__(self):
        """성과 분석기 초기화"""
        self.metrics = {}
        self.analysis_results = {}
    
    def analyze(self, result: BacktestResult) -> Dict:
        """
        종합적인 성과 분석 수행
        
        Args:
            result: 백테스트 결과
            
        Returns:
            Dict: 분석 결과
        """
        # 기본 메트릭
        basic_metrics = self._calculate_basic_metrics(result)
        
        # 리스크 메트릭
        risk_metrics = self._calculate_risk_metrics(result)
        
        # 거래 패턴 분석
        trade_patterns = self._analyze_trade_patterns(result)
        
        # 월별 성과
        monthly_performance = self._calculate_monthly_performance(result)
        
        # 최적/최악 거래
        best_worst_trades = self._find_best_worst_trades(result)
        
        # Rolling 메트릭
        rolling_metrics = self._calculate_rolling_metrics(result)
        
        # 종합 분석 결과
        self.analysis_results = {
            "basic_metrics": basic_metrics,
            "risk_metrics": risk_metrics,
            "trade_patterns": trade_patterns,
            "monthly_performance": monthly_performance,
            "best_worst_trades": best_worst_trades,
            "rolling_metrics": rolling_metrics,
            "strategy_score": self._calculate_strategy_score(result)
        }
        
        return self.analysis_results
    
    def _calculate_basic_metrics(self, result: BacktestResult) -> Dict:
        """
        기본 성과 메트릭 계산
        
        Args:
            result: 백테스트 결과
            
        Returns:
            Dict: 기본 메트릭
        """
        # 연율화 수익률
        if result.start_date and result.end_date:
            days = (result.end_date - result.start_date).days
            if days > 0:
                annualized_return = (1 + result.total_return) ** (365 / days) - 1
            else:
                annualized_return = 0
        else:
            annualized_return = 0
        
        # 거래당 평균 수익
        avg_profit_per_trade = 0
        if result.total_trades > 0:
            total_profit = result.final_capital - result.initial_capital
            avg_profit_per_trade = total_profit / result.total_trades
        
        # 승패 비율
        win_loss_ratio = 0
        if result.average_loss != 0:
            win_loss_ratio = abs(result.average_win / result.average_loss)
        
        # 기대값
        expectancy = (
            result.win_rate * result.average_win +
            (1 - result.win_rate) * result.average_loss
        )
        
        return {
            "총 수익률": f"{result.total_return:.2%}",
            "연율화 수익률": f"{annualized_return:.2%}",
            "총 거래 횟수": result.total_trades,
            "승률": f"{result.win_rate:.2%}",
            "승패 비율": f"{win_loss_ratio:.2f}",
            "Profit Factor": f"{result.profit_factor:.2f}",
            "거래당 평균 수익": f"{avg_profit_per_trade:,.0f} KRW",
            "기대값": f"{expectancy:,.0f} KRW",
        }
    
    def _calculate_risk_metrics(self, result: BacktestResult) -> Dict:
        """
        리스크 메트릭 계산
        
        Args:
            result: 백테스트 결과
            
        Returns:
            Dict: 리스크 메트릭
        """
        # Recovery Factor
        recovery_factor = 0
        if result.max_drawdown < 0:
            recovery_factor = result.total_return / abs(result.max_drawdown)
        
        # Risk-Reward Ratio
        risk_reward_ratio = 0
        if result.max_drawdown < 0:
            risk_reward_ratio = result.total_return / abs(result.max_drawdown)
        
        # Equity curve 분석
        if not result.equity_curve.empty:
            returns = result.equity_curve.pct_change().dropna()
            
            # 변동성 (연율화)
            volatility = returns.std() * np.sqrt(365 * 24)
            
            # 왜도와 첨도
            skewness = returns.skew()
            kurtosis = returns.kurtosis()
            
            # VaR (95% 신뢰수준)
            var_95 = np.percentile(returns, 5)
            
            # CVaR (Conditional VaR)
            cvar_95 = returns[returns <= var_95].mean()
        else:
            volatility = skewness = kurtosis = var_95 = cvar_95 = 0
        
        return {
            "Sharpe Ratio": f"{result.sharpe_ratio:.2f}",
            "Sortino Ratio": f"{result.sortino_ratio:.2f}",
            "Calmar Ratio": f"{result.calmar_ratio:.2f}",
            "최대 낙폭": f"{result.max_drawdown:.2%}",
            "최대 낙폭 기간": f"{result.max_drawdown_duration} 시간",
            "Recovery Factor": f"{recovery_factor:.2f}",
            "Risk-Reward Ratio": f"{risk_reward_ratio:.2f}",
            "변동성 (연율화)": f"{volatility:.2%}",
            "왜도": f"{skewness:.2f}",
            "첨도": f"{kurtosis:.2f}",
            "VaR (95%)": f"{var_95:.2%}",
            "CVaR (95%)": f"{cvar_95:.2%}",
        }
    
    def _analyze_trade_patterns(self, result: BacktestResult) -> Dict:
        """
        거래 패턴 분석
        
        Args:
            result: 백테스트 결과
            
        Returns:
            Dict: 거래 패턴 분석
        """
        if not result.trades:
            return {"message": "거래 내역이 없습니다"}
        
        # 거래 시간 분석
        trade_hours = {}
        trade_days = {}
        
        for trade in result.trades:
            hour = trade.timestamp.hour
            day = trade.timestamp.strftime("%A")
            
            trade_hours[hour] = trade_hours.get(hour, 0) + 1
            trade_days[day] = trade_days.get(day, 0) + 1
        
        # 가장 활발한 시간과 요일
        most_active_hour = max(trade_hours, key=trade_hours.get) if trade_hours else None
        most_active_day = max(trade_days, key=trade_days.get) if trade_days else None
        
        # 연속 승/패 분석
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        current_wins = 0
        current_losses = 0
        
        # 매수/매도 쌍을 매칭하여 승패 계산
        buy_trades = [t for t in result.trades if t.side == 'buy']
        sell_trades = [t for t in result.trades if t.side == 'sell']
        
        for i in range(min(len(buy_trades), len(sell_trades))):
            pnl = (sell_trades[i].price - buy_trades[i].price) * buy_trades[i].amount
            pnl -= (buy_trades[i].fee + sell_trades[i].fee)
            
            if pnl > 0:
                current_wins += 1
                current_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, current_losses)
        
        # 평균 보유 시간
        holding_times = []
        for i in range(min(len(buy_trades), len(sell_trades))):
            holding_time = (sell_trades[i].timestamp - buy_trades[i].timestamp).total_seconds() / 3600
            holding_times.append(holding_time)
        
        avg_holding_time = np.mean(holding_times) if holding_times else 0
        
        return {
            "가장 활발한 시간": f"{most_active_hour}시" if most_active_hour else "N/A",
            "가장 활발한 요일": most_active_day if most_active_day else "N/A",
            "최대 연속 승리": max_consecutive_wins,
            "최대 연속 패배": max_consecutive_losses,
            "평균 보유 시간": f"{avg_holding_time:.1f} 시간",
            "매수 거래": len(buy_trades),
            "매도 거래": len(sell_trades),
        }
    
    def _calculate_monthly_performance(self, result: BacktestResult) -> Dict:
        """
        월별 성과 계산
        
        Args:
            result: 백테스트 결과
            
        Returns:
            Dict: 월별 성과
        """
        if result.equity_curve.empty:
            return {"message": "Equity curve가 없습니다"}
        
        # 월별 수익률 계산
        monthly_returns = result.equity_curve.resample('M').last().pct_change().dropna()
        
        if monthly_returns.empty:
            return {"message": "월별 데이터가 부족합니다"}
        
        # 월별 통계
        best_month = monthly_returns.max()
        worst_month = monthly_returns.min()
        avg_monthly = monthly_returns.mean()
        positive_months = (monthly_returns > 0).sum()
        total_months = len(monthly_returns)
        
        # 월별 상세 내역
        monthly_details = {}
        for date, ret in monthly_returns.items():
            month_str = date.strftime("%Y-%m")
            monthly_details[month_str] = f"{ret:.2%}"
        
        return {
            "최고 월 수익률": f"{best_month:.2%}",
            "최악 월 수익률": f"{worst_month:.2%}",
            "평균 월 수익률": f"{avg_monthly:.2%}",
            "수익 월 비율": f"{positive_months}/{total_months} ({positive_months/total_months:.1%})",
            "월별 상세": monthly_details
        }
    
    def _find_best_worst_trades(self, result: BacktestResult) -> Dict:
        """
        최고/최악 거래 찾기
        
        Args:
            result: 백테스트 결과
            
        Returns:
            Dict: 최고/최악 거래 정보
        """
        if not result.trades:
            return {"message": "거래 내역이 없습니다"}
        
        # 매수/매도 쌍을 매칭하여 PnL 계산
        buy_trades = [t for t in result.trades if t.side == 'buy']
        sell_trades = [t for t in result.trades if t.side == 'sell']
        
        trade_pnls = []
        for i in range(min(len(buy_trades), len(sell_trades))):
            buy = buy_trades[i]
            sell = sell_trades[i]
            
            pnl = (sell.price - buy.price) * buy.amount
            pnl -= (buy.fee + sell.fee)
            
            trade_pnls.append({
                "buy_time": buy.timestamp,
                "sell_time": sell.timestamp,
                "buy_price": buy.price,
                "sell_price": sell.price,
                "amount": buy.amount,
                "pnl": pnl,
                "return": pnl / (buy.price * buy.amount)
            })
        
        if not trade_pnls:
            return {"message": "완료된 거래가 없습니다"}
        
        # 최고/최악 거래
        best_trade = max(trade_pnls, key=lambda x: x['pnl'])
        worst_trade = min(trade_pnls, key=lambda x: x['pnl'])
        
        return {
            "최고 거래": {
                "매수 시간": best_trade['buy_time'].strftime("%Y-%m-%d %H:%M"),
                "매도 시간": best_trade['sell_time'].strftime("%Y-%m-%d %H:%M"),
                "수익": f"{best_trade['pnl']:,.0f} KRW",
                "수익률": f"{best_trade['return']:.2%}"
            },
            "최악 거래": {
                "매수 시간": worst_trade['buy_time'].strftime("%Y-%m-%d %H:%M"),
                "매도 시간": worst_trade['sell_time'].strftime("%Y-%m-%d %H:%M"),
                "손실": f"{worst_trade['pnl']:,.0f} KRW",
                "손실률": f"{worst_trade['return']:.2%}"
            }
        }
    
    def _calculate_rolling_metrics(self, result: BacktestResult) -> Dict:
        """
        Rolling 메트릭 계산
        
        Args:
            result: 백테스트 결과
            
        Returns:
            Dict: Rolling 메트릭
        """
        if result.equity_curve.empty:
            return {"message": "Equity curve가 없습니다"}
        
        # Rolling 윈도우 설정 (30일)
        window = 30 * 24  # 시간 단위
        
        if len(result.equity_curve) < window:
            return {"message": "Rolling 계산을 위한 데이터가 부족합니다"}
        
        returns = result.equity_curve.pct_change().dropna()
        
        # Rolling Sharpe Ratio
        rolling_sharpe = (
            returns.rolling(window).mean() / 
            returns.rolling(window).std()
        ) * np.sqrt(365 * 24)
        
        # Rolling Max Drawdown
        rolling_max = result.equity_curve.rolling(window).max()
        rolling_dd = (result.equity_curve - rolling_max) / rolling_max
        
        return {
            "30일 Rolling Sharpe (현재)": f"{rolling_sharpe.iloc[-1]:.2f}" if not rolling_sharpe.empty else "N/A",
            "30일 Rolling Sharpe (평균)": f"{rolling_sharpe.mean():.2f}" if not rolling_sharpe.empty else "N/A",
            "30일 Rolling DD (현재)": f"{rolling_dd.iloc[-1]:.2%}" if not rolling_dd.empty else "N/A",
            "30일 Rolling DD (최대)": f"{rolling_dd.min():.2%}" if not rolling_dd.empty else "N/A",
        }
    
    def _calculate_strategy_score(self, result: BacktestResult) -> Dict:
        """
        전략 종합 점수 계산
        
        Args:
            result: 백테스트 결과
            
        Returns:
            Dict: 전략 점수
        """
        scores = {}
        
        # 수익성 점수 (0-100)
        profitability_score = min(max(result.total_return * 100, 0), 100)
        scores['수익성'] = profitability_score
        
        # 안정성 점수 (Sharpe Ratio 기반)
        stability_score = min(max(result.sharpe_ratio * 25, 0), 100)
        scores['안정성'] = stability_score
        
        # 일관성 점수 (승률 기반)
        consistency_score = result.win_rate * 100
        scores['일관성'] = consistency_score
        
        # 리스크 관리 점수 (최대 낙폭 기반)
        risk_score = max(100 + result.max_drawdown * 100, 0)
        scores['리스크 관리'] = risk_score
        
        # 종합 점수
        total_score = (
            profitability_score * 0.3 +
            stability_score * 0.3 +
            consistency_score * 0.2 +
            risk_score * 0.2
        )
        
        # 등급 부여
        if total_score >= 80:
            grade = "A (우수)"
        elif total_score >= 60:
            grade = "B (양호)"
        elif total_score >= 40:
            grade = "C (보통)"
        elif total_score >= 20:
            grade = "D (미흡)"
        else:
            grade = "F (부적합)"
        
        return {
            "수익성 점수": f"{profitability_score:.1f}/100",
            "안정성 점수": f"{stability_score:.1f}/100",
            "일관성 점수": f"{consistency_score:.1f}/100",
            "리스크 관리 점수": f"{risk_score:.1f}/100",
            "종합 점수": f"{total_score:.1f}/100",
            "전략 등급": grade
        }
    
    def generate_report(self, output_path: str = None):
        """
        종합 성과 분석 리포트 생성
        
        Args:
            output_path: 리포트 저장 경로
        """
        if not self.analysis_results:
            logger.warning("분석 결과가 없습니다. analyze() 메서드를 먼저 실행하세요.")
            return
        
        # 콘솔 출력
        print("\n" + "=" * 80)
        print("백테스트 성과 분석 리포트")
        print("=" * 80)
        
        for section, metrics in self.analysis_results.items():
            if isinstance(metrics, dict) and 'message' not in metrics:
                print(f"\n[{section.replace('_', ' ').title()}]")
                for key, value in metrics.items():
                    if isinstance(value, dict):
                        print(f"  {key}:")
                        for sub_key, sub_value in value.items():
                            print(f"    {sub_key}: {sub_value}")
                    else:
                        print(f"  {key}: {value}")
        
        # 파일 저장
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.analysis_results, f, ensure_ascii=False, indent=2, default=str)
            
            print(f"\n리포트 저장: {output_path}")
        
        return self.analysis_results