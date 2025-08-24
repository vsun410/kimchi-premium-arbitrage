"""
과적합 검증 모듈
백테스트 결과의 신뢰성을 검증하고 과적합 위험을 평가
"""

import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from scipy import stats
import logging

# 프로젝트 루트 경로 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from backtesting.engine import BacktestingEngine, BacktestResult
from strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class OverfittingValidator:
    """
    과적합 검증기
    다양한 통계적 방법으로 전략의 과적합 여부를 검증
    """
    
    def __init__(
        self,
        strategy: BaseStrategy,
        backtest_engine: BacktestingEngine,
        confidence_level: float = 0.95  # 신뢰수준
    ):
        """
        과적합 검증기 초기화
        
        Args:
            strategy: 검증할 전략
            backtest_engine: 백테스팅 엔진
            confidence_level: 통계적 신뢰수준
        """
        self.strategy = strategy
        self.backtest_engine = backtest_engine
        self.confidence_level = confidence_level
        
        self.validation_results = {}
    
    def validate(
        self,
        data: pd.DataFrame,
        n_splits: int = 5,
        test_size: float = 0.2
    ) -> Dict:
        """
        종합적인 과적합 검증 수행
        
        Args:
            data: 전체 데이터
            n_splits: 교차 검증 분할 수
            test_size: 테스트 데이터 비율
            
        Returns:
            Dict: 검증 결과
        """
        logger.info("Starting overfitting validation...")
        
        # 1. 교차 검증
        cv_results = self._cross_validation(data, n_splits)
        
        # 2. 몬테카를로 시뮬레이션
        monte_carlo_results = self._monte_carlo_simulation(data, n_simulations=100)
        
        # 3. 파라미터 민감도 분석
        sensitivity_results = self._parameter_sensitivity_analysis(data)
        
        # 4. 통계적 유의성 검정
        significance_results = self._statistical_significance_test(data)
        
        # 5. 시간 안정성 검증
        stability_results = self._temporal_stability_test(data)
        
        # 종합 점수 계산
        overfitting_score = self._calculate_overfitting_score(
            cv_results,
            monte_carlo_results,
            sensitivity_results,
            significance_results,
            stability_results
        )
        
        self.validation_results = {
            "cross_validation": cv_results,
            "monte_carlo": monte_carlo_results,
            "parameter_sensitivity": sensitivity_results,
            "statistical_significance": significance_results,
            "temporal_stability": stability_results,
            "overfitting_score": overfitting_score,
            "validation_summary": self._generate_summary(overfitting_score)
        }
        
        return self.validation_results
    
    def _cross_validation(self, data: pd.DataFrame, n_splits: int) -> Dict:
        """
        K-fold 교차 검증
        
        Args:
            data: 전체 데이터
            n_splits: 분할 수
            
        Returns:
            Dict: 교차 검증 결과
        """
        logger.info(f"Running {n_splits}-fold cross validation...")
        
        fold_size = len(data) // n_splits
        fold_results = []
        
        for i in range(n_splits):
            # 폴드 분할
            test_start = i * fold_size
            test_end = (i + 1) * fold_size if i < n_splits - 1 else len(data)
            
            # 테스트 데이터
            test_data = data.iloc[test_start:test_end]
            
            # 학습 데이터 (테스트 데이터 제외)
            train_data = pd.concat([
                data.iloc[:test_start],
                data.iloc[test_end:]
            ])
            
            # 전략 학습 (지원하는 경우)
            if hasattr(self.strategy, 'train'):
                self.strategy.train(train_data)
            
            # 테스트 데이터로 백테스트
            result = self.backtest_engine.run(test_data)
            
            fold_results.append({
                "fold": i + 1,
                "return": result.total_return,
                "sharpe": result.sharpe_ratio,
                "max_dd": result.max_drawdown,
                "win_rate": result.win_rate
            })
        
        # 통계 계산
        returns = [r['return'] for r in fold_results]
        sharpes = [r['sharpe'] for r in fold_results]
        
        cv_score = {
            "mean_return": np.mean(returns),
            "std_return": np.std(returns),
            "mean_sharpe": np.mean(sharpes),
            "std_sharpe": np.std(sharpes),
            "cv_coefficient": np.std(returns) / np.mean(returns) if np.mean(returns) != 0 else float('inf'),
            "fold_results": fold_results
        }
        
        return cv_score
    
    def _monte_carlo_simulation(self, data: pd.DataFrame, n_simulations: int) -> Dict:
        """
        몬테카를로 시뮬레이션
        랜덤 데이터로 전략 성능 비교
        
        Args:
            data: 전체 데이터
            n_simulations: 시뮬레이션 횟수
            
        Returns:
            Dict: 몬테카를로 결과
        """
        logger.info(f"Running {n_simulations} Monte Carlo simulations...")
        
        # 실제 전략 성능
        actual_result = self.backtest_engine.run(data)
        actual_return = actual_result.total_return
        actual_sharpe = actual_result.sharpe_ratio
        
        # 랜덤 시뮬레이션
        random_returns = []
        random_sharpes = []
        
        for i in range(n_simulations):
            # 랜덤 거래 신호 생성
            random_signals = np.random.choice([1, -1, 0], size=len(data))
            
            # 랜덤 백테스트 (간단한 시뮬레이션)
            random_equity = [self.backtest_engine.initial_capital]
            capital = self.backtest_engine.initial_capital
            
            for j, signal in enumerate(random_signals):
                if signal != 0 and j < len(data) - 1:
                    # 랜덤 거래 실행
                    price_change = data.iloc[j+1]['close'] / data.iloc[j]['close'] - 1
                    trade_return = signal * price_change * 0.1  # 10% 포지션
                    fee = abs(trade_return) * self.backtest_engine.maker_fee
                    capital = capital * (1 + trade_return - fee)
                    random_equity.append(capital)
            
            random_return = (capital - self.backtest_engine.initial_capital) / self.backtest_engine.initial_capital
            random_returns.append(random_return)
            
            # Sharpe ratio 계산
            if len(random_equity) > 1:
                equity_series = pd.Series(random_equity)
                returns_series = equity_series.pct_change().dropna()
                if returns_series.std() > 0:
                    random_sharpe = (returns_series.mean() / returns_series.std()) * np.sqrt(365 * 24)
                else:
                    random_sharpe = 0
            else:
                random_sharpe = 0
            
            random_sharpes.append(random_sharpe)
        
        # 통계 분석
        percentile_return = stats.percentileofscore(random_returns, actual_return)
        percentile_sharpe = stats.percentileofscore(random_sharpes, actual_sharpe)
        
        monte_carlo_results = {
            "actual_return": actual_return,
            "actual_sharpe": actual_sharpe,
            "random_mean_return": np.mean(random_returns),
            "random_std_return": np.std(random_returns),
            "random_mean_sharpe": np.mean(random_sharpes),
            "percentile_return": percentile_return,
            "percentile_sharpe": percentile_sharpe,
            "beats_random": percentile_return > 95  # 95% 이상이면 유의미
        }
        
        return monte_carlo_results
    
    def _parameter_sensitivity_analysis(self, data: pd.DataFrame) -> Dict:
        """
        파라미터 민감도 분석
        파라미터 변화에 따른 성능 변화 측정
        
        Args:
            data: 전체 데이터
            
        Returns:
            Dict: 민감도 분석 결과
        """
        logger.info("Running parameter sensitivity analysis...")
        
        # 전략 파라미터 가져오기 (예시)
        base_params = {}
        if hasattr(self.strategy, 'get_parameters'):
            base_params = self.strategy.get_parameters()
        else:
            # 기본 파라미터 사용
            base_params = {
                'lookback_period': 48,
                'entry_threshold': -0.02,
                'exit_threshold': 0.02
            }
        
        sensitivity_results = {}
        
        for param_name, base_value in base_params.items():
            param_results = []
            
            # 파라미터를 ±20% 범위에서 변경
            for multiplier in [0.8, 0.9, 1.0, 1.1, 1.2]:
                new_value = base_value * multiplier
                
                # 파라미터 업데이트 (지원하는 경우)
                if hasattr(self.strategy, 'set_parameter'):
                    self.strategy.set_parameter(param_name, new_value)
                
                # 백테스트 실행
                result = self.backtest_engine.run(data)
                
                param_results.append({
                    "multiplier": multiplier,
                    "value": new_value,
                    "return": result.total_return,
                    "sharpe": result.sharpe_ratio
                })
            
            # 원래 파라미터로 복원
            if hasattr(self.strategy, 'set_parameter'):
                self.strategy.set_parameter(param_name, base_value)
            
            # 민감도 계산
            returns = [r['return'] for r in param_results]
            sensitivity = np.std(returns) / np.mean(returns) if np.mean(returns) != 0 else float('inf')
            
            sensitivity_results[param_name] = {
                "base_value": base_value,
                "sensitivity": sensitivity,
                "results": param_results
            }
        
        # 전체 민감도 점수
        avg_sensitivity = np.mean([v['sensitivity'] for v in sensitivity_results.values()])
        
        return {
            "parameter_sensitivities": sensitivity_results,
            "average_sensitivity": avg_sensitivity,
            "is_stable": avg_sensitivity < 0.5  # 0.5 이하면 안정적
        }
    
    def _statistical_significance_test(self, data: pd.DataFrame) -> Dict:
        """
        통계적 유의성 검정
        전략 성능이 우연이 아님을 검증
        
        Args:
            data: 전체 데이터
            
        Returns:
            Dict: 유의성 검정 결과
        """
        logger.info("Running statistical significance tests...")
        
        # 전략 백테스트
        result = self.backtest_engine.run(data)
        
        # Buy-and-Hold 벤치마크
        initial_price = data.iloc[0]['close']
        final_price = data.iloc[-1]['close']
        buy_hold_return = (final_price - initial_price) / initial_price
        
        # 거래별 수익률
        trade_returns = []
        if result.trades:
            buy_trades = [t for t in result.trades if t.side == 'buy']
            sell_trades = [t for t in result.trades if t.side == 'sell']
            
            for i in range(min(len(buy_trades), len(sell_trades))):
                ret = (sell_trades[i].price - buy_trades[i].price) / buy_trades[i].price
                trade_returns.append(ret)
        
        # t-검정 (수익률이 0보다 유의미하게 큰지)
        if trade_returns:
            t_stat, p_value = stats.ttest_1samp(trade_returns, 0)
            is_significant = p_value < (1 - self.confidence_level)
        else:
            t_stat = p_value = 0
            is_significant = False
        
        # Information Ratio
        excess_return = result.total_return - buy_hold_return
        if result.equity_curve.std() > 0:
            information_ratio = excess_return / result.equity_curve.std()
        else:
            information_ratio = 0
        
        return {
            "strategy_return": result.total_return,
            "buy_hold_return": buy_hold_return,
            "excess_return": excess_return,
            "t_statistic": t_stat,
            "p_value": p_value,
            "is_significant": is_significant,
            "information_ratio": information_ratio,
            "beats_benchmark": result.total_return > buy_hold_return
        }
    
    def _temporal_stability_test(self, data: pd.DataFrame) -> Dict:
        """
        시간 안정성 검증
        다른 시기에도 일관된 성능을 보이는지 확인
        
        Args:
            data: 전체 데이터
            
        Returns:
            Dict: 시간 안정성 검증 결과
        """
        logger.info("Running temporal stability test...")
        
        # 데이터를 3개 기간으로 분할
        n_periods = 3
        period_size = len(data) // n_periods
        
        period_results = []
        
        for i in range(n_periods):
            start_idx = i * period_size
            end_idx = (i + 1) * period_size if i < n_periods - 1 else len(data)
            
            period_data = data.iloc[start_idx:end_idx]
            result = self.backtest_engine.run(period_data)
            
            period_results.append({
                "period": i + 1,
                "start": period_data.index[0],
                "end": period_data.index[-1],
                "return": result.total_return,
                "sharpe": result.sharpe_ratio,
                "win_rate": result.win_rate,
                "max_dd": result.max_drawdown
            })
        
        # 안정성 메트릭
        returns = [r['return'] for r in period_results]
        sharpes = [r['sharpe'] for r in period_results]
        win_rates = [r['win_rate'] for r in period_results]
        
        # 변동계수 (낮을수록 안정적)
        cv_return = np.std(returns) / abs(np.mean(returns)) if np.mean(returns) != 0 else float('inf')
        cv_sharpe = np.std(sharpes) / abs(np.mean(sharpes)) if np.mean(sharpes) != 0 else float('inf')
        
        # 모든 기간에서 양의 수익?
        all_positive = all(r > 0 for r in returns)
        
        return {
            "period_results": period_results,
            "cv_return": cv_return,
            "cv_sharpe": cv_sharpe,
            "all_periods_positive": all_positive,
            "is_stable": cv_return < 0.5 and all_positive
        }
    
    def _calculate_overfitting_score(
        self,
        cv_results: Dict,
        monte_carlo: Dict,
        sensitivity: Dict,
        significance: Dict,
        stability: Dict
    ) -> Dict:
        """
        종합 과적합 점수 계산
        
        Args:
            cv_results: 교차 검증 결과
            monte_carlo: 몬테카를로 결과
            sensitivity: 민감도 분석 결과
            significance: 유의성 검정 결과
            stability: 시간 안정성 결과
            
        Returns:
            Dict: 과적합 점수
        """
        scores = {}
        
        # 교차 검증 점수 (CV 계수가 낮을수록 좋음)
        cv_score = max(0, 100 - cv_results['cv_coefficient'] * 100)
        scores['cross_validation'] = cv_score
        
        # 몬테카를로 점수 (랜덤보다 좋은 성능)
        mc_score = monte_carlo['percentile_return']
        scores['monte_carlo'] = mc_score
        
        # 파라미터 안정성 점수
        param_score = 100 if sensitivity['is_stable'] else 50 - sensitivity['average_sensitivity'] * 50
        scores['parameter_stability'] = max(0, param_score)
        
        # 통계적 유의성 점수
        sig_score = 100 if significance['is_significant'] else 50
        scores['statistical_significance'] = sig_score
        
        # 시간 안정성 점수
        time_score = 100 if stability['is_stable'] else max(0, 100 - stability['cv_return'] * 100)
        scores['temporal_stability'] = time_score
        
        # 종합 점수 (가중 평균)
        total_score = (
            scores['cross_validation'] * 0.2 +
            scores['monte_carlo'] * 0.2 +
            scores['parameter_stability'] * 0.2 +
            scores['statistical_significance'] * 0.2 +
            scores['temporal_stability'] * 0.2
        )
        
        # 과적합 위험도 (0-100, 낮을수록 좋음)
        overfitting_risk = 100 - total_score
        
        return {
            "component_scores": scores,
            "total_score": total_score,
            "overfitting_risk": overfitting_risk,
            "risk_level": self._interpret_risk_level(overfitting_risk)
        }
    
    def _interpret_risk_level(self, risk: float) -> str:
        """
        과적합 위험도 해석
        
        Args:
            risk: 과적합 위험도 (0-100)
            
        Returns:
            str: 위험 수준
        """
        if risk < 20:
            return "매우 낮음 (신뢰할 수 있음)"
        elif risk < 40:
            return "낮음 (대체로 안전)"
        elif risk < 60:
            return "보통 (주의 필요)"
        elif risk < 80:
            return "높음 (과적합 가능성)"
        else:
            return "매우 높음 (과적합 위험)"
    
    def _generate_summary(self, overfitting_score: Dict) -> Dict:
        """
        검증 요약 생성
        
        Args:
            overfitting_score: 과적합 점수
            
        Returns:
            Dict: 검증 요약
        """
        risk_level = overfitting_score['overfitting_risk']
        
        recommendations = []
        
        if risk_level < 40:
            recommendations.append("전략이 안정적이며 실거래에 적합합니다.")
        else:
            recommendations.append("추가 검증이 필요합니다.")
            
            if overfitting_score['component_scores']['cross_validation'] < 60:
                recommendations.append("- 더 많은 데이터로 교차 검증 수행")
            
            if overfitting_score['component_scores']['parameter_stability'] < 60:
                recommendations.append("- 파라미터 최적화 범위 축소")
            
            if overfitting_score['component_scores']['temporal_stability'] < 60:
                recommendations.append("- 다양한 시장 조건에서 테스트")
        
        return {
            "risk_level": overfitting_score['risk_level'],
            "risk_score": f"{risk_level:.1f}%",
            "is_reliable": risk_level < 40,
            "recommendations": recommendations
        }
    
    def generate_report(self, output_path: str = None):
        """
        과적합 검증 리포트 생성
        
        Args:
            output_path: 리포트 저장 경로
        """
        if not self.validation_results:
            logger.warning("검증 결과가 없습니다. validate() 메서드를 먼저 실행하세요.")
            return
        
        # 콘솔 출력
        print("\n" + "=" * 60)
        print("과적합 검증 리포트")
        print("=" * 60)
        
        # 요약
        summary = self.validation_results['validation_summary']
        print(f"\n[검증 요약]")
        print(f"  과적합 위험: {summary['risk_level']}")
        print(f"  위험 점수: {summary['risk_score']}")
        print(f"  신뢰성: {'신뢰 가능' if summary['is_reliable'] else '추가 검증 필요'}")
        
        print(f"\n[권장사항]")
        for rec in summary['recommendations']:
            print(f"  {rec}")
        
        # 상세 점수
        scores = self.validation_results['overfitting_score']['component_scores']
        print(f"\n[상세 점수]")
        for component, score in scores.items():
            print(f"  {component}: {score:.1f}/100")
        
        # 파일 저장
        if output_path:
            import json
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.validation_results, f, ensure_ascii=False, indent=2, default=str)
            
            print(f"\n리포트 저장: {output_path}")
        
        return self.validation_results