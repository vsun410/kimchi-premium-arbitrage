"""
Walk-Forward Analysis 구현
과적합 방지를 위한 동적 재학습 백테스팅
"""

import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass
import logging

# 프로젝트 루트 경로 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from backtesting.engine import BacktestingEngine, BacktestResult
from strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardWindow:
    """Walk-forward 윈도우 정의"""
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    window_id: int


@dataclass
class WalkForwardResult:
    """Walk-forward 분석 결과"""
    windows: List[WalkForwardWindow]
    window_results: List[BacktestResult]
    combined_result: BacktestResult
    in_sample_performance: Dict[str, float]
    out_of_sample_performance: Dict[str, float]
    overfitting_score: float  # 0-1, 낮을수록 좋음


class WalkForwardAnalysis:
    """
    Walk-Forward Analysis
    주기적으로 전략을 재학습하며 백테스트 수행
    """
    
    def __init__(
        self,
        strategy: BaseStrategy,
        backtest_engine: BacktestingEngine,
        train_period_days: int = 30,  # 학습 기간 (일)
        test_period_days: int = 7,     # 테스트 기간 (일)
        overlap_days: int = 0,         # 윈도우 간 중첩 기간
        min_train_samples: int = 500   # 최소 학습 샘플 수
    ):
        """
        Walk-Forward Analysis 초기화
        
        Args:
            strategy: 테스트할 전략
            backtest_engine: 백테스팅 엔진
            train_period_days: 학습 기간 (일)
            test_period_days: 테스트 기간 (일)
            overlap_days: 윈도우 간 중첩 기간
            min_train_samples: 최소 학습 샘플 수
        """
        self.strategy = strategy
        self.backtest_engine = backtest_engine
        self.train_period = timedelta(days=train_period_days)
        self.test_period = timedelta(days=test_period_days)
        self.overlap = timedelta(days=overlap_days)
        self.min_train_samples = min_train_samples
        
        self.windows: List[WalkForwardWindow] = []
        self.window_results: List[BacktestResult] = []
        
        logger.info(
            f"WalkForwardAnalysis initialized: "
            f"train={train_period_days}d, test={test_period_days}d, overlap={overlap_days}d"
        )
    
    def run(
        self,
        data: pd.DataFrame,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> WalkForwardResult:
        """
        Walk-forward analysis 실행
        
        Args:
            data: 전체 데이터
            start_date: 분석 시작일
            end_date: 분석 종료일
            
        Returns:
            WalkForwardResult: 분석 결과
        """
        # 데이터 필터링
        if start_date:
            data = data[data.index >= start_date]
        if end_date:
            data = data[data.index <= end_date]
        
        # 윈도우 생성
        self.windows = self._create_windows(data)
        
        if not self.windows:
            raise ValueError("Walk-forward 윈도우를 생성할 수 없습니다. 데이터가 부족합니다.")
        
        logger.info(f"Created {len(self.windows)} walk-forward windows")
        
        # 각 윈도우에서 백테스트 실행
        self.window_results = []
        in_sample_metrics = []
        out_of_sample_metrics = []
        
        for i, window in enumerate(self.windows):
            logger.info(f"Processing window {i+1}/{len(self.windows)}")
            
            # 학습 데이터
            train_data = data[
                (data.index >= window.train_start) & 
                (data.index <= window.train_end)
            ]
            
            # 테스트 데이터
            test_data = data[
                (data.index >= window.test_start) & 
                (data.index <= window.test_end)
            ]
            
            # 전략 학습 (지원하는 경우)
            if hasattr(self.strategy, 'train'):
                logger.debug(f"Training strategy on {len(train_data)} samples")
                self.strategy.train(train_data)
            
            # In-sample 성능 (학습 데이터)
            in_sample_result = self.backtest_engine.run(
                train_data,
                window.train_start,
                window.train_end
            )
            in_sample_metrics.append({
                'return': in_sample_result.total_return,
                'sharpe': in_sample_result.sharpe_ratio,
                'win_rate': in_sample_result.win_rate
            })
            
            # Out-of-sample 성능 (테스트 데이터)
            out_sample_result = self.backtest_engine.run(
                test_data,
                window.test_start,
                window.test_end
            )
            out_of_sample_metrics.append({
                'return': out_sample_result.total_return,
                'sharpe': out_sample_result.sharpe_ratio,
                'win_rate': out_sample_result.win_rate
            })
            
            self.window_results.append(out_sample_result)
        
        # 결과 통합
        combined_result = self._combine_results(self.window_results)
        
        # 과적합 점수 계산
        overfitting_score = self._calculate_overfitting_score(
            in_sample_metrics,
            out_of_sample_metrics
        )
        
        # 평균 성능 계산
        avg_in_sample = {
            'avg_return': np.mean([m['return'] for m in in_sample_metrics]),
            'avg_sharpe': np.mean([m['sharpe'] for m in in_sample_metrics]),
            'avg_win_rate': np.mean([m['win_rate'] for m in in_sample_metrics])
        }
        
        avg_out_sample = {
            'avg_return': np.mean([m['return'] for m in out_of_sample_metrics]),
            'avg_sharpe': np.mean([m['sharpe'] for m in out_of_sample_metrics]),
            'avg_win_rate': np.mean([m['win_rate'] for m in out_of_sample_metrics])
        }
        
        result = WalkForwardResult(
            windows=self.windows,
            window_results=self.window_results,
            combined_result=combined_result,
            in_sample_performance=avg_in_sample,
            out_of_sample_performance=avg_out_sample,
            overfitting_score=overfitting_score
        )
        
        return result
    
    def _create_windows(self, data: pd.DataFrame) -> List[WalkForwardWindow]:
        """
        Walk-forward 윈도우 생성
        
        Args:
            data: 전체 데이터
            
        Returns:
            List[WalkForwardWindow]: 윈도우 리스트
        """
        windows = []
        window_id = 0
        
        # 시작점 설정
        data_start = data.index[0]
        data_end = data.index[-1]
        
        current_train_start = data_start
        
        while current_train_start < data_end:
            # 학습 기간
            train_end = current_train_start + self.train_period
            
            # 테스트 기간
            test_start = train_end
            test_end = test_start + self.test_period
            
            # 데이터 범위 확인
            if test_end > data_end:
                break
            
            # 학습 데이터 크기 확인
            train_data_count = len(data[
                (data.index >= current_train_start) & 
                (data.index <= train_end)
            ])
            
            if train_data_count < self.min_train_samples:
                logger.warning(
                    f"Skipping window {window_id}: insufficient training samples "
                    f"({train_data_count} < {self.min_train_samples})"
                )
                current_train_start = test_start - self.overlap
                continue
            
            window = WalkForwardWindow(
                train_start=current_train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                window_id=window_id
            )
            
            windows.append(window)
            window_id += 1
            
            # 다음 윈도우 시작점
            current_train_start = test_start - self.overlap
        
        return windows
    
    def _combine_results(self, window_results: List[BacktestResult]) -> BacktestResult:
        """
        여러 윈도우의 결과를 통합
        
        Args:
            window_results: 각 윈도우의 백테스트 결과
            
        Returns:
            BacktestResult: 통합된 결과
        """
        if not window_results:
            return BacktestResult(initial_capital=self.backtest_engine.initial_capital)
        
        # 모든 거래 통합
        all_trades = []
        for result in window_results:
            all_trades.extend(result.trades)
        
        # Equity curve 통합
        equity_curves = []
        for result in window_results:
            if not result.equity_curve.empty:
                equity_curves.append(result.equity_curve)
        
        if equity_curves:
            combined_equity = pd.concat(equity_curves).sort_index()
        else:
            combined_equity = pd.Series()
        
        # 통합 메트릭 계산
        total_return = np.mean([r.total_return for r in window_results])
        total_trades = sum([r.total_trades for r in window_results])
        winning_trades = sum([r.winning_trades for r in window_results])
        losing_trades = sum([r.losing_trades for r in window_results])
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # 리스크 메트릭 평균
        avg_sharpe = np.mean([r.sharpe_ratio for r in window_results])
        avg_sortino = np.mean([r.sortino_ratio for r in window_results])
        avg_calmar = np.mean([r.calmar_ratio for r in window_results])
        max_drawdown = min([r.max_drawdown for r in window_results])
        
        combined_result = BacktestResult(
            total_return=total_return,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            sharpe_ratio=avg_sharpe,
            sortino_ratio=avg_sortino,
            calmar_ratio=avg_calmar,
            max_drawdown=max_drawdown,
            trades=all_trades,
            equity_curve=combined_equity,
            initial_capital=self.backtest_engine.initial_capital
        )
        
        return combined_result
    
    def _calculate_overfitting_score(
        self,
        in_sample: List[Dict],
        out_sample: List[Dict]
    ) -> float:
        """
        과적합 점수 계산
        In-sample과 Out-of-sample 성능 차이를 기반으로 계산
        
        Args:
            in_sample: In-sample 메트릭
            out_sample: Out-of-sample 메트릭
            
        Returns:
            float: 과적합 점수 (0-1, 낮을수록 좋음)
        """
        if not in_sample or not out_sample:
            return 0.0
        
        # 각 메트릭별 성능 차이 계산
        return_diff = []
        sharpe_diff = []
        win_rate_diff = []
        
        for i_s, o_s in zip(in_sample, out_sample):
            # 수익률 차이
            if i_s['return'] != 0:
                return_diff.append(abs(i_s['return'] - o_s['return']) / abs(i_s['return']))
            
            # Sharpe ratio 차이
            if i_s['sharpe'] != 0:
                sharpe_diff.append(abs(i_s['sharpe'] - o_s['sharpe']) / abs(i_s['sharpe']))
            
            # 승률 차이
            if i_s['win_rate'] != 0:
                win_rate_diff.append(abs(i_s['win_rate'] - o_s['win_rate']) / i_s['win_rate'])
        
        # 평균 차이 계산
        avg_return_diff = np.mean(return_diff) if return_diff else 0
        avg_sharpe_diff = np.mean(sharpe_diff) if sharpe_diff else 0
        avg_win_diff = np.mean(win_rate_diff) if win_rate_diff else 0
        
        # 가중 평균으로 최종 점수 계산
        overfitting_score = (
            avg_return_diff * 0.4 +
            avg_sharpe_diff * 0.4 +
            avg_win_diff * 0.2
        )
        
        # 0-1 범위로 정규화
        overfitting_score = min(max(overfitting_score, 0), 1)
        
        return overfitting_score
    
    def generate_report(self, result: WalkForwardResult, output_path: str = None):
        """
        Walk-forward 분석 리포트 생성
        
        Args:
            result: Walk-forward 분석 결과
            output_path: 리포트 저장 경로
        """
        report = {
            "Walk-Forward Analysis 요약": {
                "윈도우 수": len(result.windows),
                "학습 기간": f"{self.train_period.days}일",
                "테스트 기간": f"{self.test_period.days}일",
                "중첩 기간": f"{self.overlap.days}일",
            },
            "In-Sample 성능": {
                "평균 수익률": f"{result.in_sample_performance['avg_return']:.2%}",
                "평균 Sharpe": f"{result.in_sample_performance['avg_sharpe']:.2f}",
                "평균 승률": f"{result.in_sample_performance['avg_win_rate']:.2%}",
            },
            "Out-of-Sample 성능": {
                "평균 수익률": f"{result.out_of_sample_performance['avg_return']:.2%}",
                "평균 Sharpe": f"{result.out_of_sample_performance['avg_sharpe']:.2f}",
                "평균 승률": f"{result.out_of_sample_performance['avg_win_rate']:.2%}",
            },
            "과적합 평가": {
                "과적합 점수": f"{result.overfitting_score:.3f}",
                "평가": self._interpret_overfitting_score(result.overfitting_score),
            },
            "통합 결과": {
                "총 수익률": f"{result.combined_result.total_return:.2%}",
                "Sharpe Ratio": f"{result.combined_result.sharpe_ratio:.2f}",
                "최대 낙폭": f"{result.combined_result.max_drawdown:.2%}",
                "총 거래": result.combined_result.total_trades,
            }
        }
        
        # 콘솔 출력
        print("\n" + "=" * 60)
        print("Walk-Forward Analysis 리포트")
        print("=" * 60)
        
        for section, metrics in report.items():
            print(f"\n[{section}]")
            for key, value in metrics.items():
                print(f"  {key}: {value}")
        
        # 파일 저장
        if output_path:
            import json
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            print(f"\n리포트 저장: {output_path}")
        
        return report
    
    def _interpret_overfitting_score(self, score: float) -> str:
        """
        과적합 점수 해석
        
        Args:
            score: 과적합 점수 (0-1)
            
        Returns:
            str: 해석 결과
        """
        if score < 0.2:
            return "매우 좋음 (과적합 위험 낮음)"
        elif score < 0.4:
            return "좋음 (약간의 과적합 가능성)"
        elif score < 0.6:
            return "보통 (과적합 주의 필요)"
        elif score < 0.8:
            return "나쁨 (과적합 가능성 높음)"
        else:
            return "매우 나쁨 (심각한 과적합)"