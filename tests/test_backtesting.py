"""
백테스팅 시스템 테스트
백테스팅 엔진, Walk-forward, 성과 분석, 과적합 검증 테스트
"""

import os
import sys
import unittest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# 프로젝트 루트 경로 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from backtesting.engine import BacktestingEngine, BacktestResult, Trade
from backtesting.walk_forward import WalkForwardAnalysis
from backtesting.performance import PerformanceAnalyzer
from backtesting.validator import OverfittingValidator
from strategies.mean_reversion import MeanReversionStrategy


def create_sample_data(days=30, hourly_samples=24):
    """테스트용 샘플 데이터 생성"""
    dates = pd.date_range(
        start=datetime.now() - timedelta(days=days),
        periods=days * hourly_samples,
        freq='1H'
    )
    
    # 가격 데이터 생성 (랜덤 워크)
    price = 45000000  # 4500만원 (BTC 시작 가격)
    prices = [price]
    
    for _ in range(len(dates) - 1):
        change = np.random.randn() * 0.002  # 0.2% 표준편차
        price = price * (1 + change)
        prices.append(price)
    
    # OHLCV 데이터 생성
    data = pd.DataFrame({
        'open': prices,
        'high': [p * 1.001 for p in prices],
        'low': [p * 0.999 for p in prices],
        'close': prices,
        'volume': np.random.uniform(10, 100, len(dates)),
        'kimchi_premium': np.random.uniform(-0.02, 0.05, len(dates))  # -2% ~ 5% 김프
    }, index=dates)
    
    return data


class TestBacktestingEngine(unittest.TestCase):
    """BacktestingEngine 테스트"""
    
    def setUp(self):
        """테스트 설정"""
        self.strategy = MeanReversionStrategy()
        self.engine = BacktestingEngine(
            strategy=self.strategy,
            initial_capital=40_000_000,
            maker_fee=0.0002,
            taker_fee=0.0015,
            slippage_bps=10,
            use_maker_only=True
        )
        self.sample_data = create_sample_data(days=30)
    
    def test_engine_initialization(self):
        """엔진 초기화 테스트"""
        self.assertEqual(self.engine.initial_capital, 40_000_000)
        self.assertEqual(self.engine.maker_fee, 0.0002)
        self.assertEqual(self.engine.taker_fee, 0.0015)
        self.assertEqual(self.engine.slippage_rate, 0.001)
        self.assertTrue(self.engine.use_maker_only)
    
    def test_simple_backtest(self):
        """단순 백테스트 실행 테스트"""
        result = self.engine.run(self.sample_data)
        
        # 결과 객체 확인
        self.assertIsInstance(result, BacktestResult)
        self.assertIsNotNone(result.total_return)
        self.assertIsNotNone(result.sharpe_ratio)
        self.assertIsNotNone(result.max_drawdown)
        
        # 기본 값 검증
        self.assertEqual(result.initial_capital, 40_000_000)
        self.assertGreaterEqual(result.total_trades, 0)
        
        print(f"\n[단순 백테스트 결과]")
        print(f"  총 수익률: {result.total_return:.2%}")
        print(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
        print(f"  최대 낙폭: {result.max_drawdown:.2%}")
        print(f"  거래 횟수: {result.total_trades}")
    
    def test_walk_forward_backtest(self):
        """Walk-forward 백테스트 테스트"""
        # 더 긴 데이터 필요
        long_data = create_sample_data(days=60)
        result = self.engine.run(long_data, walk_forward_window=30)
        
        self.assertIsInstance(result, BacktestResult)
        print(f"\n[Walk-forward 백테스트 결과]")
        print(f"  총 수익률: {result.total_return:.2%}")
        print(f"  거래 횟수: {result.total_trades}")
    
    def test_trade_execution(self):
        """거래 실행 시뮬레이션 테스트"""
        # 직접 거래 실행 테스트
        signal = {
            'side': 'buy',
            'amount': 0.1,
            'reason': 'test'
        }
        
        current_price = 45000000
        current_time = datetime.now()
        
        self.engine._execute_trade(signal, current_price, current_time)
        
        # 거래 기록 확인
        self.assertEqual(len(self.engine.trades), 1)
        trade = self.engine.trades[0]
        
        # 슬리피지 적용 확인
        expected_price = current_price * (1 + self.engine.slippage_rate)
        self.assertAlmostEqual(trade.price, expected_price, places=2)
        
        # 수수료 확인 (메이커 수수료)
        expected_fee = expected_price * 0.1 * self.engine.maker_fee
        self.assertAlmostEqual(trade.fee, expected_fee, places=2)
    
    def test_equity_calculation(self):
        """자산 가치 계산 테스트"""
        self.engine.capital = 30_000_000
        self.engine.position = 0.5  # 0.5 BTC
        
        current_price = 50_000_000
        equity = self.engine._calculate_equity(current_price)
        
        expected_equity = 30_000_000 + (0.5 * 50_000_000)
        self.assertEqual(equity, expected_equity)
    
    def test_report_generation(self):
        """리포트 생성 테스트"""
        result = self.engine.run(self.sample_data)
        
        # 리포트 생성 (파일 저장 없이)
        report = self.engine.generate_report(result)
        
        # 리포트 구조 확인
        self.assertIn("백테스트 요약", report)
        self.assertIn("거래 통계", report)
        self.assertIn("리스크 메트릭", report)
        self.assertIn("수수료 설정", report)


class TestWalkForwardAnalysis(unittest.TestCase):
    """WalkForwardAnalysis 테스트"""
    
    def setUp(self):
        """테스트 설정"""
        self.strategy = MeanReversionStrategy()
        self.engine = BacktestingEngine(self.strategy)
        self.walk_forward = WalkForwardAnalysis(
            strategy=self.strategy,
            backtest_engine=self.engine,
            train_period_days=20,
            test_period_days=5,
            overlap_days=0
        )
        self.sample_data = create_sample_data(days=60)
    
    def test_window_creation(self):
        """Walk-forward 윈도우 생성 테스트"""
        windows = self.walk_forward._create_windows(self.sample_data)
        
        self.assertGreater(len(windows), 0)
        
        # 첫 번째 윈도우 확인
        first_window = windows[0]
        self.assertIsNotNone(first_window.train_start)
        self.assertIsNotNone(first_window.train_end)
        self.assertIsNotNone(first_window.test_start)
        self.assertIsNotNone(first_window.test_end)
        
        # 학습 기간이 테스트 기간보다 앞서는지 확인
        self.assertLess(first_window.train_start, first_window.train_end)
        self.assertLess(first_window.train_end, first_window.test_start)
        
        print(f"\n[Walk-forward 윈도우]")
        print(f"  생성된 윈도우 수: {len(windows)}")
    
    def test_walk_forward_analysis(self):
        """Walk-forward 분석 실행 테스트"""
        result = self.walk_forward.run(self.sample_data)
        
        self.assertIsNotNone(result.windows)
        self.assertIsNotNone(result.window_results)
        self.assertIsNotNone(result.combined_result)
        self.assertIsNotNone(result.overfitting_score)
        
        print(f"\n[Walk-forward 분석 결과]")
        print(f"  윈도우 수: {len(result.windows)}")
        print(f"  In-sample 수익률: {result.in_sample_performance['avg_return']:.2%}")
        print(f"  Out-of-sample 수익률: {result.out_of_sample_performance['avg_return']:.2%}")
        print(f"  과적합 점수: {result.overfitting_score:.3f}")
    
    def test_overfitting_score_calculation(self):
        """과적합 점수 계산 테스트"""
        # 샘플 메트릭
        in_sample = [
            {'return': 0.1, 'sharpe': 1.5, 'win_rate': 0.6},
            {'return': 0.15, 'sharpe': 1.8, 'win_rate': 0.65}
        ]
        out_sample = [
            {'return': 0.08, 'sharpe': 1.2, 'win_rate': 0.55},
            {'return': 0.12, 'sharpe': 1.4, 'win_rate': 0.58}
        ]
        
        score = self.walk_forward._calculate_overfitting_score(
            in_sample, out_sample
        )
        
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)


class TestPerformanceAnalyzer(unittest.TestCase):
    """PerformanceAnalyzer 테스트"""
    
    def setUp(self):
        """테스트 설정"""
        self.analyzer = PerformanceAnalyzer()
        
        # 샘플 백테스트 결과 생성
        self.sample_result = BacktestResult(
            total_return=0.15,
            total_trades=50,
            winning_trades=30,
            losing_trades=20,
            win_rate=0.6,
            average_win=100000,
            average_loss=-50000,
            profit_factor=1.5,
            max_drawdown=-0.1,
            max_drawdown_duration=100,
            sharpe_ratio=1.2,
            sortino_ratio=1.5,
            calmar_ratio=1.5,
            trades=[],
            equity_curve=pd.Series([40000000, 41000000, 42000000, 41500000, 43000000]),
            start_date=datetime.now() - timedelta(days=30),
            end_date=datetime.now(),
            initial_capital=40000000,
            final_capital=46000000
        )
    
    def test_basic_metrics_calculation(self):
        """기본 메트릭 계산 테스트"""
        metrics = self.analyzer._calculate_basic_metrics(self.sample_result)
        
        self.assertIn("총 수익률", metrics)
        self.assertIn("연율화 수익률", metrics)
        self.assertIn("승률", metrics)
        self.assertIn("Profit Factor", metrics)
        
        print(f"\n[기본 메트릭]")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
    
    def test_risk_metrics_calculation(self):
        """리스크 메트릭 계산 테스트"""
        metrics = self.analyzer._calculate_risk_metrics(self.sample_result)
        
        self.assertIn("Sharpe Ratio", metrics)
        self.assertIn("Sortino Ratio", metrics)
        self.assertIn("Calmar Ratio", metrics)
        self.assertIn("최대 낙폭", metrics)
        
        print(f"\n[리스크 메트릭]")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
    
    def test_strategy_score_calculation(self):
        """전략 점수 계산 테스트"""
        score = self.analyzer._calculate_strategy_score(self.sample_result)
        
        self.assertIn("수익성 점수", score)
        self.assertIn("안정성 점수", score)
        self.assertIn("일관성 점수", score)
        self.assertIn("리스크 관리 점수", score)
        self.assertIn("종합 점수", score)
        self.assertIn("전략 등급", score)
        
        print(f"\n[전략 점수]")
        for key, value in score.items():
            print(f"  {key}: {value}")
    
    def test_full_analysis(self):
        """전체 분석 테스트"""
        analysis = self.analyzer.analyze(self.sample_result)
        
        self.assertIn("basic_metrics", analysis)
        self.assertIn("risk_metrics", analysis)
        self.assertIn("strategy_score", analysis)
        
        # 리포트 생성 테스트
        report = self.analyzer.generate_report()
        self.assertIsNotNone(report)


class TestOverfittingValidator(unittest.TestCase):
    """OverfittingValidator 테스트"""
    
    def setUp(self):
        """테스트 설정"""
        self.strategy = MeanReversionStrategy()
        self.engine = BacktestingEngine(self.strategy)
        self.validator = OverfittingValidator(
            strategy=self.strategy,
            backtest_engine=self.engine,
            confidence_level=0.95
        )
        self.sample_data = create_sample_data(days=90)  # 더 긴 데이터 필요
    
    def test_cross_validation(self):
        """교차 검증 테스트"""
        cv_results = self.validator._cross_validation(self.sample_data, n_splits=3)
        
        self.assertIn("mean_return", cv_results)
        self.assertIn("std_return", cv_results)
        self.assertIn("cv_coefficient", cv_results)
        self.assertIn("fold_results", cv_results)
        
        print(f"\n[교차 검증 결과]")
        print(f"  평균 수익률: {cv_results['mean_return']:.2%}")
        print(f"  수익률 표준편차: {cv_results['std_return']:.2%}")
        print(f"  변동계수: {cv_results['cv_coefficient']:.3f}")
    
    def test_monte_carlo_simulation(self):
        """몬테카를로 시뮬레이션 테스트"""
        mc_results = self.validator._monte_carlo_simulation(
            self.sample_data,
            n_simulations=10  # 테스트용으로 적은 수
        )
        
        self.assertIn("actual_return", mc_results)
        self.assertIn("random_mean_return", mc_results)
        self.assertIn("percentile_return", mc_results)
        self.assertIn("beats_random", mc_results)
        
        print(f"\n[몬테카를로 시뮬레이션]")
        print(f"  실제 수익률: {mc_results['actual_return']:.2%}")
        print(f"  랜덤 평균 수익률: {mc_results['random_mean_return']:.2%}")
        print(f"  백분위: {mc_results['percentile_return']:.1f}%")
        print(f"  랜덤 대비 우수: {mc_results['beats_random']}")
    
    def test_temporal_stability(self):
        """시간 안정성 테스트"""
        stability_results = self.validator._temporal_stability_test(self.sample_data)
        
        self.assertIn("period_results", stability_results)
        self.assertIn("cv_return", stability_results)
        self.assertIn("all_periods_positive", stability_results)
        self.assertIn("is_stable", stability_results)
        
        print(f"\n[시간 안정성 테스트]")
        print(f"  기간별 결과 수: {len(stability_results['period_results'])}")
        print(f"  수익률 변동계수: {stability_results['cv_return']:.3f}")
        print(f"  모든 기간 양수: {stability_results['all_periods_positive']}")
        print(f"  안정성 판정: {stability_results['is_stable']}")
    
    def test_full_validation(self):
        """전체 검증 테스트"""
        # 시간이 걸리므로 작은 데이터로 테스트
        small_data = create_sample_data(days=30)
        
        validation_results = self.validator.validate(
            small_data,
            n_splits=2,
            test_size=0.2
        )
        
        self.assertIn("cross_validation", validation_results)
        self.assertIn("monte_carlo", validation_results)
        self.assertIn("overfitting_score", validation_results)
        self.assertIn("validation_summary", validation_results)
        
        summary = validation_results['validation_summary']
        print(f"\n[과적합 검증 요약]")
        print(f"  위험 수준: {summary['risk_level']}")
        print(f"  위험 점수: {summary['risk_score']}")
        print(f"  신뢰성: {'신뢰 가능' if summary['is_reliable'] else '추가 검증 필요'}")


class TestIntegration(unittest.TestCase):
    """통합 테스트"""
    
    def test_full_backtesting_pipeline(self):
        """전체 백테스팅 파이프라인 테스트"""
        print("\n" + "="*60)
        print("통합 백테스팅 파이프라인 테스트")
        print("="*60)
        
        # 1. 전략 생성
        strategy = MeanReversionStrategy()
        
        # 2. 백테스팅 엔진 생성
        engine = BacktestingEngine(
            strategy=strategy,
            initial_capital=40_000_000,
            maker_fee=0.0002,
            taker_fee=0.0015,
            use_maker_only=True
        )
        
        # 3. 샘플 데이터 생성
        data = create_sample_data(days=90)
        
        # 4. 백테스트 실행
        print("\n[1] 백테스트 실행 중...")
        backtest_result = engine.run(data)
        print(f"  ✓ 총 수익률: {backtest_result.total_return:.2%}")
        print(f"  ✓ 거래 횟수: {backtest_result.total_trades}")
        
        # 5. Walk-forward 분석
        print("\n[2] Walk-forward 분석 중...")
        walk_forward = WalkForwardAnalysis(
            strategy=strategy,
            backtest_engine=engine,
            train_period_days=30,
            test_period_days=10
        )
        wf_result = walk_forward.run(data)
        print(f"  ✓ 과적합 점수: {wf_result.overfitting_score:.3f}")
        
        # 6. 성과 분석
        print("\n[3] 성과 분석 중...")
        analyzer = PerformanceAnalyzer()
        performance = analyzer.analyze(backtest_result)
        strategy_score = performance['strategy_score']
        print(f"  ✓ 전략 등급: {strategy_score['전략 등급']}")
        print(f"  ✓ 종합 점수: {strategy_score['종합 점수']}")
        
        # 7. 과적합 검증
        print("\n[4] 과적합 검증 중...")
        validator = OverfittingValidator(strategy, engine)
        validation = validator.validate(data, n_splits=3)
        summary = validation['validation_summary']
        print(f"  ✓ 과적합 위험: {summary['risk_level']}")
        print(f"  ✓ 신뢰성: {'신뢰 가능' if summary['is_reliable'] else '추가 검증 필요'}")
        
        # 최종 판정
        print("\n" + "="*60)
        print("백테스팅 시스템 검증 완료!")
        print("="*60)
        
        # 모든 결과가 None이 아닌지 확인
        self.assertIsNotNone(backtest_result)
        self.assertIsNotNone(wf_result)
        self.assertIsNotNone(performance)
        self.assertIsNotNone(validation)


def run_all_tests():
    """모든 테스트 실행"""
    # 테스트 스위트 생성
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 테스트 추가
    suite.addTests(loader.loadTestsFromTestCase(TestBacktestingEngine))
    suite.addTests(loader.loadTestsFromTestCase(TestWalkForwardAnalysis))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformanceAnalyzer))
    suite.addTests(loader.loadTestsFromTestCase(TestOverfittingValidator))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # 테스트 실행
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 결과 요약
    print("\n" + "="*60)
    print("테스트 결과 요약")
    print("="*60)
    print(f"실행된 테스트: {result.testsRun}")
    print(f"성공: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"실패: {len(result.failures)}")
    print(f"에러: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n✅ 모든 테스트 통과!")
    else:
        print("\n❌ 일부 테스트 실패")
        if result.failures:
            print("\n실패한 테스트:")
            for test, trace in result.failures:
                print(f"  - {test}")
        if result.errors:
            print("\n에러 발생 테스트:")
            for test, trace in result.errors:
                print(f"  - {test}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)