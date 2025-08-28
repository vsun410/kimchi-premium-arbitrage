"""
평가 시스템 통합 테스트
Task #19의 모든 컴포넌트를 테스트
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import tempfile
import os
from pathlib import Path

# 프로젝트 모듈
from src.evaluation.performance_metrics import PerformanceMetrics
from src.evaluation.ab_testing import ABTestFramework, ABTestResult
from src.evaluation.report_generator import ReportGenerator
from src.evaluation.model_evaluator import (
    ModelEvaluator, EvaluationResult,
    LSTMEvaluator, XGBoostEvaluator
)
from src.evaluation.live_monitor import LiveMonitor, LiveMetrics


class TestPerformanceMetrics(unittest.TestCase):
    """PerformanceMetrics 테스트"""
    
    def setUp(self):
        """테스트 데이터 준비"""
        # 샘플 포트폴리오 가치 생성
        np.random.seed(42)
        self.returns = np.random.randn(252) * 0.01 + 0.001  # 일일 수익률
        self.portfolio_values = [10000000]  # 1천만원 시작
        
        for r in self.returns:
            self.portfolio_values.append(self.portfolio_values[-1] * (1 + r))
        
        self.metrics = PerformanceMetrics(self.portfolio_values)
    
    def test_return_metrics(self):
        """수익률 메트릭 테스트"""
        total_return = self.metrics.total_return()
        annual_return = self.metrics.annual_return()
        
        self.assertIsInstance(total_return, float)
        self.assertIsInstance(annual_return, float)
        
        # 양의 수익률 기대 (평균이 양수이므로)
        self.assertGreater(total_return, 0)
    
    def test_risk_metrics(self):
        """리스크 메트릭 테스트"""
        volatility = self.metrics.volatility()
        max_dd = self.metrics.max_drawdown()
        var_95 = self.metrics.value_at_risk(0.95)
        
        self.assertIsInstance(volatility, float)
        self.assertIsInstance(max_dd, float)
        self.assertIsInstance(var_95, float)
        
        # 변동성은 양수
        self.assertGreater(volatility, 0)
        # 최대 낙폭은 음수
        self.assertLessEqual(max_dd, 0)
    
    def test_sharpe_ratio(self):
        """샤프 비율 테스트"""
        sharpe = self.metrics.sharpe_ratio()
        
        self.assertIsInstance(sharpe, float)
        # 샤프 비율 범위 확인 (-5 ~ 5)
        self.assertGreater(sharpe, -5)
        self.assertLess(sharpe, 5)
    
    def test_win_rate(self):
        """승률 테스트"""
        win_rate = self.metrics.win_rate()
        
        self.assertIsInstance(win_rate, float)
        # 승률은 0~100% 사이
        self.assertGreaterEqual(win_rate, 0)
        self.assertLessEqual(win_rate, 100)


class TestABTesting(unittest.TestCase):
    """ABTestFramework 테스트"""
    
    def setUp(self):
        """테스트 프레임워크 설정"""
        self.framework = ABTestFramework(confidence_level=0.95)
        
        # 샘플 수익률 생성
        np.random.seed(42)
        self.returns_a = np.random.randn(100) * 0.01 + 0.002  # 평균 0.2%
        self.returns_b = np.random.randn(100) * 0.01 + 0.001  # 평균 0.1%
    
    def test_compare_models(self):
        """모델 비교 테스트"""
        result = self.framework.compare_models(
            self.returns_a,
            self.returns_b,
            "Model A",
            "Model B"
        )
        
        self.assertIsInstance(result, ABTestResult)
        self.assertEqual(result.model_a, "Model A")
        self.assertEqual(result.model_b, "Model B")
        
        # 통계 검정 결과 확인
        self.assertIsInstance(result.p_value, float)
        self.assertIsInstance(result.is_significant, bool)
        self.assertIsInstance(result.effect_size, float)
    
    def test_mann_whitney_test(self):
        """Mann-Whitney U 테스트"""
        result = self.framework.mann_whitney_test(
            self.returns_a,
            self.returns_b,
            "Model A",
            "Model B"
        )
        
        self.assertIsInstance(result, ABTestResult)
        self.assertEqual(result.test_type, "mann-whitney")
    
    def test_bootstrap_confidence_interval(self):
        """부트스트랩 신뢰구간 테스트"""
        point, lower, upper = self.framework.bootstrap_confidence_interval(
            self.returns_a,
            self.returns_b,
            n_bootstrap=1000
        )
        
        self.assertIsInstance(point, float)
        self.assertIsInstance(lower, float)
        self.assertIsInstance(upper, float)
        
        # 신뢰구간 순서 확인
        self.assertLessEqual(lower, point)
        self.assertLessEqual(point, upper)
    
    def test_multiple_comparison_correction(self):
        """다중 비교 보정 테스트"""
        p_values = [0.01, 0.02, 0.05, 0.1]
        
        # Bonferroni 보정
        adjusted = self.framework.multiple_comparison_correction(
            p_values,
            method='bonferroni'
        )
        
        self.assertEqual(len(adjusted), len(p_values))
        # 보정된 p-value는 원래보다 크거나 같음
        for orig, adj in zip(p_values, adjusted):
            self.assertGreaterEqual(adj, orig)


class TestModelEvaluator(unittest.TestCase):
    """ModelEvaluator 테스트"""
    
    def setUp(self):
        """평가기 설정"""
        self.evaluator = ModelEvaluator()
        
        # 테스트 데이터 생성
        dates = pd.date_range(start='2024-01-01', periods=100, freq='h')
        self.test_data = pd.DataFrame({
            'timestamp': dates,
            'premium_rate': np.random.randn(100) * 2 + 3,
            'btc_price': np.random.randn(100) * 1000 + 50000,
            'volume': np.random.randn(100) * 100 + 1000
        })
    
    def test_lstm_evaluator(self):
        """LSTM 평가기 테스트"""
        lstm_eval = LSTMEvaluator("dummy_model.pth")
        result = lstm_eval.evaluate(self.test_data)
        
        self.assertIsInstance(result, EvaluationResult)
        self.assertEqual(result.model_type, "LSTM")
        self.assertIsNotNone(result.sharpe_ratio)
        self.assertIsNotNone(result.total_return)
    
    def test_xgboost_evaluator(self):
        """XGBoost 평가기 테스트"""
        xgb_eval = XGBoostEvaluator("dummy_model.pkl")
        result = xgb_eval.evaluate(self.test_data)
        
        self.assertIsInstance(result, EvaluationResult)
        self.assertEqual(result.model_type, "XGBoost")
        self.assertIsNotNone(result.accuracy)
    
    def test_model_comparison(self):
        """모델 비교 테스트"""
        # 더미 평가기 추가
        lstm_eval = LSTMEvaluator("dummy_lstm.pth")
        xgb_eval = XGBoostEvaluator("dummy_xgb.pkl")
        
        self.evaluator.add_evaluator("LSTM", lstm_eval)
        self.evaluator.add_evaluator("XGBoost", xgb_eval)
        
        # 평가 실행
        results = self.evaluator.evaluate_all(test_data=self.test_data)
        
        self.assertEqual(len(results), 2)
        self.assertIn("LSTM", results)
        self.assertIn("XGBoost", results)
        
        # 비교 테이블 생성
        comparison_df = self.evaluator.compare_models()
        
        self.assertIsInstance(comparison_df, pd.DataFrame)
        self.assertEqual(len(comparison_df), 2)
    
    def test_best_model_selection(self):
        """최고 모델 선택 테스트"""
        # 더미 결과 생성
        result1 = EvaluationResult(
            model_name="Model1",
            model_type="LSTM",
            evaluation_date=datetime.now(),
            total_return=10.0,
            annual_return=12.0,
            monthly_return=1.0,
            daily_return=0.04,
            volatility=15.0,
            max_drawdown=-10.0,
            var_95=-2.0,
            cvar_95=-3.0,
            sharpe_ratio=1.2,
            sortino_ratio=1.5,
            calmar_ratio=1.2,
            information_ratio=0.8,
            win_rate=55.0,
            profit_factor=1.5,
            avg_win=1.0,
            avg_loss=-0.5,
            max_consecutive_wins=5,
            max_consecutive_losses=3
        )
        
        result2 = EvaluationResult(
            model_name="Model2",
            model_type="XGBoost",
            evaluation_date=datetime.now(),
            total_return=15.0,
            annual_return=18.0,
            monthly_return=1.5,
            daily_return=0.06,
            volatility=12.0,
            max_drawdown=-8.0,
            var_95=-1.5,
            cvar_95=-2.5,
            sharpe_ratio=1.8,  # 더 높은 샤프 비율
            sortino_ratio=2.0,
            calmar_ratio=2.25,
            information_ratio=1.2,
            win_rate=60.0,
            profit_factor=2.0,
            avg_win=1.2,
            avg_loss=-0.4,
            max_consecutive_wins=7,
            max_consecutive_losses=2
        )
        
        self.evaluator.results = {"Model1": result1, "Model2": result2}
        
        best_name, best_result = self.evaluator.get_best_model('sharpe_ratio')
        
        self.assertEqual(best_name, "Model2")
        self.assertEqual(best_result.sharpe_ratio, 1.8)


class TestReportGenerator(unittest.TestCase):
    """ReportGenerator 테스트"""
    
    def setUp(self):
        """리포트 생성기 설정"""
        self.temp_dir = tempfile.mkdtemp()
        self.generator = ReportGenerator(save_dir=self.temp_dir)
        
        # 더미 평가 결과 생성
        self.eval_results = {
            "TestModel": type('obj', (object,), {
                'total_return': 10.5,
                'annual_return': 12.3,
                'sharpe_ratio': 1.5,
                'max_drawdown': -8.2,
                'win_rate': 58.0
            })()
        }
    
    def test_report_generation(self):
        """리포트 생성 테스트"""
        report_path = self.generator.generate_report(
            evaluation_results=self.eval_results,
            metadata={'test': True}
        )
        
        self.assertTrue(os.path.exists(report_path))
        self.assertTrue(report_path.endswith('.html'))
        
        # HTML 파일 내용 확인
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        self.assertIn('Model Evaluation Report', content)
        self.assertIn('TestModel', content)
        self.assertIn('Sharpe Ratio', content)
    
    def tearDown(self):
        """임시 디렉토리 정리"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)


class TestLiveMonitor(unittest.TestCase):
    """LiveMonitor 테스트"""
    
    def setUp(self):
        """모니터 설정"""
        self.monitor = LiveMonitor(initial_capital=10000000)
    
    def test_portfolio_update(self):
        """포트폴리오 업데이트 테스트"""
        portfolio_data = {
            'total_value': 10500000,
            'position_value': 500000,
            'cash_balance': 10000000,
            'premium_rate': 3.5,
            'position_size': 0.1,
            'position_side': 'long',
            'leverage': 1.0,
            'model_signal': 1,
            'model_confidence': 0.85,
            'model_name': 'TestModel'
        }
        
        self.monitor.update_portfolio(portfolio_data)
        
        self.assertIsNotNone(self.monitor.current_metrics)
        self.assertEqual(self.monitor.current_metrics.portfolio_value, 10500000)
        self.assertEqual(self.monitor.current_metrics.pnl, 500000)
        self.assertAlmostEqual(self.monitor.current_metrics.pnl_percent, 5.0, places=1)
    
    def test_trade_update(self):
        """거래 업데이트 테스트"""
        # 포트폴리오 초기화
        self.monitor.update_portfolio({'total_value': 10000000})
        
        trade_data = {
            'action': 'BUY',
            'size': 0.1,
            'price': 50000,
            'premium_rate': 3.2,
            'model': 'TestModel'
        }
        
        self.monitor.update_from_trade(trade_data)
        
        self.assertEqual(len(self.monitor.trade_history), 1)
        self.assertEqual(self.monitor.current_metrics.last_trade_action, 'BUY')
        self.assertEqual(self.monitor.current_metrics.total_trades_today, 1)
    
    def test_dashboard_data(self):
        """대시보드 데이터 테스트"""
        # 몇 개의 업데이트 추가
        for i in range(5):
            self.monitor.update_portfolio({
                'total_value': 10000000 + i * 100000,
                'premium_rate': 3.0 + i * 0.1
            })
        
        data = self.monitor.get_dashboard_data()
        
        self.assertIn('timestamps', data)
        self.assertIn('portfolio_values', data)
        self.assertIn('current_metrics', data)
        self.assertEqual(len(data['portfolio_values']), 5)
    
    def test_alert_thresholds(self):
        """알림 임계값 테스트"""
        # 큰 손실 시뮬레이션
        self.monitor.update_portfolio({'total_value': 10000000})
        self.monitor.update_portfolio({'total_value': 8500000})  # 15% 손실
        
        # 드로다운이 임계값을 초과했는지 확인
        self.assertLess(self.monitor.current_metrics.current_drawdown, -10)
    
    def test_callbacks(self):
        """콜백 테스트"""
        callback_called = {'called': False}
        
        def test_callback(metrics):
            callback_called['called'] = True
        
        self.monitor.add_callback(test_callback)
        self.monitor.update_portfolio({'total_value': 10000000})
        self.monitor._run_callbacks()
        
        self.assertTrue(callback_called['called'])


if __name__ == '__main__':
    unittest.main()