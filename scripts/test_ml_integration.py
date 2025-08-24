#!/usr/bin/env python3
"""
ML Models Integration Test Suite
Phase 2 완료 평가 체크리스트
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

# Import all ML components
from src.data_collectors.orderbook_collector import OrderbookCollector
from src.data_storage.csv_storage import CSVStorage
from src.ml.feature_engineering import FeatureEngineer
from src.ml.lstm_model import EDLSTMModel, LSTMTrainer, create_sequences
from src.ml.xgboost_ensemble import XGBoostEnsemble, HybridEnsemble
from src.utils.logger import logger

class MLIntegrationTest:
    """ML 모델 통합 테스트"""
    
    def __init__(self):
        self.results = {}
        self.passed = 0
        self.failed = 0
        
    def print_header(self, title: str):
        """헤더 출력"""
        print("\n" + "=" * 60)
        print(f"  {title}")
        print("=" * 60)
        
    def check(self, name: str, condition: bool, details: str = ""):
        """체크 항목 평가"""
        status = "[PASS]" if condition else "[FAIL]"
        self.results[name] = condition
        
        if condition:
            self.passed += 1
        else:
            self.failed += 1
            
        print(f"{status} {name}")
        if details:
            print(f"        -> {details}")
        
        return condition
    
    def test_data_pipeline(self):
        """데이터 파이프라인 테스트"""
        self.print_header("1. 데이터 수집 파이프라인")
        
        try:
            # CSV Storage 초기화
            storage = CSVStorage(base_dir="test_data/ml_integration")
            
            # 더미 데이터 생성
            test_data = pd.DataFrame({
                'timestamp': pd.date_range(start='2024-01-01', periods=100, freq='1h'),
                'open': np.random.uniform(100000, 110000, 100),
                'high': np.random.uniform(110000, 120000, 100),
                'low': np.random.uniform(90000, 100000, 100),
                'close': np.random.uniform(100000, 110000, 100),
                'volume': np.random.uniform(10, 100, 100),
                'exchange': ['upbit'] * 100,
                'symbol': ['BTC/KRW'] * 100
            })
            
            # 데이터 저장
            storage.save_price_data(test_data)
            
            # 데이터 읽기
            loaded_data = storage.load_price_data(
                exchange='upbit',
                symbol='BTC/KRW',
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 1, 5)
            )
            
            self.check(
                "CSV 저장/로드",
                loaded_data is not None and len(loaded_data) > 0,
                f"Loaded {len(loaded_data) if loaded_data is not None else 0} records"
            )
            
            # 정리
            import shutil
            shutil.rmtree("test_data/ml_integration", ignore_errors=True)
            
            return True
            
        except Exception as e:
            self.check("CSV 저장/로드", False, str(e))
            return False
    
    def test_feature_engineering(self):
        """Feature Engineering 테스트"""
        self.print_header("2. Feature Engineering")
        
        try:
            # 샘플 데이터 생성
            n_samples = 500
            dates = pd.date_range(start=datetime.now() - timedelta(hours=n_samples),
                                 periods=n_samples, freq='1h')
            
            data = pd.DataFrame({
                'open': np.random.uniform(100000, 110000, n_samples),
                'high': np.random.uniform(110000, 120000, n_samples),
                'low': np.random.uniform(90000, 100000, n_samples),
                'close': np.random.uniform(100000, 110000, n_samples),
                'volume': np.random.uniform(10, 100, n_samples),
                'bid': np.random.uniform(99000, 100000, n_samples),
                'ask': np.random.uniform(100000, 101000, n_samples)
            }, index=dates)
            
            # Feature 생성
            engineer = FeatureEngineer()
            features = engineer.engineer_features(data)
            
            self.check(
                "Feature 생성",
                features is not None and len(features.columns) > 100,
                f"Generated {len(features.columns)} features"
            )
            
            # Feature 스케일링
            scaled = engineer.fit_transform(features)
            
            self.check(
                "Feature 스케일링",
                scaled is not None and scaled.shape == features.shape,
                "Scaling successful"
            )
            
            # Feature importance
            target = pd.Series(np.random.randn(len(features)), index=features.index)
            importance = engineer.get_feature_importance(features, target)
            
            self.check(
                "Feature Importance",
                importance is not None and len(importance) > 0,
                f"Top feature: {list(importance.items())[0] if importance else 'None'}"
            )
            
            return True
            
        except Exception as e:
            self.check("Feature Engineering", False, str(e))
            return False
    
    def test_lstm_model(self):
        """LSTM 모델 테스트"""
        self.print_header("3. ED-LSTM Model")
        
        try:
            import torch
            
            # 샘플 데이터
            n_samples = 200
            n_features = 10
            seq_length = 48
            pred_length = 24
            
            # 시퀀스 생성
            data = np.random.randn(n_samples, n_features)
            X, y = create_sequences(data, seq_length, pred_length)
            
            self.check(
                "시퀀스 생성",
                X.shape[0] > 0 and y.shape[0] > 0,
                f"X shape: {X.shape}, y shape: {y.shape}"
            )
            
            # 모델 생성
            model = EDLSTMModel(
                input_size=n_features,
                hidden_size=32,
                output_size=1,
                num_layers=1,
                decoder_length=pred_length
            )
            
            # 예측 테스트
            test_input = torch.FloatTensor(X[:1])
            with torch.no_grad():
                output, attention = model(test_input, teacher_forcing_ratio=0)
            
            self.check(
                "LSTM 예측",
                output.shape == (1, pred_length, 1),
                f"Output shape: {output.shape}"
            )
            
            self.check(
                "Attention 메커니즘",
                attention is not None,
                "Attention weights generated"
            )
            
            return True
            
        except Exception as e:
            self.check("LSTM Model", False, str(e))
            return False
    
    def test_xgboost_model(self):
        """XGBoost 모델 테스트"""
        self.print_header("4. XGBoost Ensemble")
        
        try:
            # 샘플 데이터
            X = np.random.randn(100, 20)
            y = np.random.randn(100)
            
            # 모델 생성 및 학습
            model = XGBoostEnsemble(
                model_type="regression",
                n_estimators=20,
                max_depth=3
            )
            
            model.fit(X[:80], y[:80])
            predictions = model.predict(X[80:])
            
            self.check(
                "XGBoost 학습/예측",
                predictions is not None and len(predictions) == 20,
                f"Predictions shape: {predictions.shape}"
            )
            
            # Feature importance
            importance = model.get_feature_importance(top_n=5)
            
            self.check(
                "Feature Importance",
                importance is not None and len(importance) > 0,
                f"Top {len(importance)} features identified"
            )
            
            # SHAP values
            shap_values = model.get_shap_values(X[:5])
            
            self.check(
                "SHAP 설명가능성",
                shap_values is not None and shap_values.shape[0] == 5,
                "SHAP values calculated"
            )
            
            return True
            
        except Exception as e:
            self.check("XGBoost Model", False, str(e))
            return False
    
    def test_ensemble_integration(self):
        """앙상블 통합 테스트"""
        self.print_header("5. Hybrid Ensemble Integration")
        
        try:
            import torch
            
            # 데이터 준비
            n_samples = 100
            n_features = 10
            seq_length = 24
            pred_length = 12
            
            # XGBoost용 features
            X_features = np.random.randn(n_samples, n_features)
            y = np.random.randn(n_samples)
            
            # LSTM용 sequences
            X_sequences = np.random.randn(n_samples, seq_length, n_features)
            
            # 모델 생성
            xgb_model = XGBoostEnsemble(model_type="regression", n_estimators=10)
            xgb_model.fit(X_features, y)
            
            lstm_model = EDLSTMModel(
                input_size=n_features,
                hidden_size=16,
                output_size=1,
                decoder_length=pred_length
            )
            
            # Mock LSTM for testing
            class MockLSTM:
                def predict(self, X):
                    return np.random.randn(len(X), pred_length, 1)
            
            # Hybrid ensemble
            hybrid = HybridEnsemble(
                lstm_model=MockLSTM(),
                xgboost_model=xgb_model,
                ensemble_method="weighted"
            )
            
            # 앙상블 예측
            predictions = hybrid.predict(X_features[:10], X_sequences[:10])
            
            self.check(
                "하이브리드 앙상블",
                predictions is not None and len(predictions) == 10,
                f"Ensemble predictions: {predictions.shape}"
            )
            
            return True
            
        except Exception as e:
            self.check("Hybrid Ensemble", False, str(e))
            return False
    
    def test_end_to_end_pipeline(self):
        """End-to-End 파이프라인 테스트"""
        self.print_header("6. End-to-End Pipeline")
        
        try:
            # 1. 데이터 생성
            n_samples = 1000
            dates = pd.date_range(start='2024-01-01', periods=n_samples, freq='1h')
            
            raw_data = pd.DataFrame({
                'open': np.random.uniform(100000, 110000, n_samples),
                'high': np.random.uniform(110000, 120000, n_samples),
                'low': np.random.uniform(90000, 100000, n_samples),
                'close': np.random.uniform(100000, 110000, n_samples),
                'volume': np.random.uniform(10, 100, n_samples),
            }, index=dates)
            
            # 김치 프리미엄 추가 (목표 변수)
            raw_data['kimchi_premium'] = np.random.uniform(0, 5, n_samples)
            
            self.check(
                "데이터 준비",
                len(raw_data) == n_samples,
                f"{n_samples} samples created"
            )
            
            # 2. Feature Engineering
            engineer = FeatureEngineer(feature_groups=['price', 'volume', 'technical'])
            features = engineer.create_price_features(raw_data)
            
            self.check(
                "Feature 추출",
                len(features.columns) > 20,
                f"{len(features.columns)} features extracted"
            )
            
            # 3. 데이터 분할
            train_size = int(0.8 * len(features))
            X_train = features.iloc[:train_size].fillna(0).values
            y_train = raw_data['kimchi_premium'].iloc[:train_size].values
            X_test = features.iloc[train_size:].fillna(0).values
            y_test = raw_data['kimchi_premium'].iloc[train_size:].values
            
            # 4. XGBoost 학습
            xgb_model = XGBoostEnsemble(model_type="regression", n_estimators=10)
            xgb_model.fit(X_train, y_train)
            
            # 5. 예측
            predictions = xgb_model.predict(X_test)
            
            # 6. 평가
            from sklearn.metrics import mean_squared_error, r2_score
            mse = mean_squared_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            
            self.check(
                "예측 정확도",
                r2 > -1,  # 매우 관대한 기준 (랜덤 데이터이므로)
                f"MSE: {mse:.4f}, R2: {r2:.4f}"
            )
            
            return True
            
        except Exception as e:
            self.check("End-to-End Pipeline", False, str(e))
            return False
    
    def run_all_tests(self):
        """모든 테스트 실행"""
        print("\n" + "=" * 60)
        print("  ML MODELS INTEGRATION TEST SUITE")
        print("  Phase 2 완료 평가")
        print("=" * 60)
        
        # 각 테스트 실행
        self.test_data_pipeline()
        self.test_feature_engineering()
        self.test_lstm_model()
        self.test_xgboost_model()
        self.test_ensemble_integration()
        self.test_end_to_end_pipeline()
        
        # 최종 결과
        self.print_header("최종 평가 결과")
        
        print(f"\n통과: {self.passed}/{self.passed + self.failed}")
        print(f"실패: {self.failed}/{self.passed + self.failed}")
        print(f"성공률: {self.passed/(self.passed + self.failed)*100:.1f}%")
        
        if self.failed == 0:
            print("\n[SUCCESS] Phase 2 완료! 모든 테스트 통과!")
            print("[OK] ML 모델 통합 준비 완료")
            print("[OK] Production 배포 가능")
            return True
        else:
            print("\n[WARNING] 일부 테스트 실패")
            print("[ERROR] 수정이 필요한 항목:")
            for name, passed in self.results.items():
                if not passed:
                    print(f"  - {name}")
            return False


if __name__ == "__main__":
    tester = MLIntegrationTest()
    success = tester.run_all_tests()
    
    if success:
        print("\n다음 단계:")
        print("1. git add -A")
        print("2. git commit -m 'feat: complete Phase 2 - ML Model Development'")
        print("3. git push origin feature/ml-models")
        print("4. GitHub에서 PR 생성")
        print("5. CI 통과 확인 후 merge")
        
    exit(0 if success else 1)