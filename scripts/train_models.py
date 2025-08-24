#!/usr/bin/env python3
"""
ML Model Training Script with Real Data
Phase 2: 실제 1년치 데이터로 ML 모델 학습
"""

import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ml.feature_engineering import FeatureEngineer
from src.ml.lstm_model import EDLSTMModel, LSTMTrainer
from src.ml.xgboost_ensemble import XGBoostEnsemble, HybridEnsemble
from src.utils.logger import logger


class ModelTrainer:
    """ML 모델 학습 관리자"""
    
    def __init__(self, data_dir: str = 'data/historical/full'):
        self.data_dir = Path(data_dir)
        self.feature_engineer = FeatureEngineer()
        self.scaler = StandardScaler()
        
    def load_data(self) -> tuple:
        """데이터 로드 및 김프 계산"""
        print("\n[1/7] Loading historical data...")
        
        # 최신 파일 찾기
        binance_files = list(self.data_dir.glob('binance_BTC_USDT_*.csv'))
        upbit_files = list(self.data_dir.glob('upbit_BTC_KRW_*.csv'))
        
        if not binance_files or not upbit_files:
            raise FileNotFoundError("Historical data files not found!")
        
        # 가장 최근 파일 선택
        binance_file = max(binance_files, key=lambda x: x.stat().st_mtime)
        upbit_file = max(upbit_files, key=lambda x: x.stat().st_mtime)
        
        print(f"  Loading Binance: {binance_file.name}")
        binance_df = pd.read_csv(binance_file, index_col=0, parse_dates=True)
        
        print(f"  Loading Upbit: {upbit_file.name}")
        upbit_df = pd.read_csv(upbit_file, index_col=0, parse_dates=True)
        
        # 데이터 정보
        print(f"  Binance: {len(binance_df):,} candles")
        print(f"  Upbit: {len(upbit_df):,} candles")
        
        return binance_df, upbit_df
    
    def calculate_kimchi_premium(self, binance_df: pd.DataFrame, upbit_df: pd.DataFrame) -> pd.DataFrame:
        """김치 프리미엄 계산"""
        print("\n[2/7] Calculating Kimchi Premium...")
        
        # USD/KRW 환율 (실제로는 API에서 가져와야 함)
        # 여기서는 고정값 사용
        USD_KRW = 1330.0
        
        # 시간 인덱스 정렬
        binance_df = binance_df.sort_index()
        upbit_df = upbit_df.sort_index()
        
        # 1분 리샘플링 (공통 시간대 맞추기)
        binance_1m = binance_df.resample('1T').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        upbit_1m = upbit_df.resample('1T').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        # 공통 시간대만 선택
        common_index = binance_1m.index.intersection(upbit_1m.index)
        
        if len(common_index) == 0:
            raise ValueError("No common timestamps found between exchanges!")
        
        binance_aligned = binance_1m.loc[common_index]
        upbit_aligned = upbit_1m.loc[common_index]
        
        # 김프 계산
        binance_krw = binance_aligned['close'] * USD_KRW
        upbit_krw = upbit_aligned['close']
        kimchi_premium = ((upbit_krw - binance_krw) / binance_krw * 100)
        
        # 결합 데이터프레임
        combined_df = pd.DataFrame({
            'binance_close': binance_aligned['close'],
            'upbit_close': upbit_aligned['close'],
            'binance_volume': binance_aligned['volume'],
            'upbit_volume': upbit_aligned['volume'],
            'kimchi_premium': kimchi_premium
        })
        
        print(f"  Combined data: {len(combined_df):,} samples")
        print(f"  Kimchi Premium stats:")
        print(f"    Mean: {kimchi_premium.mean():.2f}%")
        print(f"    Std: {kimchi_premium.std():.2f}%")
        print(f"    Min: {kimchi_premium.min():.2f}%")
        print(f"    Max: {kimchi_premium.max():.2f}%")
        
        return combined_df
    
    def prepare_features(self, df: pd.DataFrame) -> tuple:
        """특징 엔지니어링"""
        print("\n[3/7] Feature engineering...")
        
        # 기본 OHLCV 데이터 준비
        ohlcv_df = pd.DataFrame({
            'open': df['binance_close'].shift(1),
            'high': df['binance_close'].rolling(60).max(),
            'low': df['binance_close'].rolling(60).min(),
            'close': df['binance_close'],
            'volume': df['binance_volume']
        }).dropna()
        
        # 특징 생성
        features = self.feature_engineer.engineer_features(ohlcv_df)
        
        # 타겟: 다음 시간 김프 (예측 목표)
        target = df['kimchi_premium'].shift(-60)  # 60분 후 김프 예측
        
        # NaN 제거
        valid_idx = ~(features.isna().any(axis=1) | target.isna())
        features = features[valid_idx]
        target = target[valid_idx]
        
        print(f"  Features: {features.shape}")
        print(f"  Target samples: {len(target)}")
        
        return features, target
    
    def split_data(self, features: pd.DataFrame, target: pd.Series) -> tuple:
        """데이터 분할"""
        print("\n[4/7] Splitting data...")
        
        # 시계열 분할 (순서 유지)
        split_idx = int(len(features) * 0.8)
        
        X_train = features[:split_idx]
        X_test = features[split_idx:]
        y_train = target[:split_idx]
        y_test = target[split_idx:]
        
        # Validation 분할
        val_split = int(len(X_train) * 0.8)
        X_val = X_train[val_split:]
        X_train = X_train[:val_split]
        y_val = y_train[val_split:]
        y_train = y_train[:val_split]
        
        print(f"  Train: {X_train.shape}")
        print(f"  Val: {X_val.shape}")
        print(f"  Test: {X_test.shape}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_lstm(self, X_train, X_val, y_train, y_val) -> LSTMTrainer:
        """LSTM 모델 학습"""
        print("\n[5/7] Training LSTM model...")
        
        # 스케일링
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # LSTM 모델 생성
        lstm_model = EDLSTMModel(
            input_size=X_train.shape[1],
            hidden_size=128,
            output_size=1,
            num_layers=2,
            dropout=0.3,
            use_attention=True
        )
        
        # 트레이너 생성
        lstm_trainer = LSTMTrainer(
            model=lstm_model,
            learning_rate=0.001
        )
        
        # DataLoader 생성
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train_scaled),
            torch.FloatTensor(y_train.values.reshape(-1, 1))
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val_scaled),
            torch.FloatTensor(y_val.values.reshape(-1, 1))
        )
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        
        # 학습
        history = lstm_trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            n_epochs=20,  # 실제 데이터니까 충분히 학습
            early_stopping_patience=5
        )
        
        # 학습 결과
        print(f"  Final train loss: {history['train_loss'][-1]:.4f}")
        print(f"  Final val loss: {history['val_loss'][-1]:.4f}")
        
        return lstm_trainer
    
    def train_xgboost(self, X_train, X_val, y_train, y_val) -> XGBoostEnsemble:
        """XGBoost 모델 학습"""
        print("\n[6/7] Training XGBoost ensemble...")
        
        xgb_model = XGBoostEnsemble()
        
        # 학습
        xgb_model.train(
            X_train.values, y_train.values,
            X_val.values, y_val.values
        )
        
        # 특징 중요도
        feature_importance = xgb_model.get_feature_importance(X_train.columns.tolist())
        print("  Top 5 important features:")
        for feat, imp in list(feature_importance.items())[:5]:
            print(f"    {feat}: {imp:.3f}")
        
        return xgb_model
    
    def evaluate_models(self, lstm_model, xgb_model, X_test, y_test):
        """모델 평가"""
        print("\n[7/7] Evaluating models...")
        
        # 예측
        X_test_scaled = self.scaler.transform(X_test)
        
        lstm_pred = lstm_model.predict(X_test_scaled)
        xgb_pred = xgb_model.predict(X_test.values)
        
        # 하이브리드 예측 (앙상블)
        hybrid_pred = 0.6 * lstm_pred.flatten() + 0.4 * xgb_pred
        
        # 메트릭 계산
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        models = {
            'LSTM': lstm_pred.flatten(),
            'XGBoost': xgb_pred,
            'Hybrid': hybrid_pred
        }
        
        results = {}
        for name, pred in models.items():
            # 길이 맞추기
            min_len = min(len(pred), len(y_test))
            pred = pred[:min_len]
            y_true = y_test[:min_len]
            
            mse = mean_squared_error(y_true, pred)
            mae = mean_absolute_error(y_true, pred)
            r2 = r2_score(y_true, pred)
            
            results[name] = {
                'MSE': mse,
                'MAE': mae,
                'RMSE': np.sqrt(mse),
                'R2': r2
            }
            
            print(f"\n  {name} Performance:")
            print(f"    MSE: {mse:.4f}")
            print(f"    MAE: {mae:.4f}")
            print(f"    RMSE: {np.sqrt(mse):.4f}")
            print(f"    R2 Score: {r2:.4f}")
        
        return results
    
    def save_models(self, lstm_model, xgb_model, results):
        """모델 저장"""
        print("\n[SAVE] Saving trained models...")
        
        model_dir = Path('models/trained')
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # 타임스탬프
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # LSTM 저장
        lstm_path = model_dir / f'lstm_model_{timestamp}.pth'
        torch.save(lstm_model.model.state_dict(), lstm_path)
        print(f"  LSTM saved: {lstm_path}")
        
        # XGBoost 저장
        xgb_path = model_dir / f'xgboost_model_{timestamp}.pkl'
        xgb_model.save(str(xgb_path))
        print(f"  XGBoost saved: {xgb_path}")
        
        # 결과 저장
        results_df = pd.DataFrame(results).T
        results_path = model_dir / f'training_results_{timestamp}.csv'
        results_df.to_csv(results_path)
        print(f"  Results saved: {results_path}")
        
        return model_dir / f'*_{timestamp}.*'


def main():
    """메인 학습 프로세스"""
    print("\n" + "=" * 60)
    print("  ML MODEL TRAINING WITH REAL DATA")
    print("  Phase 2 Final Step")
    print("=" * 60)
    
    try:
        trainer = ModelTrainer()
        
        # 1. 데이터 로드
        binance_df, upbit_df = trainer.load_data()
        
        # 2. 김프 계산
        combined_df = trainer.calculate_kimchi_premium(binance_df, upbit_df)
        
        # 3. 특징 준비
        features, target = trainer.prepare_features(combined_df)
        
        # 4. 데이터 분할
        X_train, X_val, X_test, y_train, y_val, y_test = trainer.split_data(features, target)
        
        # 5. LSTM 학습
        lstm_model = trainer.train_lstm(X_train, X_val, y_train, y_val)
        
        # 6. XGBoost 학습
        xgb_model = trainer.train_xgboost(X_train, X_val, y_train, y_val)
        
        # 7. 평가
        results = trainer.evaluate_models(lstm_model, xgb_model, X_test, y_test)
        
        # 8. 저장
        saved_path = trainer.save_models(lstm_model, xgb_model, results)
        
        print("\n" + "=" * 60)
        print("  TRAINING COMPLETE!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Review model performance metrics")
        print("2. Fine-tune hyperparameters if needed")
        print("3. Proceed to Phase 3: Backtesting")
        
        return 0
        
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        logger.error(f"Training failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)