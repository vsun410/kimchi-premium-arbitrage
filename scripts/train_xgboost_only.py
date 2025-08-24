#!/usr/bin/env python3
"""
XGBoost Model Training Script
Phase 2: XGBoost 모델만 먼저 학습
"""

import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ml.feature_engineering import FeatureEngineer
from src.ml.xgboost_ensemble import XGBoostEnsemble
from src.utils.logger import logger


def main():
    """XGBoost 학습"""
    print("\n" + "=" * 60)
    print("  XGBOOST MODEL TRAINING")
    print("=" * 60)
    
    try:
        # 1. 데이터 로드
        print("\n[1/5] Loading data...")
        data_dir = Path('data/historical/full')
        
        binance_file = max(data_dir.glob('binance_*.csv'), key=lambda x: x.stat().st_mtime)
        upbit_file = max(data_dir.glob('upbit_*.csv'), key=lambda x: x.stat().st_mtime)
        
        binance_df = pd.read_csv(binance_file, index_col=0, parse_dates=True)
        upbit_df = pd.read_csv(upbit_file, index_col=0, parse_dates=True)
        
        print(f"  Binance: {len(binance_df):,} samples")
        print(f"  Upbit: {len(upbit_df):,} samples")
        
        # 2. 김프 계산
        print("\n[2/5] Calculating Kimchi Premium...")
        USD_KRW = 1330.0
        
        # 공통 시간대
        common_idx = binance_df.index.intersection(upbit_df.index)
        binance_aligned = binance_df.loc[common_idx]
        upbit_aligned = upbit_df.loc[common_idx]
        
        kimchi_premium = ((upbit_aligned['close'] - binance_aligned['close'] * USD_KRW) / 
                         (binance_aligned['close'] * USD_KRW) * 100)
        
        print(f"  Kimchi Premium: {kimchi_premium.mean():.2f}% (±{kimchi_premium.std():.2f}%)")
        
        # 3. 특징 생성
        print("\n[3/5] Feature engineering...")
        fe = FeatureEngineer()
        
        # OHLCV 데이터
        ohlcv_df = pd.DataFrame({
            'open': binance_aligned['open'],
            'high': binance_aligned['high'],
            'low': binance_aligned['low'],
            'close': binance_aligned['close'],
            'volume': binance_aligned['volume']
        })
        
        features = fe.engineer_features(ohlcv_df)
        target = kimchi_premium.shift(-60)  # 60분 후 예측
        
        # NaN 제거
        valid_idx = ~(features.isna().any(axis=1) | target.isna())
        features = features[valid_idx]
        target = target[valid_idx]
        
        print(f"  Features: {features.shape}")
        
        # 4. 데이터 분할
        print("\n[4/5] Training XGBoost...")
        split_idx = int(len(features) * 0.8)
        val_split = int(split_idx * 0.8)
        
        X_train = features[:val_split].values
        y_train = target[:val_split].values
        X_val = features[val_split:split_idx].values
        y_val = target[val_split:split_idx].values
        X_test = features[split_idx:].values
        y_test = target[split_idx:].values
        
        # XGBoost 학습
        xgb = XGBoostEnsemble()
        
        # 학습 데이터 합치기 (fit은 validation을 별도로 받지 않음)
        X_train_full = np.vstack([X_train, X_val])
        y_train_full = np.hstack([y_train, y_val])
        
        xgb.fit(X_train_full, y_train_full)
        
        # 5. 평가
        print("\n[5/5] Evaluation...")
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        pred = xgb.predict(X_test)
        
        mse = mean_squared_error(y_test, pred)
        mae = mean_absolute_error(y_test, pred)
        r2 = r2_score(y_test, pred)
        
        print(f"  MSE: {mse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  RMSE: {np.sqrt(mse):.4f}")
        print(f"  R2 Score: {r2:.4f}")
        
        # 저장
        model_dir = Path('models/trained')
        model_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = model_dir / f'xgboost_{timestamp}.pkl'
        
        # joblib로 저장
        import joblib
        joblib.dump(xgb.model, model_path)
        
        print(f"\n[SAVED] Model saved to {model_path}")
        print("\n[SUCCESS] Training complete!")
        
        return 0
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        logger.error(f"XGBoost training failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())