#!/usr/bin/env python3
"""
Feature Engineering Pipeline 테스트
Task #14: 기술적 지표 및 시장 미시구조 특징 추출 테스트
"""

import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ml.feature_engineering import FeatureEngineer

warnings.filterwarnings("ignore")


def create_sample_ohlcv_data(n_samples: int = 500) -> pd.DataFrame:
    """
    샘플 OHLCV 데이터 생성
    
    Args:
        n_samples: 샘플 수
        
    Returns:
        OHLCV 데이터프레임
    """
    # 시간 인덱스
    dates = pd.date_range(
        start=datetime.now() - timedelta(hours=n_samples),
        periods=n_samples,
        freq="1h"
    )
    
    # 가격 시뮬레이션 (랜덤 워크)
    np.random.seed(42)
    base_price = 100000  # 100K
    returns = np.random.randn(n_samples) * 0.01  # 1% 변동성
    close_prices = base_price * np.exp(np.cumsum(returns))
    
    # OHLCV 데이터 생성
    data = pd.DataFrame(index=dates)
    data["close"] = close_prices
    data["open"] = close_prices * (1 + np.random.uniform(-0.005, 0.005, n_samples))
    data["high"] = np.maximum(data["open"], data["close"]) * (1 + np.random.uniform(0, 0.01, n_samples))
    data["low"] = np.minimum(data["open"], data["close"]) * (1 - np.random.uniform(0, 0.01, n_samples))
    data["volume"] = np.random.uniform(10, 100, n_samples)
    
    # 오더북 데이터 추가 (미시구조 테스트용)
    data["bid"] = data["close"] * 0.9995
    data["ask"] = data["close"] * 1.0005
    
    return data


def create_sample_orderbook_data(n_samples: int = 500) -> pd.DataFrame:
    """
    샘플 오더북 데이터 생성
    
    Args:
        n_samples: 샘플 수
        
    Returns:
        오더북 데이터프레임
    """
    dates = pd.date_range(
        start=datetime.now() - timedelta(hours=n_samples),
        periods=n_samples,
        freq="1h"
    )
    
    data = pd.DataFrame(index=dates)
    data["bid_volume"] = np.random.uniform(1, 10, n_samples)
    data["ask_volume"] = np.random.uniform(1, 10, n_samples)
    data["liquidity_score"] = np.random.uniform(60, 95, n_samples)
    data["spread"] = np.random.uniform(0.01, 0.1, n_samples)
    
    return data


def test_price_features():
    """가격 특징 테스트"""
    print("\n" + "=" * 60)
    print("TEST 1: Price Features")
    print("=" * 60)
    
    try:
        # 데이터 준비
        data = create_sample_ohlcv_data()
        engineer = FeatureEngineer(feature_groups=["price"])
        
        # 가격 특징 생성
        price_features = engineer.create_price_features(data)
        
        print(f"\nGenerated {len(price_features.columns)} price features")
        print("\nSample features:")
        
        # 주요 특징 확인
        key_features = [
            "returns", "log_returns", "sma_20", "ema_12",
            "volatility_20", "momentum_5"
        ]
        
        for feat in key_features:
            if feat in price_features.columns:
                val = price_features[feat].iloc[-1]
                if not pd.isna(val):
                    print(f"  {feat:20}: {val:,.6f}")
        
        # 검증
        if len(price_features.columns) > 30:
            print("\n[OK] Price features generated successfully")
            return True
        else:
            print("\n[FAIL] Insufficient price features")
            return False
            
    except Exception as e:
        print(f"[ERROR] Price features test failed: {e}")
        return False


def test_volume_features():
    """거래량 특징 테스트"""
    print("\n" + "=" * 60)
    print("TEST 2: Volume Features")
    print("=" * 60)
    
    try:
        data = create_sample_ohlcv_data()
        engineer = FeatureEngineer(feature_groups=["volume"])
        
        volume_features = engineer.create_volume_features(data)
        
        print(f"\nGenerated {len(volume_features.columns)} volume features")
        print("\nSample features:")
        
        key_features = ["volume_ratio", "vwap", "obv", "vroc_5", "money_flow_5"]
        
        for feat in key_features:
            if feat in volume_features.columns:
                val = volume_features[feat].iloc[-1]
                if not pd.isna(val):
                    print(f"  {feat:20}: {val:,.6f}")
        
        if len(volume_features.columns) >= 8:
            print("\n[OK] Volume features generated successfully")
            return True
        else:
            print("\n[FAIL] Insufficient volume features")
            return False
            
    except Exception as e:
        print(f"[ERROR] Volume features test failed: {e}")
        return False


def test_technical_indicators():
    """기술적 지표 테스트"""
    print("\n" + "=" * 60)
    print("TEST 3: Technical Indicators")
    print("=" * 60)
    
    try:
        data = create_sample_ohlcv_data()
        engineer = FeatureEngineer(feature_groups=["technical"])
        
        technical_features = engineer.create_technical_indicators(data)
        
        print(f"\nGenerated {len(technical_features.columns)} technical indicators")
        print("\nSample indicators:")
        
        key_indicators = [
            "rsi_14", "macd", "bb_upper", "atr_14",
            "stoch_k", "adx", "cci_20", "williams_r"
        ]
        
        for ind in key_indicators:
            if ind in technical_features.columns:
                val = technical_features[ind].iloc[-1]
                if not pd.isna(val):
                    print(f"  {ind:20}: {val:,.6f}")
        
        # RSI 범위 확인 (0-100)
        if "rsi_14" in technical_features.columns:
            rsi_values = technical_features["rsi_14"].dropna()
            if len(rsi_values) > 0:
                valid_rsi = (rsi_values >= 0).all() and (rsi_values <= 100).all()
                print(f"\nRSI range check: {'[OK]' if valid_rsi else '[FAIL]'}")
        
        if len(technical_features.columns) >= 15:
            print("\n[OK] Technical indicators generated successfully")
            return True
        else:
            print("\n[FAIL] Insufficient technical indicators")
            return False
            
    except Exception as e:
        print(f"[ERROR] Technical indicators test failed: {e}")
        return False


def test_microstructure_features():
    """시장 미시구조 특징 테스트"""
    print("\n" + "=" * 60)
    print("TEST 4: Market Microstructure Features")
    print("=" * 60)
    
    try:
        data = create_sample_ohlcv_data()
        orderbook = create_sample_orderbook_data()
        engineer = FeatureEngineer(feature_groups=["microstructure"])
        
        # returns 추가 (미시구조 특징에 필요)
        data["returns"] = data["close"].pct_change()
        
        micro_features = engineer.create_microstructure_features(data, orderbook)
        
        print(f"\nGenerated {len(micro_features.columns)} microstructure features")
        print("\nSample features:")
        
        key_features = [
            "bid_ask_spread", "price_efficiency", "order_imbalance",
            "liquidity_score", "kyles_lambda", "amihud_illiquidity"
        ]
        
        for feat in key_features:
            if feat in micro_features.columns:
                val = micro_features[feat].iloc[-1]
                if not pd.isna(val):
                    print(f"  {feat:20}: {val:,.6f}")
        
        if len(micro_features.columns) >= 5:
            print("\n[OK] Microstructure features generated successfully")
            return True
        else:
            print("\n[FAIL] Insufficient microstructure features")
            return False
            
    except Exception as e:
        print(f"[ERROR] Microstructure features test failed: {e}")
        return False


def test_time_features():
    """시간 특징 테스트"""
    print("\n" + "=" * 60)
    print("TEST 5: Time Features")
    print("=" * 60)
    
    try:
        data = create_sample_ohlcv_data()
        engineer = FeatureEngineer(feature_groups=["time"])
        
        time_features = engineer.create_time_features(data)
        
        print(f"\nGenerated {len(time_features.columns)} time features")
        print("\nSample features:")
        
        key_features = [
            "hour", "day_of_week", "hour_sin", "hour_cos",
            "is_weekend", "is_asian_session"
        ]
        
        for feat in key_features:
            if feat in time_features.columns:
                val = time_features[feat].iloc[-1]
                print(f"  {feat:20}: {val:,.3f}")
        
        # 시간 범위 확인
        if "hour" in time_features.columns:
            hour_range = (time_features["hour"] >= 0).all() and (time_features["hour"] <= 23).all()
            print(f"\nHour range check: {'[OK]' if hour_range else '[FAIL]'}")
        
        if len(time_features.columns) >= 10:
            print("\n[OK] Time features generated successfully")
            return True
        else:
            print("\n[FAIL] Insufficient time features")
            return False
            
    except Exception as e:
        print(f"[ERROR] Time features test failed: {e}")
        return False


def test_full_pipeline():
    """전체 파이프라인 테스트"""
    print("\n" + "=" * 60)
    print("TEST 6: Full Feature Engineering Pipeline")
    print("=" * 60)
    
    try:
        # 데이터 준비
        data = create_sample_ohlcv_data(1000)
        orderbook = create_sample_orderbook_data(1000)
        
        # 전체 파이프라인 실행
        engineer = FeatureEngineer(scaler_type="robust")
        features = engineer.engineer_features(
            data,
            orderbook,
            include_lags=True,
            include_rolling=True
        )
        
        print(f"\nTotal features generated: {len(features.columns)}")
        print(f"Data shape: {features.shape}")
        print(f"NaN ratio: {features.isna().sum().sum() / features.size:.2%}")
        
        # 스케일링 테스트
        print("\n[Testing feature scaling...]")
        scaled_features = engineer.fit_transform(features)
        
        # 스케일링 검증 (평균 ~0, 표준편차 ~1)
        numeric_cols = scaled_features.select_dtypes(include=[np.number]).columns
        means = scaled_features[numeric_cols].mean()
        stds = scaled_features[numeric_cols].std()
        
        mean_check = abs(means.mean()) < 0.5  # 평균이 0 근처
        std_check = abs(stds.mean() - 1) < 0.5  # 표준편차가 1 근처
        
        print(f"  Mean of means: {means.mean():.3f}")
        print(f"  Mean of stds: {stds.mean():.3f}")
        print(f"  Scaling check: {'[OK]' if mean_check and std_check else '[FAIL]'}")
        
        # 특징 중요도 테스트
        print("\n[Testing feature importance...]")
        target = pd.Series(np.random.randn(len(features)), index=features.index)
        importance = engineer.get_feature_importance(features, target, method="correlation")
        
        print(f"  Top 5 important features:")
        for feat, score in importance.head(5).items():
            print(f"    {feat:30}: {score:.3f}")
        
        # 특징 선택 테스트
        selected = engineer.select_features(features, target, n_features=30)
        print(f"\n  Selected {len(selected)} features")
        
        if len(features.columns) > 100:
            print("\n[OK] Full pipeline executed successfully")
            return True
        else:
            print("\n[FAIL] Pipeline generated insufficient features")
            return False
            
    except Exception as e:
        print(f"[ERROR] Full pipeline test failed: {e}")
        return False


def test_feature_persistence():
    """특징 저장 및 로드 테스트"""
    print("\n" + "=" * 60)
    print("TEST 7: Feature Persistence")
    print("=" * 60)
    
    try:
        import pickle
        import tempfile
        
        # 특징 생성
        data = create_sample_ohlcv_data()
        engineer = FeatureEngineer()
        features = engineer.engineer_features(data)
        
        # 스케일러 학습
        engineer.fit_transform(features)
        
        # 저장
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            pickle.dump(engineer, f)
            temp_file = f.name
        
        print(f"Engineer saved to {temp_file}")
        
        # 로드
        with open(temp_file, "rb") as f:
            loaded_engineer = pickle.load(f)
        
        # 검증
        if loaded_engineer.fitted and loaded_engineer.feature_names == engineer.feature_names:
            print("[OK] Feature engineer saved and loaded successfully")
            
            # 정리
            import os
            os.unlink(temp_file)
            return True
        else:
            print("[FAIL] Loaded engineer doesn't match original")
            return False
            
    except Exception as e:
        print(f"[ERROR] Persistence test failed: {e}")
        return False


def main():
    """메인 테스트 함수"""
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING TEST SUITE")
    print("=" * 60)
    print("\nTesting Task #14: Feature Engineering Pipeline")
    
    tests = [
        ("Price Features", test_price_features),
        ("Volume Features", test_volume_features),
        ("Technical Indicators", test_technical_indicators),
        ("Microstructure Features", test_microstructure_features),
        ("Time Features", test_time_features),
        ("Full Pipeline", test_full_pipeline),
        ("Feature Persistence", test_feature_persistence),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            print(f"\nRunning: {test_name}")
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"[ERROR] {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    
    for test_name, success in results:
        status = "[PASS]" if success else "[FAIL]"
        print(f"{test_name:25} {status}")
    
    passed = sum(1 for _, s in results if s)
    total = len(results)
    
    print("-" * 60)
    print(f"Results: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    
    if passed == total:
        print("\n[SUCCESS] Task #14 COMPLETED! Feature engineering pipeline ready.")
        print("\nKey features implemented:")
        print("  1. Price features (returns, moving averages, momentum)")
        print("  2. Volume features (VWAP, OBV, money flow)")
        print("  3. Technical indicators (RSI, MACD, Bollinger Bands)")
        print("  4. Market microstructure (spread, order imbalance)")
        print("  5. Time features (hour, day of week, trading sessions)")
        print("  6. Lag and rolling statistics")
        print("  7. Feature scaling and selection")
        return 0
    else:
        print("\n[WARN] Some tests failed. Please review.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)