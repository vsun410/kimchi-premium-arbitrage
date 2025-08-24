"""
Run Backtest with Historical Data
Phase 3: 백테스트 실행 스크립트
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backtesting.engine import BacktestEngine, TradingCosts
from src.backtesting.strategy import KimchiArbitrageStrategy, SimpleThresholdStrategy
from src.backtesting.metrics import PerformanceMetrics
from src.utils.logger import logger


def load_historical_data(data_dir: str = "data/historical/full") -> pd.DataFrame:
    """
    히스토리컬 데이터 로드 및 병합
    
    Args:
        data_dir: 데이터 디렉토리
        
    Returns:
        병합된 DataFrame
    """
    logger.info("Loading historical data...")
    
    # 가장 최근 파일 찾기
    import glob
    
    binance_files = glob.glob(os.path.join(data_dir, "binance_BTC_USDT_*.csv"))
    upbit_files = glob.glob(os.path.join(data_dir, "upbit_BTC_KRW_*.csv"))
    
    if not binance_files or not upbit_files:
        raise FileNotFoundError(f"Data files not found in {data_dir}")
    
    # 가장 최근 파일 선택
    binance_file = sorted(binance_files)[-1]
    upbit_file = sorted(upbit_files)[-1]
    
    logger.info(f"Loading Binance data from: {binance_file}")
    logger.info(f"Loading Upbit data from: {upbit_file}")
    
    # 바이낸스 데이터 로드
    binance_df = pd.read_csv(binance_file)
    binance_df['timestamp'] = pd.to_datetime(binance_df['timestamp'])
    binance_df.set_index('timestamp', inplace=True)
    
    # 업비트 데이터 로드
    upbit_df = pd.read_csv(upbit_file)
    upbit_df['timestamp'] = pd.to_datetime(upbit_df['timestamp'])
    upbit_df.set_index('timestamp', inplace=True)
    
    # 데이터 병합 (5분 리샘플링)
    logger.info("Resampling to 5-minute intervals...")
    
    binance_5m = binance_df.resample('5min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    upbit_5m = upbit_df.resample('5min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    # 컬럼명 변경
    binance_5m.columns = [f'binance_{col}' for col in binance_5m.columns]
    upbit_5m.columns = [f'upbit_{col}' for col in upbit_5m.columns]
    
    # 병합
    merged_df = pd.merge(
        binance_5m, 
        upbit_5m,
        left_index=True,
        right_index=True,
        how='inner'
    )
    
    logger.info(f"Loaded {len(merged_df)} data points from {merged_df.index[0]} to {merged_df.index[-1]}")
    
    return merged_df


def run_ml_backtest(data: pd.DataFrame, model_path: str = None) -> dict:
    """
    ML 모델 기반 백테스트
    
    Args:
        data: 가격 데이터
        model_path: 모델 경로
        
    Returns:
        백테스트 결과
    """
    logger.info("Running ML-based backtest...")
    
    # 거래 비용 설정
    costs = TradingCosts(
        upbit_fee=0.0005,      # 0.05%
        binance_fee=0.001,     # 0.1%
        slippage=0.0001,       # 0.01%
        funding_rate=0.0001    # 0.01% per 8h
    )
    
    # 백테스트 엔진
    engine = BacktestEngine(
        initial_capital=40_000_000,  # 4천만원
        max_position_size=0.1,        # 최대 0.1 BTC
        costs=costs
    )
    
    # ML 전략
    strategy = KimchiArbitrageStrategy(
        model_path=model_path,
        entry_threshold=3.0,   # 3% 이상 김프
        exit_threshold=1.0,    # 1% 이하로 떨어지면 청산
        stop_loss=-2.0,        # -2% 손절
        take_profit=5.0,       # 5% 익절
        use_ml_signal=True if model_path else False
    )
    
    # 백테스트 실행
    results = engine.run(
        data=data,
        strategy=strategy,
        start_date=data.index[0],
        end_date=data.index[-1]
    )
    
    return results


def run_simple_backtest(data: pd.DataFrame) -> dict:
    """
    단순 임계값 백테스트 (비교용)
    
    Args:
        data: 가격 데이터
        
    Returns:
        백테스트 결과
    """
    logger.info("Running simple threshold backtest...")
    
    # 거래 비용 설정
    costs = TradingCosts(
        upbit_fee=0.0005,
        binance_fee=0.001,
        slippage=0.0001,
        funding_rate=0.0001
    )
    
    # 백테스트 엔진
    engine = BacktestEngine(
        initial_capital=40_000_000,
        max_position_size=0.1,
        costs=costs
    )
    
    # 단순 전략
    strategy = SimpleThresholdStrategy(
        entry_threshold=3.0,
        exit_threshold=1.0
    )
    
    # 백테스트 실행
    results = engine.run(
        data=data,
        strategy=strategy,
        start_date=data.index[0],
        end_date=data.index[-1]
    )
    
    return results


def compare_strategies(ml_results: dict, simple_results: dict):
    """
    전략 비교 출력
    
    Args:
        ml_results: ML 전략 결과
        simple_results: 단순 전략 결과
    """
    print("\n" + "=" * 60)
    print("  STRATEGY COMPARISON")
    print("=" * 60)
    
    metrics = [
        'total_trades',
        'total_return',
        'sharpe_ratio',
        'max_drawdown',
        'win_rate'
    ]
    
    print(f"\n{'Metric':<20} {'ML Strategy':>15} {'Simple Strategy':>15} {'Improvement':>15}")
    print("-" * 65)
    
    for metric in metrics:
        ml_value = ml_results.get(metric, 0)
        simple_value = simple_results.get(metric, 0)
        
        if metric in ['total_return', 'max_drawdown', 'win_rate']:
            # Percentage metrics
            improvement = ml_value - simple_value
            print(f"{metric:<20} {ml_value:>14.2f}% {simple_value:>14.2f}% {improvement:>14.2f}%")
        elif metric == 'total_trades':
            # Count metrics
            improvement = ml_value - simple_value
            print(f"{metric:<20} {ml_value:>15.0f} {simple_value:>15.0f} {improvement:>15.0f}")
        else:
            # Ratio metrics
            improvement = ml_value - simple_value
            print(f"{metric:<20} {ml_value:>15.2f} {simple_value:>15.2f} {improvement:>15.2f}")
    
    print("=" * 65)


def save_backtest_results(results: dict, filename: str):
    """
    백테스트 결과 저장
    
    Args:
        results: 백테스트 결과
        filename: 저장 파일명
    """
    # 결과 디렉토리 생성
    results_dir = "results/backtest"
    os.makedirs(results_dir, exist_ok=True)
    
    # JSON으로 저장
    filepath = os.path.join(results_dir, filename)
    
    # Numpy/Pandas 타입 변환
    def convert_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        return obj
    
    # 재귀적으로 변환
    def convert_dict(d):
        result = {}
        for key, value in d.items():
            if isinstance(value, dict):
                result[key] = convert_dict(value)
            elif isinstance(value, list):
                result[key] = [convert_dict(item) if isinstance(item, dict) else convert_types(item) for item in value]
            else:
                result[key] = convert_types(value)
        return result
    
    converted_results = convert_dict(results)
    
    with open(filepath, 'w') as f:
        json.dump(converted_results, f, indent=2, default=str)
    
    logger.info(f"Results saved to {filepath}")


def main():
    """메인 실행 함수"""
    
    print("\n" + "=" * 60)
    print("  KIMCHI PREMIUM BACKTESTING SYSTEM")
    print("  Phase 3: Strategy Validation")
    print("=" * 60)
    
    # 1. 데이터 로드
    print("\n[1/4] Loading historical data...")
    data = load_historical_data()
    
    # 데이터 기간 출력
    print(f"Data period: {data.index[0].date()} to {data.index[-1].date()}")
    print(f"Total days: {(data.index[-1] - data.index[0]).days}")
    print(f"Data points: {len(data):,}")
    
    # 2. ML 모델 경로 확인
    model_path = "models/xgboost_ensemble.pkl"
    if os.path.exists(model_path):
        print(f"\n[✓] ML model found: {model_path}")
        use_ml = True
    else:
        print("\n[!] ML model not found, using threshold-only strategy")
        use_ml = False
        model_path = None
    
    # 3. 백테스트 실행
    print("\n[2/4] Running backtests...")
    
    # 데이터 분할 (Train: 80%, Test: 20%)
    split_idx = int(len(data) * 0.8)
    train_data = data.iloc[:split_idx]
    test_data = data.iloc[split_idx:]
    
    print(f"\nTrain period: {train_data.index[0].date()} to {train_data.index[-1].date()}")
    print(f"Test period: {test_data.index[0].date()} to {test_data.index[-1].date()}")
    
    # ML 백테스트
    if use_ml:
        print("\n- Running ML strategy backtest on test data...")
        ml_results = run_ml_backtest(test_data, model_path)
        
        # 성과 메트릭
        ml_metrics = PerformanceMetrics(ml_results)
        print("\nML Strategy Performance:")
        ml_metrics.print_report()
        
        # 결과 저장
        save_backtest_results(
            ml_results,
            f"ml_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
    
    # 단순 백테스트
    print("\n- Running simple strategy backtest on test data...")
    simple_results = run_simple_backtest(test_data)
    
    # 성과 메트릭
    simple_metrics = PerformanceMetrics(simple_results)
    print("\nSimple Strategy Performance:")
    simple_metrics.print_report()
    
    # 결과 저장
    save_backtest_results(
        simple_results,
        f"simple_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    
    # 4. 전략 비교
    if use_ml:
        print("\n[3/4] Comparing strategies...")
        compare_strategies(ml_results, simple_results)
    
    # 5. 시각화
    print("\n[4/4] Generating visualizations...")
    
    # ML 전략 시각화
    if use_ml:
        ml_metrics.plot_results(
            save_path=f"results/backtest/ml_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        )
    
    # 단순 전략 시각화
    simple_metrics.plot_results(
        save_path=f"results/backtest/simple_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    )
    
    print("\n[COMPLETE] Backtest finished!")
    print(f"Results saved in: results/backtest/")
    
    # 최종 추천
    print("\n" + "=" * 60)
    print("  RECOMMENDATIONS")
    print("=" * 60)
    
    if use_ml:
        if ml_results['sharpe_ratio'] > simple_results['sharpe_ratio']:
            print("[GOOD] ML strategy shows better risk-adjusted returns")
        else:
            print("[WARNING] Simple strategy performs better - ML model needs tuning")
        
        if ml_results['max_drawdown'] < -20:
            print("[WARNING] High drawdown detected - consider reducing position size")
        
        if ml_results['win_rate'] < 50:
            print("[WARNING] Low win rate - review entry/exit criteria")
    else:
        if simple_results['sharpe_ratio'] > 1.0:
            print("[GOOD] Simple strategy shows acceptable performance")
        else:
            print("[WARNING] Strategy needs improvement before live trading")
    
    print("\nNext steps:")
    print("1. Run walk-forward analysis for robustness check")
    print("2. Optimize parameters using grid search")
    print("3. Test with different market conditions")
    print("4. Implement paper trading for real-time validation")


if __name__ == "__main__":
    main()