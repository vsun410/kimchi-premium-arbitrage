"""
Run Backtesting for Dynamic Hedge Strategy
백테스팅 실행 스크립트
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path

from backtesting import (
    BacktestEngine,
    DataLoader,
    StrategySimulator,
    PerformanceAnalyzer,
    ReportGenerator
)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_dynamic_hedge_backtest(start_date: str = None, 
                              end_date: str = None,
                              initial_capital_krw: float = 20000000,
                              initial_capital_usd: float = 15000):
    """
    Dynamic Hedge 전략 백테스팅 실행
    
    Args:
        start_date: 시작 날짜 (YYYY-MM-DD)
        end_date: 종료 날짜 (YYYY-MM-DD)
        initial_capital_krw: 초기 KRW 자본
        initial_capital_usd: 초기 USD 자본
    """
    
    logger.info("=" * 60)
    logger.info("Dynamic Hedge Strategy Backtesting")
    logger.info("=" * 60)
    
    # 1. 데이터 로딩
    logger.info("Loading historical data...")
    data_loader = DataLoader()
    data = data_loader.load_all_data(start_date, end_date)
    
    # 데이터 정보 출력
    data_info = data_loader.get_data_info()
    logger.info(f"Upbit data: {data_info.get('upbit', {}).get('records', 0)} records")
    logger.info(f"Binance data: {data_info.get('binance', {}).get('records', 0)} records")
    logger.info(f"Premium data: Mean={data_info.get('premium', {}).get('mean_premium', 0):.2f}%")
    
    # 2. 백테스팅 엔진 초기화
    logger.info("\nInitializing backtest engine...")
    initial_capital = {
        'KRW': initial_capital_krw,
        'USD': initial_capital_usd
    }
    
    engine = BacktestEngine(initial_capital, fee_rate=0.001)
    
    # 3. 전략 시뮬레이터 초기화
    logger.info("Initializing strategy simulator...")
    simulator = StrategySimulator(engine, position_size_pct=0.02)
    
    # 4. 백테스팅 실행
    logger.info("\nRunning backtest...")
    
    # 데이터 준비
    upbit_data = data['upbit']
    binance_data = data['binance']
    
    # 시뮬레이션 실행
    total_steps = len(upbit_data)
    report_interval = total_steps // 10  # 10% 단위로 진행상황 출력
    
    for i, (timestamp, upbit_row) in enumerate(upbit_data.iterrows()):
        # 진행상황 출력
        if i % report_interval == 0:
            progress = (i / total_steps) * 100
            logger.info(f"Progress: {progress:.1f}%")
        
        # 바이낸스 데이터 확인
        if timestamp not in binance_data.index:
            continue
        
        binance_row = binance_data.loc[timestamp]
        
        # 현재 가격 업데이트
        current_prices = {
            'upbit_BTC': upbit_row['close'],
            'binance_BTC': binance_row['close']
        }
        
        engine.update_time(timestamp, current_prices)
        
        # 김치 프리미엄 계산
        kimchi_premium = ((upbit_row['close'] - binance_row['close'] * 1350) / 
                         (binance_row['close'] * 1350)) * 100
        
        # OHLCV 데이터 준비 (최근 100개)
        start_idx = max(0, i - 100)
        upbit_ohlcv = upbit_data.iloc[start_idx:i+1]
        binance_ohlcv = binance_data.iloc[start_idx:i+1]
        
        # 신호 생성
        signal_data = {
            'upbit_price': upbit_row['close'],
            'binance_price': binance_row['close'],
            'kimchi_premium': kimchi_premium,
            'upbit_ohlcv': upbit_ohlcv,
            'binance_ohlcv': binance_ohlcv,
            'volume': upbit_row.get('volume', 100)
        }
        
        signals = simulator.generate_signals(timestamp, signal_data)
        
        # 신호 실행
        for signal in signals:
            success = simulator.execute_signal(
                signal,
                upbit_row['close'],
                binance_row['close']
            )
            
            if success:
                logger.debug(f"Signal executed: {signal.action} at {timestamp}")
        
        # 포트폴리오 기록
        engine.record_portfolio()
    
    logger.info("Backtest completed!")
    
    # 5. 성과 분석
    logger.info("\nAnalyzing performance...")
    analyzer = PerformanceAnalyzer(
        engine.portfolio_history,
        engine.trades
    )
    
    performance_summary = analyzer.get_performance_summary()
    
    # 6. 결과 출력
    logger.info("\n" + "=" * 60)
    logger.info("BACKTEST RESULTS")
    logger.info("=" * 60)
    
    logger.info(f"Initial Capital: ₩{initial_capital_krw:,.0f} + ${initial_capital_usd:,.0f}")
    logger.info(f"Final Value: ₩{performance_summary.get('final_value', 0):,.0f}")
    logger.info(f"Total Return: {performance_summary.get('total_return', 0):.2f}%")
    logger.info(f"Total Profit: ₩{performance_summary.get('total_return_krw', 0):,.0f}")
    
    logger.info("\n📊 Performance Metrics:")
    logger.info(f"  Sharpe Ratio: {performance_summary.get('sharpe_ratio', 0):.2f}")
    logger.info(f"  Calmar Ratio: {performance_summary.get('calmar_ratio', 0):.2f}")
    logger.info(f"  Max Drawdown: {performance_summary.get('max_drawdown', 0):.2f}%")
    logger.info(f"  Win Rate: {performance_summary.get('win_rate', 0):.1f}%")
    logger.info(f"  Profit Factor: {performance_summary.get('profit_factor', 0):.2f}")
    
    logger.info("\n📈 Trading Statistics:")
    logger.info(f"  Total Trades: {performance_summary.get('total_trades', 0)}")
    logger.info(f"  Total Fees: ₩{performance_summary.get('total_fees', 0):,.0f}")
    
    logger.info("\n💰 Monthly Performance:")
    logger.info(f"  Average Monthly Return: {performance_summary.get('monthly_return', 0):.2f}%")
    logger.info(f"  Average Monthly Profit: ₩{performance_summary.get('monthly_return_krw', 0):,.0f}")
    
    # 목표 달성 여부 확인
    logger.info("\n🎯 Target Achievement:")
    target_monthly_profit = 2000000  # 월 200만원
    actual_monthly_profit = performance_summary.get('monthly_return_krw', 0)
    
    if actual_monthly_profit >= target_monthly_profit:
        logger.info(f"  ✅ Monthly target achieved! (₩{actual_monthly_profit:,.0f} >= ₩{target_monthly_profit:,.0f})")
    else:
        logger.info(f"  ❌ Monthly target not met (₩{actual_monthly_profit:,.0f} < ₩{target_monthly_profit:,.0f})")
    
    if performance_summary.get('sharpe_ratio', 0) >= 1.5:
        logger.info(f"  ✅ Sharpe Ratio target achieved! ({performance_summary.get('sharpe_ratio', 0):.2f} >= 1.5)")
    else:
        logger.info(f"  ❌ Sharpe Ratio below target ({performance_summary.get('sharpe_ratio', 0):.2f} < 1.5)")
    
    # 7. 리포트 생성
    logger.info("\nGenerating reports...")
    report_gen = ReportGenerator()
    report_path = report_gen.generate_full_report(
        performance_summary,
        engine.portfolio_history,
        engine.trades,
        simulator.signals
    )
    
    logger.info(f"Report saved to: {report_path}")
    
    # 8. 전략 통계
    strategy_stats = simulator.get_strategy_stats()
    logger.info("\n📊 Strategy Statistics:")
    logger.info(f"  Total Signals: {strategy_stats.get('total_signals', 0)}")
    logger.info(f"  Open Signals: {strategy_stats.get('open_signals', 0)}")
    logger.info(f"  Close Signals: {strategy_stats.get('close_signals', 0)}")
    logger.info(f"  Average Confidence: {strategy_stats.get('avg_confidence', 0):.2%}")
    
    return performance_summary


def run_scenario_backtests():
    """시나리오별 백테스팅 실행"""
    
    scenarios = [
        {
            'name': '정상 김프 차익 시나리오',
            'start_date': '2024-08-24',
            'end_date': '2025-08-24',
            'capital_krw': 20000000,
            'capital_usd': 15000
        },
        {
            'name': '소규모 자본 시나리오',
            'start_date': '2024-08-24',
            'end_date': '2025-08-24',
            'capital_krw': 5000000,
            'capital_usd': 4000
        },
        {
            'name': '대규모 자본 시나리오',
            'start_date': '2024-08-24',
            'end_date': '2025-08-24',
            'capital_krw': 100000000,
            'capital_usd': 75000
        }
    ]
    
    results = []
    
    for scenario in scenarios:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running scenario: {scenario['name']}")
        logger.info(f"{'='*60}")
        
        result = run_dynamic_hedge_backtest(
            start_date=scenario['start_date'],
            end_date=scenario['end_date'],
            initial_capital_krw=scenario['capital_krw'],
            initial_capital_usd=scenario['capital_usd']
        )
        
        result['scenario'] = scenario['name']
        results.append(result)
    
    # 시나리오 비교
    logger.info(f"\n{'='*60}")
    logger.info("SCENARIO COMPARISON")
    logger.info(f"{'='*60}")
    
    comparison_df = pd.DataFrame(results)
    comparison_df = comparison_df[['scenario', 'total_return', 'sharpe_ratio', 
                                 'max_drawdown', 'monthly_return_krw']]
    
    logger.info("\n" + comparison_df.to_string())
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Dynamic Hedge Backtesting')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--capital-krw', type=float, default=20000000,
                       help='Initial KRW capital')
    parser.add_argument('--capital-usd', type=float, default=15000,
                       help='Initial USD capital')
    parser.add_argument('--scenarios', action='store_true',
                       help='Run multiple scenarios')
    
    args = parser.parse_args()
    
    if args.scenarios:
        run_scenario_backtests()
    else:
        run_dynamic_hedge_backtest(
            start_date=args.start_date,
            end_date=args.end_date,
            initial_capital_krw=args.capital_krw,
            initial_capital_usd=args.capital_usd
        )