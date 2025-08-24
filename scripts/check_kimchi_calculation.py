"""
Check Kimchi Premium Calculation
김치 프리미엄 계산 검증
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def check_calculation():
    """김프 계산 검증"""
    import glob
    
    # 데이터 로드
    data_dir = "data/historical/full"
    binance_files = glob.glob(os.path.join(data_dir, "binance_BTC_USDT_*.csv"))
    upbit_files = glob.glob(os.path.join(data_dir, "upbit_BTC_KRW_*.csv"))
    
    binance_file = sorted(binance_files)[-1]
    upbit_file = sorted(upbit_files)[-1]
    
    print("=" * 60)
    print("  KIMCHI PREMIUM CALCULATION CHECK")
    print("=" * 60)
    
    # 바이낸스 데이터 샘플
    binance_df = pd.read_csv(binance_file)
    binance_df['timestamp'] = pd.to_datetime(binance_df['timestamp'])
    binance_df.set_index('timestamp', inplace=True)
    
    # 업비트 데이터 샘플
    upbit_df = pd.read_csv(upbit_file)
    upbit_df['timestamp'] = pd.to_datetime(upbit_df['timestamp'])
    upbit_df.set_index('timestamp', inplace=True)
    
    # 최근 24시간만
    cutoff = binance_df.index[-1] - timedelta(hours=24)
    binance_recent = binance_df[binance_df.index >= cutoff]
    upbit_recent = upbit_df[upbit_df.index >= cutoff]
    
    # 병합
    merged = pd.merge(
        binance_recent[['close']].rename(columns={'close': 'binance_close'}),
        upbit_recent[['close']].rename(columns={'close': 'upbit_close'}),
        left_index=True,
        right_index=True,
        how='inner'
    )
    
    print(f"\n[Data Sample - Last 24 hours]")
    print(f"Period: {merged.index[0]} to {merged.index[-1]}")
    print(f"Data points: {len(merged)}")
    
    # 최근 5개 데이터 포인트
    print("\n[Last 5 Data Points]")
    print("-" * 80)
    print(f"{'Timestamp':<20} {'Binance (USD)':>15} {'Upbit (KRW)':>15}")
    print("-" * 80)
    
    for idx in merged.tail(5).index:
        binance_price = merged.loc[idx, 'binance_close']
        upbit_price = merged.loc[idx, 'upbit_close']
        print(f"{idx.strftime('%Y-%m-%d %H:%M'):<20} ${binance_price:>14,.2f} KRW {upbit_price:>14,.0f}")
    
    # 다양한 환율로 김프 계산
    print("\n" + "=" * 60)
    print("  KIMCHI PREMIUM WITH DIFFERENT EXCHANGE RATES")
    print("=" * 60)
    
    # 현재 가격 (마지막 데이터)
    last_binance = merged.iloc[-1]['binance_close']
    last_upbit = merged.iloc[-1]['upbit_close']
    
    print(f"\n[Current Prices]")
    print(f"Binance: ${last_binance:,.2f} USD")
    print(f"Upbit: KRW {last_upbit:,.0f}")
    
    # 다양한 환율 테스트
    exchange_rates = [1250, 1280, 1300, 1320, 1330, 1340, 1350, 1380, 1400]
    
    print(f"\n[Kimchi Premium Calculation]")
    print(f"Formula: ((Upbit_KRW - Binance_USD * USD/KRW) / (Binance_USD * USD/KRW)) * 100")
    print("-" * 60)
    print(f"{'USD/KRW':>10} {'Binance in KRW':>18} {'Premium (%)':>12}")
    print("-" * 60)
    
    for rate in exchange_rates:
        binance_krw = last_binance * rate
        premium = ((last_upbit - binance_krw) / binance_krw) * 100
        print(f"{rate:>10} KRW {binance_krw:>17,.0f} {premium:>11.2f}%")
    
    # 실제 환율 데이터 확인
    print("\n" + "=" * 60)
    print("  ACTUAL EXCHANGE RATE DATA")
    print("=" * 60)
    
    # 환율 파일 확인
    import glob
    rate_files = glob.glob("data/exchange_rates/history/*.csv")
    
    if rate_files:
        latest_rate_file = sorted(rate_files)[-1]
        print(f"\nFound exchange rate file: {latest_rate_file}")
        
        rates_df = pd.read_csv(latest_rate_file)
        if not rates_df.empty:
            print("\n[Recent Exchange Rates]")
            print(rates_df.tail(5).to_string(index=False))
            
            if 'rate' in rates_df.columns:
                avg_rate = rates_df['rate'].mean()
                latest_rate = rates_df['rate'].iloc[-1] if len(rates_df) > 0 else 1330
                print(f"\nAverage rate: {avg_rate:.2f}")
                print(f"Latest rate: {latest_rate:.2f}")
                
                # 실제 환율로 김프 재계산
                print(f"\n[Kimchi Premium with ACTUAL rate ({latest_rate:.2f})]")
                for idx in merged.tail(10).index:
                    binance_price = merged.loc[idx, 'binance_close']
                    upbit_price = merged.loc[idx, 'upbit_close']
                    binance_krw = binance_price * latest_rate
                    premium = ((upbit_price - binance_krw) / binance_krw) * 100
                    
                    print(f"{idx.strftime('%m-%d %H:%M')} | "
                          f"Binance: ${binance_price:,.0f} -> KRW {binance_krw:,.0f} | "
                          f"Upbit: KRW {upbit_price:,.0f} | "
                          f"Premium: {premium:.2f}%")
    else:
        print("\nNo exchange rate files found!")
        print("Using default rate: 1330 KRW/USD")
    
    # 전체 기간 통계
    print("\n" + "=" * 60)
    print("  PREMIUM STATISTICS (LAST 24H)")
    print("=" * 60)
    
    # 여러 환율로 전체 통계
    for rate in [1300, 1330, 1350, 1380]:
        merged[f'premium_{rate}'] = ((merged['upbit_close'] - merged['binance_close'] * rate) / 
                                     (merged['binance_close'] * rate)) * 100
        
        stats = merged[f'premium_{rate}'].describe()
        print(f"\n[With USD/KRW = {rate}]")
        print(f"Mean: {stats['mean']:.2f}%")
        print(f"Std: {stats['std']:.2f}%")
        print(f"Min: {stats['min']:.2f}%")
        print(f"Max: {stats['max']:.2f}%")
        print(f"25%: {stats['25%']:.2f}%")
        print(f"50%: {stats['50%']:.2f}%")
        print(f"75%: {stats['75%']:.2f}%")
    
    # 역계산: 현재 프리미엄이 몇%인지 확인
    print("\n" + "=" * 60)
    print("  REVERSE CALCULATION")
    print("=" * 60)
    
    print("\nIf kimchi premium is actually between -1.5% to 1.5%,")
    print("what should the exchange rate be?")
    
    for target_premium in [-1.5, -1.0, 0, 1.0, 1.5]:
        # (upbit / binance - 1) * 100 = premium
        # upbit / binance = 1 + premium/100
        # rate = upbit / (binance * (1 + premium/100))
        
        implied_rate = last_upbit / (last_binance * (1 + target_premium/100))
        print(f"\nFor {target_premium:>5.1f}% premium -> USD/KRW should be {implied_rate:,.0f}")


if __name__ == "__main__":
    check_calculation()