"""
Test Triangular Arbitrage with Real Market Data
삼각 차익거래 실시간 테스트
"""

import sys
import os
import asyncio
import ccxt.pro as ccxtpro
from datetime import datetime
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.strategies.triangular_arbitrage import TriangularArbitrage, TriangularOpportunity
from src.utils.exchange_rate_manager import get_exchange_rate_manager
from src.utils.logger import logger


async def test_with_real_data():
    """
    실시간 데이터로 삼각 차익거래 테스트
    """
    
    print("\n" + "=" * 60)
    print("  TRIANGULAR ARBITRAGE REAL-TIME TEST")
    print("=" * 60)
    
    # Exchange 초기화
    upbit = ccxtpro.upbit({
        'enableRateLimit': True,
        'options': {
            'defaultType': 'spot'
        }
    })
    
    binance = ccxtpro.binance({
        'enableRateLimit': True,
        'options': {
            'defaultType': 'spot'
        }
    })
    
    # 환율 매니저
    rate_manager = get_exchange_rate_manager()
    current_rate = rate_manager.current_rate
    print(f"\nCurrent USD/KRW rate: {current_rate:,.2f}")
    
    # 삼각 차익거래 전략 초기화
    strategy = TriangularArbitrage(
        upbit_client=upbit,
        binance_client=binance,
        min_profit_rate=0.1,  # 0.1% 최소 수익률 (낮게 설정)
        max_execution_time=3.0
    )
    
    # 10회 테스트
    opportunities_found = []
    
    for i in range(10):
        print(f"\n[Test {i+1}/10] Checking for opportunities...")
        
        try:
            # 가격 수집
            prices = await strategy.fetch_all_prices()
            
            if all([prices['btc_krw'], prices['btc_usdt'], prices['usd_krw']]):
                print(f"  BTC/KRW (Upbit): {prices['btc_krw']:,.0f} KRW")
                print(f"  BTC/USDT (Binance): {prices['btc_usdt']:,.2f} USDT")
                print(f"  USD/KRW rate: {prices['usd_krw']:,.2f}")
                
                # 김프 계산
                kimchi_premium = ((prices['btc_krw'] - prices['btc_usdt'] * prices['usd_krw']) / 
                                (prices['btc_usdt'] * prices['usd_krw'])) * 100
                print(f"  Current Kimchi Premium: {kimchi_premium:.2f}%")
                
                # 차익거래 기회 계산
                opportunities = strategy.calculate_triangular_profit(
                    prices,
                    amount=0.001  # 0.001 BTC로 테스트
                )
                
                if opportunities:
                    print(f"\n  [FOUND] Found {len(opportunities)} opportunities!")
                    for opp in opportunities:
                        print(f"    Path: {' → '.join(opp.path)}")
                        print(f"    Expected profit: {opp.profit_rate:.3f}%")
                        print(f"    Net profit: {opp.net_profit:,.0f} KRW")
                        opportunities_found.append(opp)
                else:
                    print("  No profitable opportunities found")
                    
                # 오더북 기반 정밀 계산도 테스트
                if prices.get('btc_krw_orderbook') and prices.get('btc_usdt_orderbook'):
                    orderbook_opps = strategy.calculate_with_orderbook(prices, amount=0.001)
                    if orderbook_opps:
                        print(f"  [ORDERBOOK] Orderbook analysis found {len(orderbook_opps)} opportunities")
                        opportunities_found.extend(orderbook_opps)
            else:
                print("  Failed to fetch some prices")
                
        except Exception as e:
            print(f"  Error: {e}")
            
        # 대기
        await asyncio.sleep(3)
    
    # 결과 분석
    await upbit.close()
    await binance.close()
    
    print("\n" + "=" * 60)
    print("  TEST RESULTS SUMMARY")
    print("=" * 60)
    
    if opportunities_found:
        # DataFrame으로 변환
        df = pd.DataFrame([{
            'path': ' → '.join(opp.path),
            'profit_rate': opp.profit_rate,
            'net_profit': opp.net_profit,
            'fees': opp.fees,
            'execution_time': opp.execution_time
        } for opp in opportunities_found])
        
        print(f"\nTotal opportunities found: {len(opportunities_found)}")
        print(f"Average profit rate: {df['profit_rate'].mean():.3f}%")
        print(f"Max profit rate: {df['profit_rate'].max():.3f}%")
        print(f"Min profit rate: {df['profit_rate'].min():.3f}%")
        
        print("\nPath frequency:")
        for path, count in df['path'].value_counts().items():
            print(f"  {path}: {count} times")
        
        # 수익성 평가
        print("\n[Profitability Analysis]")
        profitable = df[df['profit_rate'] > 0.1]
        if len(profitable) > 0:
            print(f"Profitable opportunities: {len(profitable)} ({len(profitable)/len(df)*100:.1f}%)")
            print(f"Average profitable rate: {profitable['profit_rate'].mean():.3f}%")
            
            # 월간 수익 예상 (하루 100회 거래 가정)
            avg_profit_krw = profitable['net_profit'].mean()
            daily_profit = avg_profit_krw * 100  # 100 trades per day
            monthly_profit = daily_profit * 30
            monthly_return = (monthly_profit / 40_000_000) * 100
            
            print(f"\n[Monthly Projection (100 trades/day)]")
            print(f"Average profit per trade: {avg_profit_krw:,.0f} KRW")
            print(f"Daily profit: {daily_profit:,.0f} KRW")
            print(f"Monthly profit: {monthly_profit:,.0f} KRW")
            print(f"Monthly return on 40M KRW: {monthly_return:.2f}%")
        else:
            print("No profitable opportunities after fees")
    else:
        print("\nNo opportunities found during test period")
        print("This suggests the market is currently efficient")
    
    print("\n[Market Efficiency]")
    print("If few opportunities found, it means:")
    print("1. Exchange prices are well-aligned")
    print("2. Arbitrageurs are actively trading")
    print("3. Need faster execution or lower fees")
    
    return opportunities_found


async def test_execution_simulation():
    """
    실행 시뮬레이션 테스트
    """
    print("\n" + "=" * 60)
    print("  EXECUTION SIMULATION TEST")
    print("=" * 60)
    
    # Mock opportunity
    mock_opportunity = TriangularOpportunity(
        path=['KRW→BTC', 'BTC→USDT', 'USDT→KRW'],
        profit_rate=0.25,
        volume=0.001,
        timestamp=datetime.now(),
        exchange='upbit+binance',
        fees=0.0025,
        net_profit=10000,
        execution_time=2.5
    )
    
    print(f"\n[Mock Opportunity]")
    print(f"Path: {' → '.join(mock_opportunity.path)}")
    print(f"Expected profit: {mock_opportunity.profit_rate:.3f}%")
    print(f"Volume: {mock_opportunity.volume} BTC")
    
    # 실행 시뮬레이션
    print("\n[Simulated Execution Steps]")
    print("1. Check account balances")
    print("2. Place market order on Upbit (BTC/KRW)")
    print("3. Transfer BTC to Binance (if needed)")
    print("4. Place market order on Binance (BTC/USDT)")
    print("5. Convert USDT to KRW")
    print("6. Calculate actual profit")
    
    # 리스크 분석
    print("\n[Risk Analysis]")
    print("- Price movement risk: Market can move during execution")
    print("- Transfer delay risk: Inter-exchange transfers take time")
    print("- Slippage risk: Large orders may have slippage")
    print("- Fee uncertainty: Actual fees may differ")
    
    # 성공 확률 계산
    success_factors = {
        'price_stability': 0.8,  # 80% chance prices stay stable
        'execution_speed': 0.9,  # 90% chance of fast execution
        'no_slippage': 0.7,      # 70% chance of minimal slippage
        'fee_accuracy': 0.95     # 95% chance fees are as expected
    }
    
    success_probability = np.prod(list(success_factors.values()))
    
    print(f"\n[Success Probability]")
    for factor, prob in success_factors.items():
        print(f"  {factor}: {prob*100:.0f}%")
    print(f"  Overall: {success_probability*100:.1f}%")
    
    # 리스크 조정 수익
    risk_adjusted_profit = mock_opportunity.net_profit * success_probability
    print(f"\n[Risk-Adjusted Profit]")
    print(f"Expected: {mock_opportunity.net_profit:,.0f} KRW")
    print(f"Risk-adjusted: {risk_adjusted_profit:,.0f} KRW")


def main():
    """
    메인 테스트 실행
    """
    print("\n[START] Starting Triangular Arbitrage Tests\n")
    
    # 실시간 테스트
    print("[1/2] Running real-time market test...")
    opportunities = asyncio.run(test_with_real_data())
    
    # 실행 시뮬레이션
    print("\n[2/2] Running execution simulation...")
    asyncio.run(test_execution_simulation())
    
    # 최종 평가
    print("\n" + "=" * 60)
    print("  FINAL ASSESSMENT")
    print("=" * 60)
    
    if opportunities and len(opportunities) > 0:
        avg_profit = np.mean([opp.profit_rate for opp in opportunities])
        if avg_profit > 0.15:
            print("\n[SUCCESS] VIABLE STRATEGY")
            print("Triangular arbitrage shows potential")
            print("Consider implementing with:")
            print("- Fast execution system")
            print("- Pre-positioned funds")
            print("- Automated monitoring")
        else:
            print("\n[WARNING] MARGINAL STRATEGY")
            print("Low profit margins detected")
            print("Success depends on:")
            print("- Very low fees")
            print("- Perfect execution")
            print("- High volume")
    else:
        print("\n[FAIL] NOT CURRENTLY VIABLE")
        print("Market is too efficient")
        print("Consider:")
        print("- Other arbitrage strategies")
        print("- Different coin pairs")
        print("- Market making instead")
    
    print("\nNext recommended step: Test ETH and XRP pairs for better opportunities")


if __name__ == "__main__":
    main()