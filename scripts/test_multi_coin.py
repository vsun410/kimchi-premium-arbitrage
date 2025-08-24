"""
Test Multi-Coin Arbitrage Opportunities
멀티코인 차익거래 기회 테스트
"""

import sys
import os
import asyncio
import ccxt.pro as ccxtpro
from datetime import datetime
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.strategies.multi_coin_arbitrage import MultiCoinArbitrage, ArbitrageOpportunity
from src.utils.exchange_rate_manager import get_exchange_rate_manager
from src.utils.logger import logger


async def test_multi_coin_arbitrage():
    """
    BTC, ETH, XRP 차익거래 기회 테스트
    """
    
    print("\n" + "=" * 60)
    print("  MULTI-COIN ARBITRAGE TEST")
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
    
    # 멀티코인 전략 초기화
    strategy = MultiCoinArbitrage(
        upbit_client=upbit,
        binance_client=binance,
        min_profit_rate=0.15  # 0.15% 최소 수익률
    )
    
    # 5회 테스트
    all_opportunities = []
    coin_stats = {coin: {'premium': [], 'volume': []} for coin in ['BTC', 'ETH', 'XRP']}
    
    for test_num in range(5):
        print(f"\n[Test {test_num+1}/5]")
        
        try:
            # 모든 코인 가격 수집
            all_prices = await strategy.fetch_all_prices()
            
            # 각 코인별 상태 출력
            for coin, prices in all_prices.items():
                if prices:
                    kimchi_premium = strategy.calculate_kimchi_premium(coin, prices, current_rate)
                    
                    print(f"\n{coin}:")
                    print(f"  Upbit: {prices['upbit_price']:,.0f} KRW")
                    print(f"  Binance: {prices['binance_price']:,.2f} USDT")
                    print(f"  Kimchi Premium: {kimchi_premium:.3f}%")
                    
                    # 통계 수집
                    coin_stats[coin]['premium'].append(kimchi_premium)
                    coin_stats[coin]['volume'].append(prices['upbit_volume'])
                    
                    # 스프레드 계산
                    upbit_spread = ((prices['upbit_ask'] - prices['upbit_bid']) / prices['upbit_bid']) * 100
                    binance_spread = ((prices['binance_ask'] - prices['binance_bid']) / prices['binance_bid']) * 100
                    print(f"  Spread (Upbit): {upbit_spread:.3f}%")
                    print(f"  Spread (Binance): {binance_spread:.3f}%")
            
            # 차익거래 기회 찾기
            opportunities = strategy.find_opportunities(all_prices, current_rate)
            
            if opportunities:
                print(f"\n[FOUND] {len(opportunities)} opportunities!")
                for opp in opportunities[:3]:  # 상위 3개만
                    print(f"  {opp.coin}: {opp.strategy} - {opp.profit_rate:.3f}% profit")
                    all_opportunities.extend(opportunities)
            else:
                print("\n[No opportunities found this round]")
                
        except Exception as e:
            print(f"Error in test {test_num+1}: {e}")
            
        # 대기
        await asyncio.sleep(5)
    
    # 거래소 종료
    await upbit.close()
    await binance.close()
    
    # 결과 분석
    print("\n" + "=" * 60)
    print("  TEST RESULTS ANALYSIS")
    print("=" * 60)
    
    # 코인별 김프 통계
    print("\n[Kimchi Premium Statistics]")
    for coin in ['BTC', 'ETH', 'XRP']:
        if coin_stats[coin]['premium']:
            premiums = coin_stats[coin]['premium']
            print(f"\n{coin}:")
            print(f"  Average: {np.mean(premiums):.3f}%")
            print(f"  Max: {np.max(premiums):.3f}%")
            print(f"  Min: {np.min(premiums):.3f}%")
            print(f"  Std Dev: {np.std(premiums):.3f}%")
    
    # 기회 분석
    if all_opportunities:
        df = pd.DataFrame([{
            'coin': opp.coin,
            'strategy': opp.strategy,
            'profit_rate': opp.profit_rate,
            'risk_score': opp.risk_score,
            'net_profit': opp.net_profit
        } for opp in all_opportunities])
        
        print("\n[Opportunity Analysis]")
        print(f"Total opportunities found: {len(all_opportunities)}")
        
        print("\n[By Coin]")
        for coin in ['BTC', 'ETH', 'XRP']:
            coin_opps = df[df['coin'] == coin]
            if len(coin_opps) > 0:
                print(f"{coin}: {len(coin_opps)} opportunities, avg profit {coin_opps['profit_rate'].mean():.3f}%")
        
        print("\n[By Strategy]")
        for strategy in df['strategy'].unique():
            strat_opps = df[df['strategy'] == strategy]
            print(f"{strategy}: {len(strat_opps)} times, avg profit {strat_opps['profit_rate'].mean():.3f}%")
        
        # 최고 수익 기회
        best = df.loc[df['profit_rate'].idxmax()]
        print(f"\n[Best Opportunity]")
        print(f"Coin: {best['coin']}")
        print(f"Strategy: {best['strategy']}")
        print(f"Profit rate: {best['profit_rate']:.3f}%")
        print(f"Risk score: {best['risk_score']:.2f}")
        
    else:
        print("\n[No opportunities found during test period]")
    
    return all_opportunities, coin_stats


async def compare_coin_characteristics():
    """
    코인별 특성 비교 분석
    """
    print("\n" + "=" * 60)
    print("  COIN CHARACTERISTICS COMPARISON")
    print("=" * 60)
    
    upbit = ccxtpro.upbit()
    binance = ccxtpro.binance()
    
    try:
        # 각 코인의 24시간 통계
        print("\n[24h Statistics]")
        
        for coin in ['BTC', 'ETH', 'XRP']:
            upbit_ticker = await upbit.fetch_ticker(f"{coin}/KRW")
            binance_ticker = await binance.fetch_ticker(f"{coin}/USDT")
            
            print(f"\n{coin}:")
            print(f"  24h Change (Upbit): {upbit_ticker['percentage']:.2f}%")
            print(f"  24h Change (Binance): {binance_ticker['percentage']:.2f}%")
            print(f"  24h Volume (Upbit): {upbit_ticker['quoteVolume']:,.0f} KRW")
            print(f"  24h Volume (Binance): {binance_ticker['quoteVolume']:,.0f} USDT")
            
            # 변동성 계산 (high-low / average)
            upbit_volatility = ((upbit_ticker['high'] - upbit_ticker['low']) / upbit_ticker['vwap']) * 100
            binance_volatility = ((binance_ticker['high'] - binance_ticker['low']) / binance_ticker['vwap']) * 100
            
            print(f"  Volatility (Upbit): {upbit_volatility:.2f}%")
            print(f"  Volatility (Binance): {binance_volatility:.2f}%")
        
    finally:
        await upbit.close()
        await binance.close()


def calculate_monthly_projection(opportunities, coin_stats):
    """
    월간 수익 예상 계산
    """
    print("\n" + "=" * 60)
    print("  MONTHLY PROFIT PROJECTION")
    print("=" * 60)
    
    if not opportunities:
        print("\nNo data for projection")
        return
    
    # 코인별 계산
    for coin in ['BTC', 'ETH', 'XRP']:
        coin_opps = [opp for opp in opportunities if opp.coin == coin]
        
        if coin_opps:
            avg_profit_rate = np.mean([opp.profit_rate for opp in coin_opps])
            avg_net_profit = np.mean([opp.net_profit for opp in coin_opps])
            frequency = len(coin_opps) / 5  # 5 tests로 나눔
            
            # 하루 10회 거래 가정 (보수적)
            daily_trades = 10 * frequency
            daily_profit = avg_net_profit * daily_trades
            monthly_profit = daily_profit * 30
            
            # 필요 자본 계산
            if coin == 'BTC':
                required_capital = 0.01 * 159000000  # 0.01 BTC
            elif coin == 'ETH':
                required_capital = 0.5 * 5500000  # 0.5 ETH
            else:  # XRP
                required_capital = 10000 * 920  # 10000 XRP
            
            monthly_return = (monthly_profit / required_capital) * 100
            
            print(f"\n{coin}:")
            print(f"  Opportunity frequency: {frequency*100:.1f}% of time")
            print(f"  Average profit per trade: {avg_net_profit:,.0f} KRW")
            print(f"  Daily profit (10 trades): {daily_profit:,.0f} KRW")
            print(f"  Monthly profit: {monthly_profit:,.0f} KRW")
            print(f"  Required capital: {required_capital:,.0f} KRW")
            print(f"  Monthly ROI: {monthly_return:.2f}%")


def main():
    """
    메인 테스트 실행
    """
    print("\n[START] Multi-Coin Arbitrage Testing\n")
    
    # 1. 멀티코인 차익거래 테스트
    print("[1/3] Testing arbitrage opportunities...")
    opportunities, coin_stats = asyncio.run(test_multi_coin_arbitrage())
    
    # 2. 코인 특성 비교
    print("\n[2/3] Comparing coin characteristics...")
    asyncio.run(compare_coin_characteristics())
    
    # 3. 월간 수익 예상
    print("\n[3/3] Calculating monthly projections...")
    calculate_monthly_projection(opportunities, coin_stats)
    
    # 최종 평가
    print("\n" + "=" * 60)
    print("  FINAL ASSESSMENT")
    print("=" * 60)
    
    # 코인별 점수 계산
    scores = {}
    for coin in ['BTC', 'ETH', 'XRP']:
        score = 0
        
        # 김프 평균 (높을수록 좋음)
        if coin_stats[coin]['premium']:
            avg_premium = np.mean(coin_stats[coin]['premium'])
            score += min(avg_premium * 10, 30)  # 최대 30점
        
        # 기회 빈도
        coin_opps = [opp for opp in opportunities if opp.coin == coin]
        frequency_score = (len(coin_opps) / max(len(opportunities), 1)) * 30  # 최대 30점
        score += frequency_score
        
        # 특성 점수
        if coin == 'BTC':
            score += 30  # 유동성 최고
        elif coin == 'ETH':
            score += 20  # 균형
        else:  # XRP
            score += 25  # 빠른 전송
        
        scores[coin] = score
    
    # 순위 출력
    print("\n[Coin Rankings for Arbitrage]")
    sorted_coins = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for i, (coin, score) in enumerate(sorted_coins, 1):
        print(f"{i}. {coin}: {score:.1f} points")
    
    # 추천사항
    print("\n[Recommendations]")
    best_coin = sorted_coins[0][0]
    print(f"1. Focus on {best_coin} for best opportunities")
    print("2. Use XRP for quick inter-exchange transfers")
    print("3. Keep BTC positions for stability")
    print("4. Monitor ETH during high volatility periods")
    
    if opportunities:
        avg_profit = np.mean([opp.profit_rate for opp in opportunities])
        if avg_profit > 0.2:
            print("\n[VERDICT] Multi-coin strategy shows promise")
            print("Consider implementing with proper risk management")
        else:
            print("\n[VERDICT] Limited opportunities detected")
            print("Market efficiency is high across all coins")
    else:
        print("\n[VERDICT] No viable opportunities found")
        print("Consider alternative strategies")


if __name__ == "__main__":
    main()