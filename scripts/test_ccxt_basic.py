#!/usr/bin/env python3
"""
CCXT Pro 기본 연결 테스트
"""

import asyncio

import ccxt.pro as ccxt


async def test_upbit():
    """Upbit 연결 테스트"""
    print("Testing Upbit connection...")
    exchange = ccxt.upbit()

    try:
        # 마켓 정보 로드
        markets = await exchange.load_markets()
        print(f"  Upbit markets loaded: {len(markets)} pairs")

        # BTC/KRW 티커 가져오기
        ticker = await exchange.fetch_ticker("BTC/KRW")
        print(f"  BTC/KRW Price: {ticker['last']:,.0f} KRW")

        # 지원되는 timeframes 확인
        print(f"  Supported timeframes: {exchange.timeframes}")

    except Exception as e:
        print(f"  Error: {e}")
    finally:
        await exchange.close()


async def test_binance():
    """Binance 연결 테스트"""
    print("\nTesting Binance connection...")
    exchange = ccxt.binance({"options": {"defaultType": "future"}})

    try:
        # 마켓 정보 로드
        markets = await exchange.load_markets()
        futures = [m for m in markets.values() if m["future"]]
        print(f"  Binance futures loaded: {len(futures)} pairs")

        # BTC/USDT 선물 티커 가져오기
        ticker = await exchange.fetch_ticker("BTC/USDT")
        print(f"  BTC/USDT Price: ${ticker['last']:,.2f}")

        # 지원되는 timeframes 확인
        print(f"  Supported timeframes: {exchange.timeframes}")

    except Exception as e:
        print(f"  Error: {e}")
    finally:
        await exchange.close()


async def test_websocket():
    """WebSocket 스트림 테스트"""
    print("\nTesting WebSocket streams...")

    # Upbit WebSocket
    upbit = ccxt.upbit()
    binance = ccxt.binance()

    try:
        print("  Watching BTC/KRW ticker on Upbit...")
        upbit_ticker = await upbit.watch_ticker("BTC/KRW")
        upbit_price = upbit_ticker["last"]
        print(f"    Upbit BTC/KRW: {upbit_price:,.0f} KRW")

        print("  Watching BTC/USDT ticker on Binance...")
        binance_ticker = await binance.watch_ticker("BTC/USDT")
        binance_price = binance_ticker["last"]
        print(f"    Binance BTC/USDT: ${binance_price:,.2f}")

        # 김프 계산 (간단 버전)
        usd_krw = 1385  # 임시 환율 (실제 환율에 가깝게)

        if upbit_price and binance_price:
            kimchi_premium = ((upbit_price / (binance_price * usd_krw)) - 1) * 100
            print(f"\n  Estimated Kimchi Premium: {kimchi_premium:.2f}%")
            print(f"  (Using USD/KRW rate: {usd_krw})")

    except Exception as e:
        print(f"  WebSocket Error: {e}")
    finally:
        await upbit.close()
        await binance.close()


async def main():
    print("=" * 50)
    print("CCXT Pro Basic Connection Test")
    print("=" * 50)

    # REST API 테스트
    await test_upbit()
    await test_binance()

    # WebSocket 테스트 (짧게)
    try:
        await asyncio.wait_for(test_websocket(), timeout=5)
    except asyncio.TimeoutError:
        print("\n  WebSocket test completed (timeout)")

    print("\n" + "=" * 50)
    print("Test completed!")


if __name__ == "__main__":
    asyncio.run(main())
