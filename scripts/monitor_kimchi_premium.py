#!/usr/bin/env python3
"""
실시간 김치 프리미엄 모니터링
WebSocket 데이터와 연동하여 실시간 김프 계산
"""

import asyncio
import json
import signal
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.kimchi_premium import kimchi_calculator
from src.data.exchange_rate_manager import rate_manager
from src.data.websocket_manager import ws_manager
from src.utils.logger import logger


class KimchiPremiumMonitor:
    """실시간 김프 모니터"""

    def __init__(self):
        """초기화"""
        self.running = False
        self.upbit_price = None
        self.binance_price = None
        self.last_premium = None
        self.update_count = 0

    async def start(self):
        """모니터링 시작"""
        logger.info("Starting real-time kimchi premium monitoring")
        self.running = True

        # WebSocket 콜백 등록
        await self._register_callbacks()

        # WebSocket 연결
        await ws_manager.connect("upbit")
        await ws_manager.connect("binance")

        # BTC/KRW (Upbit) 및 BTC/USDT (Binance) 구독
        await ws_manager.subscribe_ticker("upbit", "BTC/KRW")
        await ws_manager.subscribe_ticker("binance", "BTC/USDT")

        # 환율 모니터링 시작 (별도 태스크)
        rate_task = asyncio.create_task(rate_manager.start_monitoring(interval=60))

        # 주기적 상태 출력
        status_task = asyncio.create_task(self._print_status())

        try:
            # 메인 루프
            while self.running:
                await asyncio.sleep(1)

                # 김프 계산 (가격이 모두 있을 때만)
                if self.upbit_price and self.binance_price:
                    await self._calculate_premium()

        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
        finally:
            self.running = False
            rate_task.cancel()
            status_task.cancel()
            await ws_manager.disconnect("upbit")
            await ws_manager.disconnect("binance")

    async def _register_callbacks(self):
        """WebSocket 콜백 등록"""

        async def on_ticker(data):
            """티커 업데이트 콜백"""
            try:
                if data["exchange"] == "upbit" and data["symbol"] == "BTC/KRW":
                    self.upbit_price = data["last"]
                    self.update_count += 1

                elif data["exchange"] == "binance" and data["symbol"] == "BTC/USDT":
                    self.binance_price = data["last"]
                    self.update_count += 1

            except Exception as e:
                logger.error(f"Error in ticker callback: {e}")

        ws_manager.register_callback("ticker", on_ticker)

    async def _calculate_premium(self):
        """김프 계산"""
        try:
            # 김프 계산
            data = await kimchi_calculator.calculate_premium(self.upbit_price, self.binance_price)

            self.last_premium = data

            # 시그널에 따른 로깅
            if data.signal.value in ["strong_buy", "strong_sell", "anomaly"]:
                logger.warning(
                    f"SIGNAL: {data.signal.value.upper()} | "
                    f"Premium: {data.premium_rate:.2f}% | "
                    f"Confidence: {data.confidence:.2%}"
                )

        except Exception as e:
            logger.error(f"Error calculating premium: {e}")

    async def _print_status(self):
        """주기적 상태 출력"""
        while self.running:
            await asyncio.sleep(10)  # 10초마다

            if self.last_premium:
                print("\n" + "=" * 60)
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Kimchi Premium Status")
                print("-" * 60)

                print(f"Upbit BTC:     {self.upbit_price:,.0f} KRW")
                print(f"Binance BTC:   {self.binance_price:,.2f} USDT")
                print(f"Exchange Rate: {self.last_premium.exchange_rate:.2f} KRW/USD")
                print()
                print(f"Premium Rate:  {self.last_premium.premium_rate:.2f}%")
                print(f"Premium KRW:   {self.last_premium.premium_krw:,.0f} KRW")
                print(f"Signal:        {self.last_premium.signal.value}")
                print(f"Confidence:    {self.last_premium.confidence:.2%}")
                print(f"Updates:       {self.update_count}")

                # 통계
                stats = kimchi_calculator.get_statistics()
                if stats["daily_high"] is not None:
                    print()
                    print("Daily Statistics:")
                    print(f"  High:  {stats['daily_high']:.2f}%")
                    print(f"  Low:   {stats['daily_low']:.2f}%")
                    print(f"  Avg:   {stats['daily_avg']:.2f}%")

                    if stats["volatility_24h"]:
                        print(f"  Vol:   {stats['volatility_24h']:.2f}")

                    if stats["ma_signal"]:
                        print(f"  MA:    {stats['ma_signal']}")

                print("=" * 60)


async def main():
    """메인 함수"""
    print("\n" + "=" * 60)
    print("REAL-TIME KIMCHI PREMIUM MONITOR")
    print("=" * 60)
    print("\nConnecting to exchanges...")
    print("Press Ctrl+C to stop\n")

    monitor = KimchiPremiumMonitor()

    # 시그널 핸들러
    def signal_handler(sig, frame):
        print("\n\nShutting down...")
        monitor.running = False

    signal.signal(signal.SIGINT, signal_handler)

    # 모니터링 시작
    await monitor.start()

    print("\nMonitoring stopped.")


if __name__ == "__main__":
    asyncio.run(main())
