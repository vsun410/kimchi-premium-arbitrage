#!/usr/bin/env python3
"""
현재 환율 확인 스크립트
"""

import sys
import asyncio
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.exchange_rate_manager import rate_manager


async def check_rate():
    print("\n" + "=" * 50)
    print("Current USD/KRW Exchange Rate")
    print("=" * 50)
    
    # 환율 가져오기
    rate = await rate_manager.get_current_rate()
    
    if rate:
        print(f"\nCurrent Rate: {rate:.2f} KRW/USD")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 1달러로 살 수 있는 원화
        print(f"\n[USD to KRW Conversion]")
        print(f"1 USD = {rate:.2f} KRW")
        print(f"100 USD = {rate*100:,.0f} KRW")
        print(f"1,000 USD = {rate*1000:,.0f} KRW")
        
        # 원화로 살 수 있는 달러
        print(f"\n[KRW to USD Conversion]")
        print(f"1,000,000 KRW = {1000000/rate:,.2f} USD")
        print(f"10,000,000 KRW = {10000000/rate:,.2f} USD")
        
        # 상태 확인
        status = rate_manager.get_status()
        print(f"\nData Source: ExchangeRate-API (Free)")
        print(f"Last Update: {status.get('last_update', 'N/A')}")
        
    else:
        print("\nFailed to fetch exchange rate.")
    
    print("\n" + "=" * 50)


if __name__ == "__main__":
    asyncio.run(check_rate())