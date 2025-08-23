#!/usr/bin/env python3
"""
API 연결 테스트 스크립트
실제 거래소 API에 연결하여 인증 확인
"""

import sys
import os
from pathlib import Path
import ccxt
import time
from datetime import datetime

# 프로젝트 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.env_manager import EnvManager


class APIConnectionTester:
    """API 연결 테스터"""
    
    def __init__(self):
        self.env = EnvManager()
        
    def test_upbit_connection(self) -> bool:
        """업비트 API 연결 테스트"""
        print("\n[업비트 API 테스트]")
        
        try:
            creds = self.env.get_upbit_credentials()
            
            # CCXT로 업비트 연결
            exchange = ccxt.upbit({
                'apiKey': creds.access_key,
                'secret': creds.secret_key,
                'enableRateLimit': True,
            })
            
            # 1. 서버 시간 확인
            print("  1. 서버 연결 확인...", end="")
            server_time = exchange.fetch_time()
            print(f" [OK] (서버 시간: {datetime.fromtimestamp(server_time/1000)})")
            
            # 2. 잔고 조회 (인증 필요)
            print("  2. 인증 확인 (잔고 조회)...", end="")
            balance = exchange.fetch_balance()
            krw_balance = balance.get('KRW', {}).get('free', 0)
            print(f" [OK] (KRW 잔고: {krw_balance:,.0f}원)")
            
            # 3. 마켓 정보 조회
            print("  3. 마켓 정보 조회...", end="")
            markets = exchange.fetch_markets()
            btc_market = next((m for m in markets if m['symbol'] == 'BTC/KRW'), None)
            if btc_market:
                print(f" [OK] (BTC/KRW 마켓 확인)")
            else:
                print(f" [WARN] (BTC/KRW 마켓 없음)")
            
            # 4. 현재가 조회
            print("  4. BTC 현재가 조회...", end="")
            ticker = exchange.fetch_ticker('BTC/KRW')
            print(f" [OK] (현재가: {ticker['last']:,.0f}원)")
            
            print("\n[SUCCESS] 업비트 API 연결 성공!")
            return True
            
        except ccxt.BaseError as e:
            print(f"\n[FAIL] CCXT 오류: {e}")
            return False
        except Exception as e:
            print(f"\n[FAIL] 업비트 연결 실패: {e}")
            return False
    
    def test_binance_connection(self) -> bool:
        """바이낸스 API 연결 테스트"""
        print("\n[바이낸스 API 테스트]")
        
        try:
            creds = self.env.get_binance_credentials()
            
            # CCXT로 바이낸스 연결
            exchange = ccxt.binance({
                'apiKey': creds.access_key,
                'secret': creds.secret_key,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future',  # 선물 거래
                }
            })
            
            # 1. 서버 시간 확인
            print("  1. 서버 연결 확인...", end="")
            server_time = exchange.fetch_time()
            print(f" [OK] (서버 시간: {datetime.fromtimestamp(server_time/1000)})")
            
            # 2. 잔고 조회 (인증 필요)
            print("  2. 인증 확인 (잔고 조회)...", end="")
            balance = exchange.fetch_balance()
            usdt_balance = balance.get('USDT', {}).get('free', 0)
            print(f" [OK] (USDT 잔고: {usdt_balance:,.2f})")
            
            # 3. 선물 마켓 정보 조회
            print("  3. 선물 마켓 정보 조회...", end="")
            markets = exchange.fetch_markets()
            btc_perp = next((m for m in markets if m['symbol'] == 'BTC/USDT:USDT'), None)
            if btc_perp:
                print(f" [OK] (BTC/USDT 무기한 선물 확인)")
            else:
                print(f" [WARN] (BTC/USDT 선물 마켓 없음)")
            
            # 4. 현재가 조회
            print("  4. BTC 선물 현재가 조회...", end="")
            ticker = exchange.fetch_ticker('BTC/USDT')
            print(f" [OK] (현재가: ${ticker['last']:,.2f})")
            
            # 5. 펀딩비 조회
            print("  5. 펀딩비 조회...", end="")
            funding_rate = exchange.fetch_funding_rate('BTC/USDT')
            print(f" [OK] (펀딩비: {funding_rate['fundingRate']*100:.4f}%)")
            
            print("\n[SUCCESS] 바이낸스 API 연결 성공!")
            return True
            
        except ccxt.BaseError as e:
            print(f"\n[FAIL] CCXT 오류: {e}")
            if 'API-key' in str(e):
                print("  -> API 키가 잘못되었거나 권한이 없습니다")
                print("  -> 바이낸스에서 API 권한 설정을 확인하세요:")
                print("     - Enable Futures 체크")
                print("     - Enable Reading 체크")
            return False
        except Exception as e:
            print(f"\n[FAIL] 바이낸스 연결 실패: {e}")
            return False
    
    def test_exchange_rate_api(self) -> bool:
        """환율 API 테스트"""
        print("\n[환율 API 테스트]")
        
        try:
            import requests
            
            api_key = self.env.get_exchange_rate_api_key()
            
            if api_key == "free_tier":
                print("  무료 API 사용 (exchangerate-api.com)")
                url = "https://api.exchangerate-api.com/v4/latest/USD"
            else:
                # 유료 API 사용 시
                url = f"https://api.exchangerate-api.com/v4/latest/USD?access_key={api_key}"
            
            print("  환율 데이터 요청...", end="")
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                krw_rate = data.get('rates', {}).get('KRW')
                
                if krw_rate:
                    print(f" [OK] (USD/KRW: {krw_rate:,.2f}원)")
                    print(f"  마지막 업데이트: {data.get('date', 'N/A')}")
                    return True
                else:
                    print(f" [FAIL] KRW 환율 데이터 없음")
                    return False
            else:
                print(f" [FAIL] HTTP {response.status_code}")
                return False
                
        except Exception as e:
            print(f"\n[FAIL] 환율 API 오류: {e}")
            return False
    
    def test_all(self) -> bool:
        """모든 API 테스트"""
        print("=" * 50)
        print("API 연결 종합 테스트")
        print("=" * 50)
        
        results = {
            'upbit': False,
            'binance': False,
            'exchange_rate': False
        }
        
        # 업비트 테스트
        try:
            results['upbit'] = self.test_upbit_connection()
        except Exception as e:
            print(f"[ERROR] 업비트 테스트 중 오류: {e}")
        
        time.sleep(1)  # Rate limit 방지
        
        # 바이낸스 테스트
        try:
            results['binance'] = self.test_binance_connection()
        except Exception as e:
            print(f"[ERROR] 바이낸스 테스트 중 오류: {e}")
        
        time.sleep(1)
        
        # 환율 API 테스트
        try:
            results['exchange_rate'] = self.test_exchange_rate_api()
        except Exception as e:
            print(f"[ERROR] 환율 API 테스트 중 오류: {e}")
        
        # 결과 요약
        print("\n" + "=" * 50)
        print("테스트 결과 요약")
        print("=" * 50)
        
        for name, success in results.items():
            status = "[OK]" if success else "[FAIL]"
            print(f"  {status} {name.upper()}")
        
        all_success = all(results.values())
        
        if all_success:
            print("\n[SUCCESS] 모든 API 연결 테스트 통과!")
            print("시스템을 시작할 준비가 되었습니다.")
        else:
            print("\n[WARNING] 일부 API 연결 실패")
            print("실패한 API의 키와 권한을 확인하세요.")
        
        return all_success


def main():
    """메인 실행"""
    tester = APIConnectionTester()
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "upbit":
            tester.test_upbit_connection()
        elif sys.argv[1] == "binance":
            tester.test_binance_connection()
        elif sys.argv[1] == "rate":
            tester.test_exchange_rate_api()
        else:
            print(f"알 수 없는 옵션: {sys.argv[1]}")
            print("사용법: python test_api_connection.py [upbit|binance|rate]")
    else:
        # 전체 테스트
        tester.test_all()


if __name__ == "__main__":
    main()