"""
환율 데이터 관리자
실시간 및 히스토리컬 USD/KRW 환율 데이터 수집
"""

import asyncio
import json
import os
from collections import deque
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

import aiohttp

from src.models.schemas import ExchangeRate
from src.utils.logger import logger
from src.utils.metrics import metrics_collector


class RateSource(Enum):
    """환율 데이터 소스"""

    EXCHANGERATE_API = "exchangerate-api"  # 무료: 1500회/월
    FIXER_IO = "fixer.io"  # 유료
    CURRENCYLAYER = "currencylayer"  # 유료
    KOREAN_BANK = "korean_bank"  # 한국은행
    FALLBACK = "fallback"  # 고정값 fallback


class ExchangeRateManager:
    """환율 관리자"""

    def __init__(self):
        """초기화"""
        # API 키 설정
        self.api_keys = {
            RateSource.EXCHANGERATE_API: os.getenv("EXCHANGERATE_API_KEY"),
            RateSource.FIXER_IO: os.getenv("FIXER_API_KEY"),
            RateSource.CURRENCYLAYER: os.getenv("CURRENCYLAYER_API_KEY"),
        }

        # 캐시 설정
        self.cache = {}
        self.cache_duration = 60  # 60초 캐시
        self.last_update = None

        # 히스토리 버퍼 (최근 24시간)
        self.rate_history = deque(maxlen=1440)  # 분당 1개, 24시간

        # 현재 환율
        self.current_rate = None
        self.fallback_rate = 1385.0  # 기본 fallback 값

        # API 호출 카운터
        self.api_call_count = {}
        self.api_limits = {
            RateSource.EXCHANGERATE_API: 1500,  # 월간
            RateSource.FIXER_IO: 10000,  # 월간
            RateSource.CURRENCYLAYER: 10000,  # 월간
        }

        logger.info("Exchange rate manager initialized")

    async def get_current_rate(self, force_refresh: bool = False) -> Optional[float]:
        """
        현재 환율 조회

        Args:
            force_refresh: 캐시 무시하고 새로 가져오기

        Returns:
            USD/KRW 환율
        """
        # 캐시 체크
        if not force_refresh and self._is_cache_valid():
            return self.current_rate

        # 우선순위대로 시도
        sources = [
            RateSource.EXCHANGERATE_API,
            RateSource.KOREAN_BANK,
            RateSource.FIXER_IO,
            RateSource.CURRENCYLAYER,
        ]

        for source in sources:
            try:
                rate = await self._fetch_from_source(source)
                if rate:
                    self.current_rate = rate
                    self.last_update = datetime.now()

                    # 히스토리에 추가
                    self.rate_history.append(
                        {"timestamp": self.last_update, "rate": rate, "source": source.value}
                    )

                    # 메트릭 업데이트
                    metrics_collector.update_balance("exchange_rate", "USD/KRW", rate)

                    logger.info(f"Exchange rate updated: {rate:.2f} KRW/USD from {source.value}")
                    return rate

            except Exception as e:
                logger.warning(f"Failed to fetch from {source.value}: {e}")
                continue

        # 모든 소스 실패 시 fallback
        logger.warning(f"All sources failed, using fallback rate: {self.fallback_rate}")
        self.current_rate = self.fallback_rate
        return self.fallback_rate

    async def _fetch_from_source(self, source: RateSource) -> Optional[float]:
        """특정 소스에서 환율 가져오기"""

        # API 키가 딕셔너리 자체가 비어있으면 모든 소스 실패
        if not self.api_keys:
            return None

        if source == RateSource.EXCHANGERATE_API:
            return await self._fetch_exchangerate_api()
        elif source == RateSource.KOREAN_BANK:
            return await self._fetch_korean_bank()
        elif source == RateSource.FIXER_IO:
            return await self._fetch_fixer_io()
        elif source == RateSource.CURRENCYLAYER:
            return await self._fetch_currencylayer()
        else:
            return None

    async def _fetch_exchangerate_api(self) -> Optional[float]:
        """ExchangeRate-API에서 환율 가져오기"""
        api_key = self.api_keys.get(RateSource.EXCHANGERATE_API)

        # API 키 없으면 무료 버전 사용
        if api_key:
            url = f"https://v6.exchangerate-api.com/v6/{api_key}/pair/USD/KRW"
        else:
            url = "https://api.exchangerate-api.com/v4/latest/USD"

        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()

                    if api_key:
                        # v6 API 응답
                        return float(data.get("conversion_rate", 0))
                    else:
                        # v4 API 응답 (무료)
                        rates = data.get("rates", {})
                        return float(rates.get("KRW", 0))

        return None

    async def _fetch_korean_bank(self) -> Optional[float]:
        """한국은행 API에서 환율 가져오기"""
        # 한국은행 API는 인증이 필요하고 복잡하므로
        # 실제 구현 시 별도 처리 필요
        # 여기서는 더미 구현

        # 실제 구현 예시:
        # url = "https://www.koreaexim.go.kr/site/program/financial/exchangeJSON"
        # params = {
        #     'authkey': self.api_keys.get('KOREAN_BANK_KEY'),
        #     'searchdate': datetime.now().strftime('%Y%m%d'),
        #     'data': 'AP01'
        # }

        logger.debug("Korean Bank API not implemented, skipping")
        return None

    async def _fetch_fixer_io(self) -> Optional[float]:
        """Fixer.io에서 환율 가져오기"""
        api_key = self.api_keys.get(RateSource.FIXER_IO)
        if not api_key:
            return None

        url = f"http://data.fixer.io/api/latest"
        params = {"access_key": api_key, "base": "USD", "symbols": "KRW"}

        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get("success"):
                        rates = data.get("rates", {})
                        return float(rates.get("KRW", 0))

        return None

    async def _fetch_currencylayer(self) -> Optional[float]:
        """CurrencyLayer에서 환율 가져오기"""
        api_key = self.api_keys.get(RateSource.CURRENCYLAYER)
        if not api_key:
            return None

        url = "http://api.currencylayer.com/live"
        params = {"access_key": api_key, "source": "USD", "currencies": "KRW"}

        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get("success"):
                        quotes = data.get("quotes", {})
                        return float(quotes.get("USDKRW", 0))

        return None

    def _is_cache_valid(self) -> bool:
        """캐시 유효성 체크"""
        if not self.current_rate or not self.last_update:
            return False

        elapsed = (datetime.now() - self.last_update).total_seconds()
        return elapsed < self.cache_duration

    async def get_historical_rates(
        self, start_date: datetime, end_date: datetime
    ) -> List[ExchangeRate]:
        """
        히스토리컬 환율 데이터 조회

        Args:
            start_date: 시작일
            end_date: 종료일

        Returns:
            환율 데이터 리스트
        """
        rates = []

        # 캐시된 히스토리에서 먼저 찾기
        for item in self.rate_history:
            if start_date <= item["timestamp"] <= end_date:
                rates.append(
                    ExchangeRate(
                        timestamp=item["timestamp"], usd_krw=item["rate"], source=item["source"]
                    )
                )

        # 부족한 데이터는 API에서 가져오기
        if len(rates) < (end_date - start_date).days:
            # 히스토리컬 API 호출 (구현 필요)
            logger.warning("Historical rate data API not fully implemented")

        return rates

    def get_average_rate(self, hours: int = 24) -> Optional[float]:
        """
        최근 N시간 평균 환율

        Args:
            hours: 시간 수

        Returns:
            평균 환율
        """
        if not self.rate_history:
            return None

        cutoff = datetime.now() - timedelta(hours=hours)
        recent_rates = [item["rate"] for item in self.rate_history if item["timestamp"] > cutoff]

        if recent_rates:
            return sum(recent_rates) / len(recent_rates)

        return self.current_rate

    def get_rate_volatility(self, hours: int = 24) -> Optional[float]:
        """
        환율 변동성 계산

        Args:
            hours: 시간 수

        Returns:
            변동성 (표준편차)
        """
        if not self.rate_history:
            return None

        cutoff = datetime.now() - timedelta(hours=hours)
        recent_rates = [item["rate"] for item in self.rate_history if item["timestamp"] > cutoff]

        if len(recent_rates) < 2:
            return 0.0

        # 표준편차 계산
        avg = sum(recent_rates) / len(recent_rates)
        variance = sum((r - avg) ** 2 for r in recent_rates) / len(recent_rates)
        return variance**0.5

    async def start_monitoring(self, interval: int = 60):
        """
        환율 모니터링 시작

        Args:
            interval: 업데이트 간격 (초)
        """
        logger.info(f"Starting exchange rate monitoring (interval: {interval}s)")

        while True:
            try:
                await self.get_current_rate()

                # 변동성 체크
                volatility = self.get_rate_volatility(hours=1)
                if volatility and volatility > 10:  # 10원 이상 변동
                    logger.warning(f"High exchange rate volatility: {volatility:.2f} KRW")

                await asyncio.sleep(interval)

            except Exception as e:
                logger.error(f"Error in rate monitoring: {e}")
                await asyncio.sleep(interval)

    def set_fallback_rate(self, rate: float):
        """Fallback 환율 설정"""
        self.fallback_rate = rate
        logger.info(f"Fallback rate set to: {rate:.2f}")

    def get_status(self) -> Dict[str, Any]:
        """환율 관리자 상태"""
        return {
            "current_rate": self.current_rate,
            "last_update": self.last_update.isoformat() if self.last_update else None,
            "fallback_rate": self.fallback_rate,
            "history_size": len(self.rate_history),
            "average_24h": self.get_average_rate(24),
            "volatility_24h": self.get_rate_volatility(24),
            "cache_valid": self._is_cache_valid(),
        }


# 전역 환율 관리자
rate_manager = ExchangeRateManager()


if __name__ == "__main__":
    # 환율 관리자 테스트
    async def test():
        print("Exchange Rate Manager Test")
        print("-" * 40)

        # 현재 환율 가져오기
        rate = await rate_manager.get_current_rate()
        print(f"Current USD/KRW rate: {rate:.2f}")

        # 평균 환율
        avg_rate = rate_manager.get_average_rate(1)
        if avg_rate:
            print(f"Average rate (1h): {avg_rate:.2f}")

        # 상태 확인
        status = rate_manager.get_status()
        print(f"\nStatus: {json.dumps(status, indent=2, default=str)}")

        # 모니터링 테스트 (5초만)
        print("\nStarting monitoring for 5 seconds...")
        try:
            await asyncio.wait_for(rate_manager.start_monitoring(interval=2), timeout=5)
        except asyncio.TimeoutError:
            print("Monitoring test completed")

    asyncio.run(test())
