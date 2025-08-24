"""
실시간 환율 데이터 수집기
여러 소스에서 USD/KRW 환율을 수집하고 검증

목적: 정확한 실시간 환율 제공
결과: 하드코딩 없이 항상 최신 환율 사용
평가: 다중 소스 검증으로 신뢰성 확보
"""

import asyncio
import aiohttp
from typing import Dict, Optional, List, Tuple
from datetime import datetime, timedelta
import logging
import json
import statistics
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ExchangeRateData:
    """환율 데이터"""
    rate: float
    source: str
    timestamp: datetime
    confidence: float  # 0-1 신뢰도


class RealTimeExchangeRateFetcher:
    """
    실시간 환율 수집기
    
    핵심 원칙:
    1. 절대 하드코딩 금지
    2. 다중 소스에서 교차 검증
    3. 캐싱으로 API 호출 최소화
    """
    
    def __init__(self):
        """초기화"""
        self.sources = {
            'exchangerate-api': {
                'url': 'https://api.exchangerate-api.com/v4/latest/USD',
                'priority': 1,
                'free_tier': True
            },
            'fxratesapi': {
                'url': 'https://api.fxratesapi.com/latest',
                'priority': 2,
                'free_tier': True
            },
            'exchangerate.host': {
                'url': 'https://api.exchangerate.host/latest',
                'priority': 3,
                'free_tier': True
            }
        }
        
        # 캐시
        self.cache: Dict[str, ExchangeRateData] = {}
        self.cache_duration = timedelta(minutes=5)
        
        # 통계
        self.fetch_history: List[ExchangeRateData] = []
        self.max_history = 100
        
        # 검증 임계값
        self.max_deviation_pct = 2.0  # 소스 간 최대 2% 차이 허용
        
        logger.info("RealTimeExchangeRateFetcher initialized")
    
    async def get_current_rate(self, force_refresh: bool = False) -> float:
        """
        현재 USD/KRW 환율 조회
        
        Args:
            force_refresh: 캐시 무시하고 새로 조회
            
        Returns:
            현재 환율
            
        Raises:
            ValueError: 환율 조회 실패
        """
        # 캐시 확인 (force_refresh가 아닌 경우)
        if not force_refresh:
            cached_rate = self._get_cached_rate()
            if cached_rate is not None:
                logger.debug(f"Using cached rate: {cached_rate}")
                return cached_rate
        
        # 모든 소스에서 환율 수집
        rates = await self._fetch_from_all_sources()
        
        if not rates:
            # 캐시된 값이라도 반환 (오래되었더라도)
            if self.cache:
                oldest_rate = min(self.cache.values(), key=lambda x: x.timestamp)
                logger.warning(f"All sources failed, using old cached rate: {oldest_rate.rate}")
                return oldest_rate.rate
            
            raise ValueError("Failed to fetch exchange rate from any source")
        
        # 검증 및 평균 계산
        validated_rate = self._validate_and_average(rates)
        
        # 캐시 업데이트
        self._update_cache(validated_rate)
        
        # 이력 저장
        self._add_to_history(validated_rate)
        
        return validated_rate.rate
    
    async def _fetch_from_all_sources(self) -> List[ExchangeRateData]:
        """
        모든 소스에서 환율 수집
        
        Returns:
            수집된 환율 리스트
        """
        rates = []
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            for source_name, source_info in self.sources.items():
                tasks.append(
                    self._fetch_from_source(session, source_name, source_info)
                )
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, ExchangeRateData):
                    rates.append(result)
                elif isinstance(result, Exception):
                    logger.warning(f"Source fetch error: {result}")
        
        return rates
    
    async def _fetch_from_source(
        self,
        session: aiohttp.ClientSession,
        source_name: str,
        source_info: Dict
    ) -> ExchangeRateData:
        """
        특정 소스에서 환율 수집
        
        Args:
            session: HTTP 세션
            source_name: 소스 이름
            source_info: 소스 정보
            
        Returns:
            환율 데이터
        """
        try:
            if source_name == 'exchangerate-api':
                async with session.get(source_info['url']) as response:
                    data = await response.json()
                    rate = data['rates']['KRW']
                    
            elif source_name == 'fxratesapi':
                params = {'base': 'USD', 'currencies': 'KRW'}
                async with session.get(source_info['url'], params=params) as response:
                    data = await response.json()
                    rate = data['rates']['KRW']
                    
            elif source_name == 'exchangerate.host':
                params = {'base': 'USD', 'symbols': 'KRW'}
                async with session.get(source_info['url'], params=params) as response:
                    data = await response.json()
                    rate = data['rates']['KRW']
            
            else:
                raise ValueError(f"Unknown source: {source_name}")
            
            # 환율 유효성 체크 (합리적인 범위)
            if not (1000 < rate < 2000):
                raise ValueError(f"Invalid rate: {rate}")
            
            return ExchangeRateData(
                rate=rate,
                source=source_name,
                timestamp=datetime.now(),
                confidence=1.0 / source_info['priority']
            )
            
        except Exception as e:
            logger.error(f"Failed to fetch from {source_name}: {e}")
            raise
    
    def _validate_and_average(self, rates: List[ExchangeRateData]) -> ExchangeRateData:
        """
        수집된 환율 검증 및 평균 계산
        
        Args:
            rates: 수집된 환율 리스트
            
        Returns:
            검증된 평균 환율
        """
        if not rates:
            raise ValueError("No rates to validate")
        
        # 단일 소스인 경우
        if len(rates) == 1:
            logger.warning("Only one source available, using without validation")
            return rates[0]
        
        # 환율 값만 추출
        rate_values = [r.rate for r in rates]
        
        # 중앙값 계산
        median_rate = statistics.median(rate_values)
        
        # 이상치 필터링 (중앙값에서 2% 이상 차이나는 값 제외)
        filtered_rates = []
        for rate_data in rates:
            deviation_pct = abs((rate_data.rate - median_rate) / median_rate * 100)
            if deviation_pct <= self.max_deviation_pct:
                filtered_rates.append(rate_data)
            else:
                logger.warning(
                    f"Outlier detected from {rate_data.source}: "
                    f"{rate_data.rate} (deviation: {deviation_pct:.2f}%)"
                )
        
        if not filtered_rates:
            # 모든 값이 이상치인 경우 원본 사용
            filtered_rates = rates
        
        # 가중 평균 계산 (신뢰도 기반)
        total_confidence = sum(r.confidence for r in filtered_rates)
        weighted_rate = sum(r.rate * r.confidence for r in filtered_rates) / total_confidence
        
        # 표준편차 계산 (신뢰도 지표)
        if len(filtered_rates) > 1:
            stdev = statistics.stdev([r.rate for r in filtered_rates])
            confidence = max(0, 1 - (stdev / weighted_rate))  # 변동성이 클수록 신뢰도 낮음
        else:
            confidence = filtered_rates[0].confidence
        
        return ExchangeRateData(
            rate=weighted_rate,
            source=f"aggregate_{len(filtered_rates)}_sources",
            timestamp=datetime.now(),
            confidence=confidence
        )
    
    def _get_cached_rate(self) -> Optional[float]:
        """
        캐시된 환율 조회
        
        Returns:
            캐시된 환율 또는 None
        """
        if not self.cache:
            return None
        
        # 가장 최근 캐시 확인
        latest_key = max(self.cache.keys())
        cached_data = self.cache[latest_key]
        
        # 캐시 유효성 확인
        if datetime.now() - cached_data.timestamp < self.cache_duration:
            return cached_data.rate
        
        return None
    
    def _update_cache(self, rate_data: ExchangeRateData):
        """
        캐시 업데이트
        
        Args:
            rate_data: 환율 데이터
        """
        cache_key = rate_data.timestamp.strftime("%Y%m%d%H%M%S")
        self.cache[cache_key] = rate_data
        
        # 오래된 캐시 제거 (최대 10개 유지)
        if len(self.cache) > 10:
            oldest_keys = sorted(self.cache.keys())[:-10]
            for key in oldest_keys:
                del self.cache[key]
    
    def _add_to_history(self, rate_data: ExchangeRateData):
        """
        이력 추가
        
        Args:
            rate_data: 환율 데이터
        """
        self.fetch_history.append(rate_data)
        
        # 최대 개수 유지
        if len(self.fetch_history) > self.max_history:
            self.fetch_history = self.fetch_history[-self.max_history:]
    
    def get_statistics(self) -> Dict:
        """
        통계 조회
        
        Returns:
            환율 통계
        """
        if not self.fetch_history:
            return {
                'count': 0,
                'current_rate': None,
                'avg_rate': None,
                'min_rate': None,
                'max_rate': None,
                'stdev': None,
                'avg_confidence': None
            }
        
        rates = [h.rate for h in self.fetch_history]
        confidences = [h.confidence for h in self.fetch_history]
        
        return {
            'count': len(self.fetch_history),
            'current_rate': self.fetch_history[-1].rate,
            'avg_rate': statistics.mean(rates),
            'min_rate': min(rates),
            'max_rate': max(rates),
            'stdev': statistics.stdev(rates) if len(rates) > 1 else 0,
            'avg_confidence': statistics.mean(confidences)
        }
    
    async def continuous_update(self, interval_seconds: int = 300):
        """
        지속적인 환율 업데이트
        
        Args:
            interval_seconds: 업데이트 간격 (기본 5분)
        """
        logger.info(f"Starting continuous rate updates every {interval_seconds} seconds")
        
        while True:
            try:
                rate = await self.get_current_rate(force_refresh=True)
                stats = self.get_statistics()
                
                logger.info(
                    f"Exchange rate updated: {rate:.2f} KRW/USD "
                    f"(confidence: {stats['avg_confidence']:.2%})"
                )
                
                await asyncio.sleep(interval_seconds)
                
            except Exception as e:
                logger.error(f"Rate update error: {e}")
                await asyncio.sleep(interval_seconds)


# 싱글톤 인스턴스
_rate_fetcher_instance = None


def get_rate_fetcher() -> RealTimeExchangeRateFetcher:
    """
    환율 수집기 싱글톤 인스턴스 반환
    
    Returns:
        환율 수집기 인스턴스
    """
    global _rate_fetcher_instance
    if _rate_fetcher_instance is None:
        _rate_fetcher_instance = RealTimeExchangeRateFetcher()
    return _rate_fetcher_instance


async def get_current_exchange_rate() -> float:
    """
    현재 환율 조회 (간편 함수)
    
    Returns:
        현재 USD/KRW 환율
    """
    fetcher = get_rate_fetcher()
    return await fetcher.get_current_rate()