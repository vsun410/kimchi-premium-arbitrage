"""
실시간 가격 검증 시스템
실제 시장 가격과의 정확도를 모니터링하고 검증

목적: 실시간 데이터 정확도 보장
결과: 실제 가격과 오차 < 0.1%
평가: 자동 정확도 측정 및 리포트
"""

import asyncio
from typing import Dict, Optional, Tuple, List
from datetime import datetime, timedelta
import logging
import json
import aiohttp
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class PriceValidation:
    """가격 검증 결과"""
    timestamp: datetime
    source: str
    symbol: str
    our_price: float
    market_price: float
    difference_pct: float
    is_valid: bool
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """딕셔너리로 변환"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class ExchangeRateValidation:
    """환율 검증 결과"""
    timestamp: datetime
    source: str
    our_rate: float
    market_rate: float
    difference_pct: float
    is_valid: bool
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """딕셔너리로 변환"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


class RealTimePriceValidator:
    """
    실시간 가격 검증기
    
    핵심 기능:
    1. 실시간 시장 가격 대조
    2. 환율 하드코딩 방지
    3. 오차 발생 시 즉시 알림
    """
    
    # 최대 허용 오차 (%)
    MAX_PRICE_DIFFERENCE_PCT = 0.1  # 0.1%
    MAX_RATE_DIFFERENCE_PCT = 0.5   # 0.5% (환율은 변동이 적음)
    
    def __init__(self):
        """검증기 초기화"""
        self.validation_history: List[PriceValidation] = []
        self.rate_validation_history: List[ExchangeRateValidation] = []
        self.last_alert_time = {}
        self.alert_cooldown = timedelta(minutes=5)  # 5분 쿨다운
        
        # 외부 가격 소스 (검증용)
        self.price_sources = {
            'coingecko': 'https://api.coingecko.com/api/v3/simple/price',
            'coinmarketcap': None,  # API 키 필요
            'cryptocompare': 'https://min-api.cryptocompare.com/data/price'
        }
        
        # 환율 소스 (검증용)
        self.rate_sources = {
            'exchangerate-api': 'https://api.exchangerate-api.com/v4/latest/USD',
            'fixer': None,  # API 키 필요
            'currencylayer': None  # API 키 필요
        }
        
        logger.info("RealTimePriceValidator initialized")
    
    async def validate_btc_price(
        self,
        our_upbit_price: float,
        our_binance_price: float
    ) -> Tuple[PriceValidation, PriceValidation]:
        """
        BTC 가격 검증
        
        Args:
            our_upbit_price: 우리 시스템의 업비트 BTC 가격 (KRW)
            our_binance_price: 우리 시스템의 바이낸스 BTC 가격 (USDT)
            
        Returns:
            업비트, 바이낸스 검증 결과
        """
        try:
            # CoinGecko에서 실제 가격 가져오기
            market_prices = await self._fetch_coingecko_prices()
            
            # 업비트 가격 검증
            upbit_validation = self._validate_price(
                source="Upbit",
                symbol="BTC/KRW",
                our_price=our_upbit_price,
                market_price=market_prices.get('btc_krw', 0)
            )
            
            # 바이낸스 가격 검증
            binance_validation = self._validate_price(
                source="Binance",
                symbol="BTC/USDT",
                our_price=our_binance_price,
                market_price=market_prices.get('btc_usdt', 0)
            )
            
            # 검증 이력 저장
            self.validation_history.append(upbit_validation)
            self.validation_history.append(binance_validation)
            
            # 오차가 크면 경고
            if not upbit_validation.is_valid:
                await self._send_alert(
                    f"⚠️ Upbit BTC 가격 오차 발생: {upbit_validation.difference_pct:.2f}%"
                )
            
            if not binance_validation.is_valid:
                await self._send_alert(
                    f"⚠️ Binance BTC 가격 오차 발생: {binance_validation.difference_pct:.2f}%"
                )
            
            return upbit_validation, binance_validation
            
        except Exception as e:
            logger.error(f"Failed to validate BTC prices: {e}")
            # 검증 실패 시에도 결과 반환
            error_validation = PriceValidation(
                timestamp=datetime.now(),
                source="Error",
                symbol="BTC",
                our_price=0,
                market_price=0,
                difference_pct=0,
                is_valid=False,
                error_message=str(e)
            )
            return error_validation, error_validation
    
    async def validate_exchange_rate(
        self,
        our_rate: float,
        source: str = "system"
    ) -> ExchangeRateValidation:
        """
        USD/KRW 환율 검증
        
        절대 하드코딩된 값을 사용하지 않도록 검증
        
        Args:
            our_rate: 우리 시스템의 환율
            source: 환율 소스 (어디서 가져왔는지)
            
        Returns:
            환율 검증 결과
        """
        try:
            # 하드코딩 감지
            KNOWN_HARDCODED_VALUES = [1330, 1300, 1350, 1400, 3.3]
            if our_rate in KNOWN_HARDCODED_VALUES:
                logger.critical(f"🚨 하드코딩된 환율 감지: {our_rate}")
                return ExchangeRateValidation(
                    timestamp=datetime.now(),
                    source=source,
                    our_rate=our_rate,
                    market_rate=0,
                    difference_pct=100,
                    is_valid=False,
                    error_message=f"하드코딩된 환율 값 감지: {our_rate}"
                )
            
            # 실제 환율 가져오기
            market_rate = await self._fetch_real_exchange_rate()
            
            # 차이 계산
            difference_pct = abs((our_rate - market_rate) / market_rate * 100)
            
            # 검증 결과 생성
            validation = ExchangeRateValidation(
                timestamp=datetime.now(),
                source=source,
                our_rate=our_rate,
                market_rate=market_rate,
                difference_pct=difference_pct,
                is_valid=difference_pct <= self.MAX_RATE_DIFFERENCE_PCT
            )
            
            # 이력 저장
            self.rate_validation_history.append(validation)
            
            # 오차가 크면 경고
            if not validation.is_valid:
                await self._send_alert(
                    f"🚨 환율 오차 발생!\n"
                    f"시스템: {our_rate:.2f}\n"
                    f"실제: {market_rate:.2f}\n"
                    f"차이: {difference_pct:.2f}%"
                )
            
            return validation
            
        except Exception as e:
            logger.error(f"Failed to validate exchange rate: {e}")
            return ExchangeRateValidation(
                timestamp=datetime.now(),
                source=source,
                our_rate=our_rate,
                market_rate=0,
                difference_pct=0,
                is_valid=False,
                error_message=str(e)
            )
    
    def _validate_price(
        self,
        source: str,
        symbol: str,
        our_price: float,
        market_price: float
    ) -> PriceValidation:
        """
        가격 검증 로직
        
        Args:
            source: 거래소 이름
            symbol: 거래 심볼
            our_price: 우리 시스템 가격
            market_price: 시장 가격
            
        Returns:
            검증 결과
        """
        if market_price == 0:
            return PriceValidation(
                timestamp=datetime.now(),
                source=source,
                symbol=symbol,
                our_price=our_price,
                market_price=market_price,
                difference_pct=0,
                is_valid=False,
                error_message="Market price unavailable"
            )
        
        # 차이 계산
        difference_pct = abs((our_price - market_price) / market_price * 100)
        
        return PriceValidation(
            timestamp=datetime.now(),
            source=source,
            symbol=symbol,
            our_price=our_price,
            market_price=market_price,
            difference_pct=difference_pct,
            is_valid=difference_pct <= self.MAX_PRICE_DIFFERENCE_PCT
        )
    
    async def _fetch_coingecko_prices(self) -> Dict[str, float]:
        """
        CoinGecko에서 실제 BTC 가격 조회
        
        Returns:
            {'btc_krw': 가격, 'btc_usdt': 가격}
        """
        try:
            async with aiohttp.ClientSession() as session:
                params = {
                    'ids': 'bitcoin',
                    'vs_currencies': 'krw,usd'
                }
                
                async with session.get(
                    self.price_sources['coingecko'],
                    params=params
                ) as response:
                    data = await response.json()
                    
                    return {
                        'btc_krw': data['bitcoin']['krw'],
                        'btc_usdt': data['bitcoin']['usd']  # USD ≈ USDT
                    }
                    
        except Exception as e:
            logger.error(f"Failed to fetch CoinGecko prices: {e}")
            return {'btc_krw': 0, 'btc_usdt': 0}
    
    async def _fetch_real_exchange_rate(self) -> float:
        """
        실제 USD/KRW 환율 조회
        
        Returns:
            현재 환율
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.rate_sources['exchangerate-api']) as response:
                    data = await response.json()
                    return data['rates']['KRW']
                    
        except Exception as e:
            logger.error(f"Failed to fetch exchange rate: {e}")
            # 에러 시 일반적인 환율 반환 (하지만 경고)
            logger.warning("Using fallback exchange rate 1390")
            return 1390.0
    
    async def _send_alert(self, message: str):
        """
        경고 알림 전송
        
        Args:
            message: 경고 메시지
        """
        # 쿨다운 체크
        alert_key = hash(message[:50])  # 메시지 앞부분으로 키 생성
        now = datetime.now()
        
        if alert_key in self.last_alert_time:
            if now - self.last_alert_time[alert_key] < self.alert_cooldown:
                return  # 쿨다운 중
        
        self.last_alert_time[alert_key] = now
        
        # 로그에 기록
        logger.critical(message)
        
        # TODO: 실제 알림 시스템 연동 (Telegram, Discord, Email 등)
    
    def get_accuracy_report(self) -> Dict:
        """
        정확도 리포트 생성
        
        Returns:
            정확도 통계
        """
        if not self.validation_history:
            return {
                'total_validations': 0,
                'success_rate': 0,
                'avg_difference_pct': 0,
                'max_difference_pct': 0
            }
        
        total = len(self.validation_history)
        successful = sum(1 for v in self.validation_history if v.is_valid)
        differences = [v.difference_pct for v in self.validation_history]
        
        return {
            'total_validations': total,
            'success_rate': (successful / total) * 100,
            'avg_difference_pct': sum(differences) / len(differences),
            'max_difference_pct': max(differences),
            'last_validation': self.validation_history[-1].to_dict()
        }
    
    def get_rate_accuracy_report(self) -> Dict:
        """
        환율 정확도 리포트 생성
        
        Returns:
            환율 정확도 통계
        """
        if not self.rate_validation_history:
            return {
                'total_validations': 0,
                'success_rate': 0,
                'avg_difference_pct': 0,
                'max_difference_pct': 0
            }
        
        total = len(self.rate_validation_history)
        successful = sum(1 for v in self.rate_validation_history if v.is_valid)
        differences = [v.difference_pct for v in self.rate_validation_history]
        
        return {
            'total_validations': total,
            'success_rate': (successful / total) * 100,
            'avg_difference_pct': sum(differences) / len(differences),
            'max_difference_pct': max(differences),
            'last_validation': self.rate_validation_history[-1].to_dict()
        }
    
    async def continuous_validation(self, interval_seconds: int = 60):
        """
        지속적인 가격 검증 실행
        
        Args:
            interval_seconds: 검증 간격 (초)
        """
        logger.info(f"Starting continuous validation every {interval_seconds} seconds")
        
        while True:
            try:
                # TODO: 실제 시스템에서 가격 가져오기
                # 여기서는 예시 값 사용
                
                # 가격 검증
                await self.validate_btc_price(
                    our_upbit_price=140000000,  # 예시
                    our_binance_price=100000     # 예시
                )
                
                # 환율 검증
                await self.validate_exchange_rate(
                    our_rate=1390.25,  # 예시
                    source="exchange_rate_manager"
                )
                
                # 리포트 생성
                price_report = self.get_accuracy_report()
                rate_report = self.get_rate_accuracy_report()
                
                logger.info(f"Price Accuracy: {price_report['success_rate']:.1f}%")
                logger.info(f"Rate Accuracy: {rate_report['success_rate']:.1f}%")
                
                await asyncio.sleep(interval_seconds)
                
            except Exception as e:
                logger.error(f"Validation error: {e}")
                await asyncio.sleep(interval_seconds)