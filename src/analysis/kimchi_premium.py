"""
김치 프리미엄 계산 및 분석 모듈
실시간 김프 계산, 이상치 감지, 유동성 분석
"""

import asyncio
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import numpy as np
from collections import deque

from src.utils.logger import logger
from src.utils.metrics import metrics_collector
from src.utils.alerts import alert_manager, AlertLevel
from src.data.exchange_rate_manager import rate_manager
from src.data.websocket_manager import ws_manager


class PremiumSignal(Enum):
    """김프 시그널"""
    STRONG_BUY = "strong_buy"      # 김프 > 5%
    BUY = "buy"                     # 김프 > 4%
    NEUTRAL = "neutral"             # 2% < 김프 < 4%
    SELL = "sell"                   # 김프 < 2%
    STRONG_SELL = "strong_sell"     # 김프 < 1% or 역프
    ANOMALY = "anomaly"             # 이상치


@dataclass
class KimchiPremiumData:
    """김치 프리미엄 데이터"""
    timestamp: datetime
    upbit_price: float              # 업비트 BTC/KRW 가격
    binance_price: float            # 바이낸스 BTC/USDT 가격
    exchange_rate: float            # USD/KRW 환율
    premium_rate: float             # 김프율 (%)
    premium_krw: float              # 김프 원화 금액
    signal: PremiumSignal          # 시그널
    liquidity_score: float          # 유동성 점수 (0-100)
    spread_upbit: float            # 업비트 스프레드
    spread_binance: float          # 바이낸스 스프레드
    confidence: float              # 신뢰도 (0-1)


class KimchiPremiumCalculator:
    """김치 프리미엄 계산기"""
    
    def __init__(self):
        """초기화"""
        # 임계값 설정
        self.thresholds = {
            'strong_buy': 5.0,      # 강한 매수 신호
            'buy': 4.0,             # 매수 신호
            'sell': 2.0,            # 매도 신호
            'strong_sell': 1.0,     # 강한 매도 신호
            'anomaly': 10.0         # 이상치
        }
        
        # 히스토리 버퍼
        self.premium_history = deque(maxlen=1440)  # 24시간 (분당)
        self.ma_short = deque(maxlen=20)           # 20분 이동평균
        self.ma_long = deque(maxlen=60)            # 60분 이동평균
        
        # 현재 상태
        self.current_premium = None
        self.last_calculation = None
        
        # 통계
        self.daily_high = None
        self.daily_low = None
        self.daily_avg = None
        
        logger.info("Kimchi premium calculator initialized")
    
    async def calculate_premium(self,
                               upbit_price: float,
                               binance_price: float,
                               exchange_rate: Optional[float] = None) -> KimchiPremiumData:
        """
        김치 프리미엄 계산
        
        공식: (업비트 KRW / (바이낸스 USDT * 환율) - 1) * 100
        
        Args:
            upbit_price: 업비트 BTC/KRW 가격
            binance_price: 바이낸스 BTC/USDT 가격
            exchange_rate: USD/KRW 환율 (None이면 자동 조회)
            
        Returns:
            김치 프리미엄 데이터
        """
        try:
            # 환율 조회
            if exchange_rate is None:
                exchange_rate = await rate_manager.get_current_rate()
                if not exchange_rate:
                    raise ValueError("Failed to get exchange rate")
            
            # 김프 계산
            binance_krw = binance_price * exchange_rate
            premium_rate = ((upbit_price / binance_krw) - 1) * 100
            premium_krw = upbit_price - binance_krw
            
            # 시그널 결정
            signal = self._determine_signal(premium_rate)
            
            # 유동성 점수 (임시 - 실제로는 오더북 데이터 필요)
            liquidity_score = self._calculate_liquidity_score()
            
            # 스프레드 (임시 - 실제로는 오더북 데이터 필요)
            spread_upbit = 0.05  # 0.05%
            spread_binance = 0.02  # 0.02%
            
            # 신뢰도 계산
            confidence = self._calculate_confidence(
                premium_rate,
                liquidity_score,
                spread_upbit,
                spread_binance
            )
            
            # 데이터 생성
            data = KimchiPremiumData(
                timestamp=datetime.now(),
                upbit_price=upbit_price,
                binance_price=binance_price,
                exchange_rate=exchange_rate,
                premium_rate=premium_rate,
                premium_krw=premium_krw,
                signal=signal,
                liquidity_score=liquidity_score,
                spread_upbit=spread_upbit,
                spread_binance=spread_binance,
                confidence=confidence
            )
            
            # 히스토리 업데이트
            self._update_history(data)
            
            # 메트릭 업데이트
            metrics_collector.update_kimchi_premium(premium_rate)
            
            # 알림 체크
            await self._check_alerts(data)
            
            self.current_premium = data
            self.last_calculation = datetime.now()
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to calculate kimchi premium: {e}")
            raise
    
    def _determine_signal(self, premium_rate: float) -> PremiumSignal:
        """시그널 결정"""
        if abs(premium_rate) > self.thresholds['anomaly']:
            return PremiumSignal.ANOMALY
        elif premium_rate > self.thresholds['strong_buy']:
            return PremiumSignal.STRONG_BUY
        elif premium_rate > self.thresholds['buy']:
            return PremiumSignal.BUY
        elif premium_rate < self.thresholds['strong_sell']:
            return PremiumSignal.STRONG_SELL
        elif premium_rate < self.thresholds['sell']:
            return PremiumSignal.SELL
        else:
            return PremiumSignal.NEUTRAL
    
    def _calculate_liquidity_score(self) -> float:
        """유동성 점수 계산 (0-100)"""
        # TODO: 실제 오더북 데이터 기반 계산
        # 임시로 랜덤값 반환
        import random
        return random.uniform(60, 90)
    
    def _calculate_confidence(self,
                             premium_rate: float,
                             liquidity_score: float,
                             spread_upbit: float,
                             spread_binance: float) -> float:
        """
        신뢰도 계산 (0-1)
        
        고려 요소:
        - 유동성 점수
        - 스프레드
        - 김프 안정성
        """
        confidence = 1.0
        
        # 유동성 반영 (50% 가중치)
        confidence *= (liquidity_score / 100) * 0.5 + 0.5
        
        # 스프레드 반영 (낮을수록 좋음)
        total_spread = spread_upbit + spread_binance
        if total_spread > 0.1:  # 0.1% 이상
            confidence *= 0.9
        if total_spread > 0.2:  # 0.2% 이상
            confidence *= 0.8
        
        # 이상치는 신뢰도 낮춤
        if abs(premium_rate) > 10:
            confidence *= 0.5
        
        return min(max(confidence, 0), 1)
    
    def _update_history(self, data: KimchiPremiumData):
        """히스토리 업데이트"""
        self.premium_history.append(data)
        self.ma_short.append(data.premium_rate)
        self.ma_long.append(data.premium_rate)
        
        # 일일 통계 업데이트
        if self.premium_history:
            today_data = [d.premium_rate for d in self.premium_history 
                         if d.timestamp.date() == datetime.now().date()]
            
            if today_data:
                self.daily_high = max(today_data)
                self.daily_low = min(today_data)
                self.daily_avg = sum(today_data) / len(today_data)
    
    async def _check_alerts(self, data: KimchiPremiumData):
        """알림 체크"""
        # 강한 시그널 알림
        if data.signal == PremiumSignal.STRONG_BUY:
            await alert_manager.send_alert(
                message=f"Strong BUY signal: Kimchi premium {data.premium_rate:.2f}%",
                level=AlertLevel.INFO,
                title="Kimchi Premium Alert",
                details={
                    'premium': f"{data.premium_rate:.2f}%",
                    'upbit': f"{data.upbit_price:,.0f} KRW",
                    'binance': f"{data.binance_price:,.2f} USDT",
                    'confidence': f"{data.confidence:.2%}"
                }
            )
        
        # 이상치 알림
        elif data.signal == PremiumSignal.ANOMALY:
            await alert_manager.send_critical_alert(
                message=f"Anomaly detected: Kimchi premium {data.premium_rate:.2f}%",
                premium=data.premium_rate,
                upbit_price=data.upbit_price,
                binance_price=data.binance_price
            )
    
    def get_ma_cross_signal(self) -> Optional[str]:
        """이동평균 교차 시그널"""
        if len(self.ma_short) < 20 or len(self.ma_long) < 60:
            return None
        
        ma_short_val = sum(self.ma_short) / len(self.ma_short)
        ma_long_val = sum(self.ma_long) / len(self.ma_long)
        
        if ma_short_val > ma_long_val:
            return "golden_cross"  # 골든크로스
        elif ma_short_val < ma_long_val:
            return "death_cross"   # 데드크로스
        else:
            return "neutral"
    
    def get_volatility(self, hours: int = 24) -> Optional[float]:
        """변동성 계산"""
        if not self.premium_history:
            return None
        
        cutoff = datetime.now() - timedelta(hours=hours)
        recent_data = [d.premium_rate for d in self.premium_history
                      if d.timestamp > cutoff]
        
        if len(recent_data) < 2:
            return 0.0
        
        return float(np.std(recent_data))
    
    def get_statistics(self) -> Dict[str, Any]:
        """통계 정보"""
        return {
            'current_premium': self.current_premium.premium_rate if self.current_premium else None,
            'daily_high': self.daily_high,
            'daily_low': self.daily_low,
            'daily_avg': self.daily_avg,
            'volatility_24h': self.get_volatility(24),
            'ma_signal': self.get_ma_cross_signal(),
            'history_size': len(self.premium_history),
            'last_calculation': self.last_calculation
        }
    
    def detect_anomaly(self, premium_rate: float) -> bool:
        """
        이상치 감지
        
        Z-score 방식:
        - 3 표준편차 이상 벗어나면 이상치
        """
        if len(self.premium_history) < 100:
            # 데이터 부족시 임계값 방식
            return abs(premium_rate) > self.thresholds['anomaly']
        
        recent_rates = [d.premium_rate for d in self.premium_history]
        mean = np.mean(recent_rates)
        std = np.std(recent_rates)
        
        if std == 0:
            return False
        
        z_score = abs((premium_rate - mean) / std)
        return z_score > 3
    
    async def start_monitoring(self, interval: int = 10):
        """
        실시간 모니터링 시작
        
        Args:
            interval: 계산 간격 (초)
        """
        logger.info(f"Starting kimchi premium monitoring (interval: {interval}s)")
        
        # WebSocket 콜백 등록
        async def on_ticker_update(data):
            """티커 업데이트 콜백"""
            # 업비트와 바이낸스 가격이 모두 있을 때만 계산
            if hasattr(self, '_last_upbit_price') and hasattr(self, '_last_binance_price'):
                if data['exchange'] == 'upbit':
                    self._last_upbit_price = data['last']
                elif data['exchange'] == 'binance':
                    self._last_binance_price = data['last']
                
                # 김프 계산
                await self.calculate_premium(
                    self._last_upbit_price,
                    self._last_binance_price
                )
        
        ws_manager.register_callback('ticker', on_ticker_update)
        
        # 주기적 계산 (백업)
        while True:
            try:
                # TODO: 실제 가격 데이터 가져오기
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error in premium monitoring: {e}")
                await asyncio.sleep(interval)


# 전역 계산기
kimchi_calculator = KimchiPremiumCalculator()


if __name__ == "__main__":
    # 김프 계산 테스트
    async def test():
        print("Kimchi Premium Calculator Test")
        print("-" * 40)
        
        # 테스트 데이터
        upbit_price = 159_400_000    # 159.4M KRW
        binance_price = 115_000       # 115K USDT
        exchange_rate = 1386.14       # KRW/USD
        
        # 김프 계산
        data = await kimchi_calculator.calculate_premium(
            upbit_price,
            binance_price,
            exchange_rate
        )
        
        print(f"Upbit: {data.upbit_price:,.0f} KRW")
        print(f"Binance: {data.binance_price:,.2f} USDT")
        print(f"Exchange Rate: {data.exchange_rate:.2f} KRW/USD")
        print(f"\nKimchi Premium: {data.premium_rate:.2f}%")
        print(f"Premium Amount: {data.premium_krw:,.0f} KRW")
        print(f"Signal: {data.signal.value}")
        print(f"Confidence: {data.confidence:.2%}")
        
        # 통계
        stats = kimchi_calculator.get_statistics()
        print(f"\nStatistics: {stats}")
    
    asyncio.run(test())