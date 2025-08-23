"""
단순 임계값 기반 트레이딩 전략
김치 프리미엄 임계값에 따른 진입/청산 로직
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from src.analysis.kimchi_premium import KimchiPremiumData, PremiumSignal
from src.utils.logger import logger


class SignalType(str, Enum):
    """트레이딩 시그널 타입"""

    ENTER_LONG = "enter_long"  # 업비트 매수 진입
    EXIT_LONG = "exit_long"  # 업비트 매수 청산
    ENTER_SHORT = "enter_short"  # 바이낸스 숏 진입
    EXIT_SHORT = "exit_short"  # 바이낸스 숏 청산
    HOLD = "hold"  # 대기
    CLOSE_ALL = "close_all"  # 전체 포지션 청산


class PositionSide(str, Enum):
    """포지션 방향"""

    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"


@dataclass
class Position:
    """포지션 정보"""

    symbol: str
    side: PositionSide
    size: float  # BTC 수량
    entry_price: float
    entry_time: datetime
    exchange: str  # 'upbit' or 'binance'
    
    def get_pnl(self, current_price: float) -> float:
        """현재 수익률 계산"""
        if self.side == PositionSide.LONG:
            return (current_price - self.entry_price) * self.size
        elif self.side == PositionSide.SHORT:
            return (self.entry_price - current_price) * self.size
        return 0.0


class TradingSignal(BaseModel):
    """트레이딩 시그널"""

    timestamp: datetime
    signal_type: SignalType
    premium_rate: float = Field(description="현재 김치 프리미엄률")
    confidence: float = Field(ge=0, le=1, description="신뢰도")
    position_size: float = Field(gt=0, description="포지션 크기 (BTC)")
    reason: str = Field(description="시그널 생성 이유")
    risk_score: float = Field(ge=0, le=1, description="리스크 점수")


class SimpleThresholdStrategy:
    """단순 임계값 기반 전략"""

    def __init__(
        self,
        enter_threshold: float = 4.0,  # 진입 김프 (%)
        exit_threshold: float = 2.0,  # 청산 김프 (%)
        reverse_exit_threshold: float = -1.0,  # 역프리미엄 청산
        capital: float = 20_000_000,  # 자본금 (KRW)
        max_position_pct: float = 0.01,  # 최대 포지션 크기 (자본금의 1%)
        min_liquidity_score: float = 70.0,  # 최소 유동성 점수
    ):
        """
        초기화

        Args:
            enter_threshold: 진입 김프 임계값
            exit_threshold: 청산 김프 임계값
            reverse_exit_threshold: 역프리미엄 청산 임계값
            capital: 자본금
            max_position_pct: 최대 포지션 비율
            min_liquidity_score: 최소 유동성 점수
        """
        self.enter_threshold = enter_threshold
        self.exit_threshold = exit_threshold
        self.reverse_exit_threshold = reverse_exit_threshold
        self.capital = capital
        self.max_position_pct = max_position_pct
        self.min_liquidity_score = min_liquidity_score

        # 포지션 관리
        self.positions: Dict[str, Position] = {}
        self.signal_history: List[TradingSignal] = []
        self.max_positions = 1  # 동시 최대 포지션 수

        logger.info(
            f"Simple threshold strategy initialized: "
            f"enter={enter_threshold}%, exit={exit_threshold}%"
        )

    def calculate_position_size(
        self, premium_data: KimchiPremiumData, btc_price_krw: float
    ) -> float:
        """
        포지션 크기 계산

        Args:
            premium_data: 김프 데이터
            btc_price_krw: BTC 원화 가격

        Returns:
            포지션 크기 (BTC)
        """
        # 자본금의 1% 사용
        position_value_krw = self.capital * self.max_position_pct

        # 리스크 조정 (신뢰도 기반)
        risk_adjusted_value = position_value_krw * premium_data.confidence

        # BTC 수량 계산
        btc_amount = risk_adjusted_value / btc_price_krw

        # 최소 거래 단위 조정 (0.001 BTC)
        btc_amount = round(btc_amount, 3)

        return max(btc_amount, 0.001)  # 최소 0.001 BTC

    def generate_signal(self, premium_data: KimchiPremiumData) -> Optional[TradingSignal]:
        """
        트레이딩 시그널 생성

        Args:
            premium_data: 김치 프리미엄 데이터

        Returns:
            트레이딩 시그널 또는 None
        """
        premium_rate = premium_data.premium_rate
        has_position = len(self.positions) > 0

        # 현재 포지션이 없는 경우
        if not has_position:
            # 진입 조건 체크
            if self._check_entry_conditions(premium_data):
                # 포지션 크기 계산
                position_size = self.calculate_position_size(
                    premium_data, premium_data.upbit_price
                )

                signal = TradingSignal(
                    timestamp=premium_data.timestamp,
                    signal_type=SignalType.ENTER_LONG,
                    premium_rate=premium_rate,
                    confidence=premium_data.confidence,
                    position_size=position_size,
                    reason=f"김프 {premium_rate:.2f}% > {self.enter_threshold}% (진입)",
                    risk_score=self._calculate_risk_score(premium_data),
                )

                self._record_signal(signal)
                return signal

        # 포지션이 있는 경우
        else:
            # 청산 조건 체크
            exit_reason = self._check_exit_conditions(premium_data)
            if exit_reason:
                signal = TradingSignal(
                    timestamp=premium_data.timestamp,
                    signal_type=SignalType.EXIT_LONG,
                    premium_rate=premium_rate,
                    confidence=premium_data.confidence,
                    position_size=list(self.positions.values())[0].size,
                    reason=exit_reason,
                    risk_score=self._calculate_risk_score(premium_data),
                )

                self._record_signal(signal)
                return signal

        # 홀드 시그널
        return None

    def _check_entry_conditions(self, premium_data: KimchiPremiumData) -> bool:
        """진입 조건 확인"""
        # 1. 김프가 임계값 이상
        if premium_data.premium_rate < self.enter_threshold:
            return False

        # 2. 유동성 충분
        if premium_data.liquidity_score < self.min_liquidity_score:
            logger.debug(
                f"Insufficient liquidity: {premium_data.liquidity_score:.1f} < {self.min_liquidity_score}"
            )
            return False

        # 3. 신뢰도 충분 (60% 이상)
        if premium_data.confidence < 0.6:
            logger.debug(f"Low confidence: {premium_data.confidence:.2%}")
            return False

        # 4. 이상치가 아님
        if premium_data.signal == PremiumSignal.ANOMALY:
            logger.debug("Anomaly detected, skipping entry")
            return False

        # 5. 최대 포지션 수 체크
        if len(self.positions) >= self.max_positions:
            logger.debug("Maximum positions reached")
            return False

        return True

    def _check_exit_conditions(self, premium_data: KimchiPremiumData) -> Optional[str]:
        """
        청산 조건 확인

        Returns:
            청산 이유 또는 None
        """
        premium_rate = premium_data.premium_rate

        # 1. 김프가 청산 임계값 이하
        if premium_rate < self.exit_threshold:
            return f"김프 {premium_rate:.2f}% < {self.exit_threshold}% (목표 청산)"

        # 2. 역프리미엄 발생
        if premium_rate < self.reverse_exit_threshold:
            return f"역프리미엄 {premium_rate:.2f}% (긴급 청산)"

        # 3. 이상치 감지
        if premium_data.signal == PremiumSignal.ANOMALY:
            return "이상치 감지 (리스크 청산)"

        # 4. 낮은 신뢰도
        if premium_data.confidence < 0.3:
            return f"신뢰도 하락 {premium_data.confidence:.2%} (안전 청산)"

        return None

    def _calculate_risk_score(self, premium_data: KimchiPremiumData) -> float:
        """리스크 점수 계산 (0~1, 낮을수록 안전)"""
        risk_score = 0.0

        # 1. 스프레드 리스크 (높을수록 위험)
        spread_risk = (premium_data.spread_upbit + premium_data.spread_binance) * 10
        risk_score += min(spread_risk, 0.3)

        # 2. 유동성 리스크 (낮을수록 위험)
        liquidity_risk = max(0, (100 - premium_data.liquidity_score) / 100) * 0.3
        risk_score += liquidity_risk

        # 3. 신뢰도 리스크 (낮을수록 위험)
        confidence_risk = (1 - premium_data.confidence) * 0.2
        risk_score += confidence_risk

        # 4. 김프 극단값 리스크
        if abs(premium_data.premium_rate) > 10:
            risk_score += 0.2

        return min(risk_score, 1.0)

    def _record_signal(self, signal: TradingSignal):
        """시그널 기록"""
        self.signal_history.append(signal)

        # 최근 100개만 유지
        if len(self.signal_history) > 100:
            self.signal_history = self.signal_history[-100:]

    def open_position(
        self,
        symbol: str,
        side: PositionSide,
        size: float,
        entry_price: float,
        exchange: str,
    ) -> Position:
        """
        포지션 오픈 (시뮬레이션)

        Args:
            symbol: 심볼
            side: 방향
            size: 크기
            entry_price: 진입가
            exchange: 거래소

        Returns:
            포지션 객체
        """
        position = Position(
            symbol=symbol,
            side=side,
            size=size,
            entry_price=entry_price,
            entry_time=datetime.now(),
            exchange=exchange,
        )

        position_key = f"{exchange}_{symbol}"
        self.positions[position_key] = position

        logger.info(
            f"Position opened: {exchange} {symbol} {side.value} "
            f"{size:.4f} BTC @ {entry_price:,.0f}"
        )

        return position

    def close_position(self, position_key: str, exit_price: float) -> Optional[float]:
        """
        포지션 청산 (시뮬레이션)

        Args:
            position_key: 포지션 키
            exit_price: 청산가

        Returns:
            실현 손익
        """
        if position_key not in self.positions:
            logger.warning(f"Position {position_key} not found")
            return None

        position = self.positions[position_key]
        pnl = position.get_pnl(exit_price)

        logger.info(
            f"Position closed: {position.exchange} {position.symbol} "
            f"{position.side.value} {position.size:.4f} BTC "
            f"PnL: {pnl:,.0f} KRW"
        )

        del self.positions[position_key]
        return pnl

    def get_statistics(self) -> Dict:
        """전략 통계"""
        total_signals = len(self.signal_history)
        
        if total_signals == 0:
            return {
                "total_signals": 0,
                "enter_signals": 0,
                "exit_signals": 0,
                "avg_confidence": 0,
                "avg_risk_score": 0,
                "current_positions": len(self.positions),
            }

        enter_signals = sum(
            1 for s in self.signal_history 
            if s.signal_type == SignalType.ENTER_LONG
        )
        exit_signals = sum(
            1 for s in self.signal_history 
            if s.signal_type == SignalType.EXIT_LONG
        )

        return {
            "total_signals": total_signals,
            "enter_signals": enter_signals,
            "exit_signals": exit_signals,
            "avg_confidence": sum(s.confidence for s in self.signal_history) / total_signals,
            "avg_risk_score": sum(s.risk_score for s in self.signal_history) / total_signals,
            "current_positions": len(self.positions),
            "last_signal": self.signal_history[-1].signal_type.value if self.signal_history else None,
        }


# 전략 인스턴스
simple_strategy = SimpleThresholdStrategy()


if __name__ == "__main__":
    # 테스트
    from src.analysis.kimchi_premium import KimchiPremiumData

    # 테스트 데이터
    test_data = KimchiPremiumData(
        timestamp=datetime.now(),
        upbit_price=160_000_000,
        binance_price=115_000,
        exchange_rate=1386.14,
        premium_rate=4.5,  # 4.5% 김프
        premium_krw=4_500_000,
        signal=PremiumSignal.BUY,
        liquidity_score=85.0,
        spread_upbit=0.05,
        spread_binance=0.02,
        confidence=0.85,
    )

    # 시그널 생성
    signal = simple_strategy.generate_signal(test_data)
    
    if signal:
        print(f"Signal: {signal.signal_type.value}")
        print(f"Reason: {signal.reason}")
        print(f"Position size: {signal.position_size:.4f} BTC")
        print(f"Risk score: {signal.risk_score:.2%}")