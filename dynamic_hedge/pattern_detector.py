"""
Task 31: Triangle Pattern Detector
삼각수렴 패턴 탐지 및 돌파 예측
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from scipy.signal import argrelextrema

logger = logging.getLogger(__name__)


@dataclass
class ConvergenceMetrics:
    """수렴 메트릭"""
    volatility_ratio: float  # 현재/과거 변동성 비율
    convergence_rate: float  # 수렴 속도
    time_to_apex: float  # apex까지 예상 시간 (시간 단위)
    confidence: float  # 패턴 신뢰도 (0-1)


@dataclass
class BreakoutPrediction:
    """돌파 예측"""
    predicted_direction: str  # 'up' or 'down'
    probability: float  # 0-1
    expected_time: datetime
    expected_magnitude: float  # 예상 변동폭 (%)
    confidence: float  # 예측 신뢰도


class TrianglePatternDetector:
    """
    삼각수렴 패턴 탐지기
    - 가격 변동성 축소 감지
    - 수렴 각도 계산
    - 돌파 예상 시점 예측
    - False breakout 필터링
    """
    
    def __init__(self, min_pattern_length: int = 20, 
                 convergence_threshold: float = 0.7):
        """
        Args:
            min_pattern_length: 패턴 인식 최소 캔들 수
            convergence_threshold: 수렴 인정 임계값
        """
        self.min_pattern_length = min_pattern_length
        self.convergence_threshold = convergence_threshold
        self.detected_patterns: List[Dict] = []
        self.pattern_history: List[Dict] = []
        
    def detect_triangle(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        삼각수렴 패턴 감지
        
        Args:
            df: OHLCV 데이터프레임
            
        Returns:
            감지된 패턴 정보 또는 None
        """
        if len(df) < self.min_pattern_length:
            return None
            
        # 고점/저점 찾기
        highs = self._find_swing_highs(df)
        lows = self._find_swing_lows(df)
        
        if len(highs) < 2 or len(lows) < 2:
            return None
            
        # 추세선 계산
        upper_trend = self._calculate_trendline(highs, 'high')
        lower_trend = self._calculate_trendline(lows, 'low')
        
        if not upper_trend or not lower_trend:
            return None
            
        # 수렴 확인
        convergence = self._check_convergence(upper_trend, lower_trend, df)
        
        if not convergence:
            return None
            
        # 패턴 분류
        pattern_type = self._classify_pattern(upper_trend, lower_trend)
        
        # 메트릭 계산
        metrics = self._calculate_metrics(df, upper_trend, lower_trend)
        
        # 돌파 예측
        prediction = self._predict_breakout(pattern_type, metrics, df)
        
        pattern_info = {
            'type': pattern_type,
            'upper_trend': upper_trend,
            'lower_trend': lower_trend,
            'metrics': metrics,
            'prediction': prediction,
            'detected_at': datetime.now(),
            'start_time': df.index[0],
            'end_time': df.index[-1]
        }
        
        self.detected_patterns.append(pattern_info)
        
        return pattern_info
    
    def _find_swing_highs(self, df: pd.DataFrame, order: int = 5) -> pd.DataFrame:
        """
        스윙 하이 찾기
        
        Args:
            df: OHLCV 데이터
            order: 극값 판단 범위
            
        Returns:
            스윙 하이 데이터프레임
        """
        high_idx = argrelextrema(df['high'].values, np.greater, order=order)[0]
        
        if len(high_idx) == 0:
            return pd.DataFrame()
            
        return df.iloc[high_idx][['high', 'volume']]
    
    def _find_swing_lows(self, df: pd.DataFrame, order: int = 5) -> pd.DataFrame:
        """
        스윙 로우 찾기
        
        Args:
            df: OHLCV 데이터
            order: 극값 판단 범위
            
        Returns:
            스윙 로우 데이터프레임
        """
        low_idx = argrelextrema(df['low'].values, np.less, order=order)[0]
        
        if len(low_idx) == 0:
            return pd.DataFrame()
            
        return df.iloc[low_idx][['low', 'volume']]
    
    def _calculate_trendline(self, points: pd.DataFrame, 
                           price_col: str) -> Optional[Dict]:
        """
        추세선 계산
        
        Args:
            points: 포인트 데이터
            price_col: 가격 컬럼명
            
        Returns:
            추세선 정보
        """
        if len(points) < 2:
            return None
            
        # 시간을 숫자로 변환
        x = np.arange(len(points))
        y = points[price_col].values
        
        # 선형 회귀
        coeffs = np.polyfit(x, y, 1)
        slope = coeffs[0]
        intercept = coeffs[1]
        
        # 추세선 값 계산
        trend_values = np.polyval(coeffs, x)
        
        # R^2 계산
        ss_res = np.sum((y - trend_values) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_squared,
            'points': points,
            'start_time': points.index[0],
            'end_time': points.index[-1]
        }
    
    def _check_convergence(self, upper: Dict, lower: Dict, 
                         df: pd.DataFrame) -> bool:
        """
        수렴 여부 확인
        
        Args:
            upper: 상단 추세선
            lower: 하단 추세선
            df: 전체 데이터
            
        Returns:
            수렴 여부
        """
        # 기울기 차이로 수렴 확인
        slope_diff = upper['slope'] - lower['slope']
        
        # 상단은 하락, 하단은 상승 또는 평평해야 수렴
        is_converging = slope_diff < 0
        
        if not is_converging:
            return False
            
        # 현재 간격과 미래 간격 비교
        current_gap = df['high'].iloc[-1] - df['low'].iloc[-1]
        initial_gap = df['high'].iloc[0] - df['low'].iloc[0]
        
        gap_ratio = current_gap / initial_gap if initial_gap > 0 else 1
        
        return gap_ratio < self.convergence_threshold
    
    def _classify_pattern(self, upper: Dict, lower: Dict) -> str:
        """
        삼각형 패턴 분류
        
        Args:
            upper: 상단 추세선
            lower: 하단 추세선
            
        Returns:
            패턴 타입
        """
        upper_slope = upper['slope']
        lower_slope = lower['slope']
        
        # 기울기 임계값
        flat_threshold = 0.0001
        
        if abs(upper_slope) < flat_threshold and lower_slope > flat_threshold:
            return 'ascending'  # 상승 삼각형
        elif upper_slope < -flat_threshold and abs(lower_slope) < flat_threshold:
            return 'descending'  # 하락 삼각형
        elif upper_slope < -flat_threshold and lower_slope > flat_threshold:
            return 'symmetric'  # 대칭 삼각형
        else:
            return 'wedge'  # 쐐기형
    
    def _calculate_metrics(self, df: pd.DataFrame, 
                         upper: Dict, lower: Dict) -> ConvergenceMetrics:
        """
        수렴 메트릭 계산
        
        Args:
            df: OHLCV 데이터
            upper: 상단 추세선
            lower: 하단 추세선
            
        Returns:
            수렴 메트릭
        """
        # 변동성 계산
        recent_volatility = df['high'].tail(10).std() / df['close'].tail(10).mean()
        past_volatility = df['high'].head(10).std() / df['close'].head(10).mean()
        volatility_ratio = recent_volatility / past_volatility if past_volatility > 0 else 1
        
        # 수렴 속도
        slope_diff = abs(upper['slope'] - lower['slope'])
        convergence_rate = slope_diff * 100  # 정규화
        
        # Apex까지 시간 계산
        if upper['slope'] != lower['slope']:
            # 두 선이 만나는 지점
            x_intersect = (lower['intercept'] - upper['intercept']) / (upper['slope'] - lower['slope'])
            # 현재 시점부터 교점까지 캔들 수
            candles_to_apex = max(0, x_intersect - len(df))
            # 15분봉 기준 시간 변환
            time_to_apex = candles_to_apex * 0.25  # 시간 단위
        else:
            time_to_apex = float('inf')
            
        # 패턴 신뢰도
        confidence = (upper['r_squared'] + lower['r_squared']) / 2
        
        return ConvergenceMetrics(
            volatility_ratio=volatility_ratio,
            convergence_rate=convergence_rate,
            time_to_apex=time_to_apex,
            confidence=confidence
        )
    
    def _predict_breakout(self, pattern_type: str, 
                        metrics: ConvergenceMetrics,
                        df: pd.DataFrame) -> BreakoutPrediction:
        """
        돌파 예측
        
        Args:
            pattern_type: 패턴 타입
            metrics: 수렴 메트릭
            df: OHLCV 데이터
            
        Returns:
            돌파 예측
        """
        # 패턴별 기본 돌파 방향
        direction_probs = {
            'ascending': ('up', 0.65),
            'descending': ('down', 0.65),
            'symmetric': ('up', 0.5),  # 중립, 추가 분석 필요
            'wedge': ('down', 0.55)
        }
        
        direction, base_prob = direction_probs.get(pattern_type, ('up', 0.5))
        
        # 대칭 삼각형의 경우 추가 분석
        if pattern_type == 'symmetric':
            # 최근 추세로 방향 결정
            ma20 = df['close'].tail(20).mean()
            ma50 = df['close'].tail(50).mean()
            if ma20 > ma50:
                direction = 'up'
                base_prob = 0.55
            else:
                direction = 'down'
                base_prob = 0.55
        
        # 볼륨 분석으로 확률 조정
        recent_volume = df['volume'].tail(5).mean()
        avg_volume = df['volume'].mean()
        volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1
        
        # 볼륨이 감소하면 돌파 임박
        if volume_ratio < 0.7:
            probability = min(base_prob + 0.1, 0.9)
        else:
            probability = base_prob
            
        # 예상 돌파 시간
        expected_time = datetime.now() + timedelta(hours=metrics.time_to_apex)
        
        # 예상 변동폭 (과거 변동성 기반)
        historical_volatility = df['close'].pct_change().std()
        expected_magnitude = historical_volatility * 2 * 100  # 2 표준편차, 퍼센트
        
        # 예측 신뢰도
        prediction_confidence = metrics.confidence * 0.7 + (1 - metrics.volatility_ratio) * 0.3
        
        return BreakoutPrediction(
            predicted_direction=direction,
            probability=probability,
            expected_time=expected_time,
            expected_magnitude=expected_magnitude,
            confidence=prediction_confidence
        )
    
    def validate_breakout(self, price_data: pd.Series, 
                         pattern: Dict) -> Tuple[bool, str]:
        """
        돌파 유효성 검증 (False breakout 필터링)
        
        Args:
            price_data: 현재 가격 데이터
            pattern: 패턴 정보
            
        Returns:
            (유효 여부, 이유)
        """
        # 돌파 확인
        upper_price = self._get_trend_price(pattern['upper_trend'], len(price_data))
        lower_price = self._get_trend_price(pattern['lower_trend'], len(price_data))
        
        current_price = price_data.iloc[-1]
        prev_price = price_data.iloc[-2]
        
        # 돌파 방향 확인
        if current_price > upper_price and prev_price <= upper_price:
            breakout_direction = 'up'
        elif current_price < lower_price and prev_price >= lower_price:
            breakout_direction = 'down'
        else:
            return False, "No breakout detected"
            
        # 돌파 강도 확인 (2% 이상)
        if breakout_direction == 'up':
            breakout_strength = (current_price - upper_price) / upper_price
        else:
            breakout_strength = (lower_price - current_price) / lower_price
            
        if breakout_strength < 0.02:
            return False, f"Weak breakout: {breakout_strength:.2%}"
            
        # 볼륨 확인
        # 실제 구현에서는 볼륨 데이터도 함께 확인
        
        return True, f"Valid {breakout_direction} breakout"
    
    def _get_trend_price(self, trend: Dict, position: int) -> float:
        """
        특정 위치의 추세선 가격 계산
        
        Args:
            trend: 추세선 정보
            position: 위치 인덱스
            
        Returns:
            추세선 가격
        """
        return trend['slope'] * position + trend['intercept']
    
    def get_active_patterns(self) -> List[Dict]:
        """
        활성 패턴 목록 조회
        
        Returns:
            활성 패턴 리스트
        """
        # 24시간 이내 감지된 패턴
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        active = [
            p for p in self.detected_patterns
            if p['detected_at'] > cutoff_time
        ]
        
        return active
    
    def calculate_pattern_strength(self, pattern: Dict) -> float:
        """
        패턴 강도 계산
        
        Args:
            pattern: 패턴 정보
            
        Returns:
            패턴 강도 (0-1)
        """
        metrics = pattern['metrics']
        
        # 각 요소별 점수
        convergence_score = min(metrics.convergence_rate / 0.1, 1.0)
        volatility_score = 1 - metrics.volatility_ratio
        confidence_score = metrics.confidence
        
        # 가중 평균
        strength = (
            convergence_score * 0.3 +
            volatility_score * 0.3 +
            confidence_score * 0.4
        )
        
        return min(max(strength, 0), 1)