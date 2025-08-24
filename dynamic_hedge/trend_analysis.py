"""
Task 29: Trend Analysis Engine
추세선 자동 그리기, 삼각수렴 패턴 인식, 돌파 강도 측정
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from scipy import stats
from scipy.signal import find_peaks

logger = logging.getLogger(__name__)


@dataclass
class TrendLine:
    """추세선 데이터 클래스"""
    start_time: datetime
    end_time: datetime
    start_price: float
    end_price: float
    slope: float
    intercept: float
    strength: float  # 0-1, 추세선의 신뢰도
    touches: int  # 추세선에 닿은 횟수
    line_type: str  # 'support' or 'resistance'


@dataclass
class BreakoutSignal:
    """돌파 신호 데이터 클래스"""
    timestamp: datetime
    price: float
    volume: float
    direction: str  # 'up' or 'down'
    strength: float  # 0-1, 돌파 강도
    volume_confirmation: bool
    pattern_type: str  # 'trendline', 'triangle', 'channel'


@dataclass
class TrianglePattern:
    """삼각수렴 패턴 데이터 클래스"""
    pattern_type: str  # 'ascending', 'descending', 'symmetric'
    apex_time: datetime  # 수렴 예상 시점
    upper_line: TrendLine
    lower_line: TrendLine
    convergence_angle: float
    volatility_compression: float  # 변동성 축소 정도


class TrendAnalysisEngine:
    """
    추세선 분석 엔진
    - 자동 추세선 그리기
    - 삼각수렴 패턴 인식
    - 돌파 강도 측정
    - Support/Resistance 레벨 추적
    """
    
    def __init__(self, window_size: int = 100, min_touches: int = 3):
        """
        Args:
            window_size: 분석 윈도우 크기 (캔들 개수)
            min_touches: 유효한 추세선으로 인정하는 최소 터치 횟수
        """
        self.window_size = window_size
        self.min_touches = min_touches
        self.support_levels: List[float] = []
        self.resistance_levels: List[float] = []
        self.active_patterns: List[TrianglePattern] = []
        
    def analyze(self, df: pd.DataFrame) -> Dict:
        """
        전체 추세 분석 수행
        
        Args:
            df: OHLCV 데이터프레임
            
        Returns:
            분석 결과 딕셔너리
        """
        if len(df) < self.window_size:
            return {
                'trend_lines': [],
                'support_resistance': {},
                'triangle_patterns': [],
                'breakout_signals': []
            }
            
        # 추세선 찾기
        trend_lines = self._find_trend_lines(df)
        
        # Support/Resistance 레벨 찾기
        sr_levels = self._find_support_resistance(df)
        
        # 삼각수렴 패턴 찾기
        triangle_patterns = self._detect_triangle_patterns(trend_lines, df)
        
        # 돌파 신호 확인
        breakout_signals = self._check_breakouts(df, trend_lines, sr_levels)
        
        return {
            'trend_lines': trend_lines,
            'support_resistance': sr_levels,
            'triangle_patterns': triangle_patterns,
            'breakout_signals': breakout_signals
        }
    
    def _find_trend_lines(self, df: pd.DataFrame) -> List[TrendLine]:
        """
        자동으로 추세선 찾기
        
        Args:
            df: OHLCV 데이터프레임
            
        Returns:
            추세선 리스트
        """
        trend_lines = []
        
        # 고점 기반 저항선 찾기
        highs = df['high'].values
        high_peaks, _ = find_peaks(highs, distance=5)
        
        if len(high_peaks) >= 2:
            resistance_line = self._fit_trend_line(
                df.iloc[high_peaks], 
                'high', 
                'resistance'
            )
            if resistance_line:
                trend_lines.append(resistance_line)
        
        # 저점 기반 지지선 찾기
        lows = -df['low'].values
        low_peaks, _ = find_peaks(lows, distance=5)
        
        if len(low_peaks) >= 2:
            support_line = self._fit_trend_line(
                df.iloc[low_peaks], 
                'low', 
                'support'
            )
            if support_line:
                trend_lines.append(support_line)
                
        return trend_lines
    
    def _fit_trend_line(self, points_df: pd.DataFrame, 
                        price_col: str, line_type: str) -> Optional[TrendLine]:
        """
        포인트들에 추세선 피팅
        
        Args:
            points_df: 추세선 포인트 데이터프레임
            price_col: 가격 컬럼명 ('high' or 'low')
            line_type: 'support' or 'resistance'
            
        Returns:
            추세선 객체 또는 None
        """
        if len(points_df) < self.min_touches:
            return None
            
        # 시간을 숫자로 변환 (선형 회귀용)
        x = np.arange(len(points_df))
        y = points_df[price_col].values
        
        # 선형 회귀
        slope, intercept, r_value, _, _ = stats.linregress(x, y)
        
        # R^2 값이 너무 낮으면 추세선으로 인정하지 않음
        if abs(r_value) < 0.7:
            return None
            
        return TrendLine(
            start_time=points_df.index[0],
            end_time=points_df.index[-1],
            start_price=y[0],
            end_price=y[-1],
            slope=slope,
            intercept=intercept,
            strength=abs(r_value),
            touches=len(points_df),
            line_type=line_type
        )
    
    def _find_support_resistance(self, df: pd.DataFrame) -> Dict[str, List[float]]:
        """
        Support/Resistance 레벨 찾기
        
        Args:
            df: OHLCV 데이터프레임
            
        Returns:
            {'support': [...], 'resistance': [...]}
        """
        # 가격 빈도 분석으로 주요 레벨 찾기
        price_levels = []
        
        # 고점/저점 클러스터링
        highs = df['high'].values
        lows = df['low'].values
        
        # 히스토그램으로 빈도 높은 가격대 찾기
        hist_high, bins_high = np.histogram(highs, bins=20)
        hist_low, bins_low = np.histogram(lows, bins=20)
        
        # 빈도 높은 레벨 추출
        resistance_levels = []
        support_levels = []
        
        # 상위 20% 빈도 레벨을 주요 레벨로 선정
        threshold_high = np.percentile(hist_high, 80)
        threshold_low = np.percentile(hist_low, 80)
        
        for i, count in enumerate(hist_high):
            if count >= threshold_high:
                resistance_levels.append((bins_high[i] + bins_high[i+1]) / 2)
                
        for i, count in enumerate(hist_low):
            if count >= threshold_low:
                support_levels.append((bins_low[i] + bins_low[i+1]) / 2)
        
        return {
            'support': sorted(support_levels),
            'resistance': sorted(resistance_levels)
        }
    
    def _detect_triangle_patterns(self, trend_lines: List[TrendLine], 
                                 df: pd.DataFrame) -> List[TrianglePattern]:
        """
        삼각수렴 패턴 감지
        
        Args:
            trend_lines: 추세선 리스트
            df: OHLCV 데이터프레임
            
        Returns:
            삼각패턴 리스트
        """
        patterns = []
        
        # 상승/하락 추세선 쌍 찾기
        upper_lines = [l for l in trend_lines if l.line_type == 'resistance']
        lower_lines = [l for l in trend_lines if l.line_type == 'support']
        
        for upper in upper_lines:
            for lower in lower_lines:
                # 시간 겹침 확인
                overlap_start = max(upper.start_time, lower.start_time)
                overlap_end = min(upper.end_time, lower.end_time)
                
                if overlap_start >= overlap_end:
                    continue
                    
                # 수렴 확인
                pattern = self._classify_triangle(upper, lower, df)
                if pattern:
                    patterns.append(pattern)
                    
        return patterns
    
    def _classify_triangle(self, upper: TrendLine, lower: TrendLine, 
                          df: pd.DataFrame) -> Optional[TrianglePattern]:
        """
        삼각형 패턴 분류
        
        Args:
            upper: 상단 추세선
            lower: 하단 추세선
            df: OHLCV 데이터프레임
            
        Returns:
            삼각패턴 객체 또는 None
        """
        # 기울기로 패턴 분류
        upper_slope = upper.slope
        lower_slope = lower.slope
        
        # 수렴 각도 계산
        convergence_angle = abs(upper_slope - lower_slope)
        
        # 수렴하지 않으면 None
        if upper_slope >= lower_slope:
            return None
            
        # 패턴 타입 결정
        if upper_slope > 0 and lower_slope < 0:
            pattern_type = 'symmetric'
        elif upper_slope < 0 and abs(lower_slope) < 0.01:
            pattern_type = 'descending'
        elif abs(upper_slope) < 0.01 and lower_slope > 0:
            pattern_type = 'ascending'
        else:
            return None
            
        # 수렴 시점 계산
        # y1 = upper_slope * x + upper_intercept
        # y2 = lower_slope * x + lower_intercept
        # 교점: upper_slope * x + upper_intercept = lower_slope * x + lower_intercept
        if upper_slope != lower_slope:
            x_apex = (lower.intercept - upper.intercept) / (upper_slope - lower_slope)
            # 현재 시점으로부터 x_apex 캔들 후
            apex_time = df.index[-1] + timedelta(minutes=x_apex * 15)  # 15분봉 기준
        else:
            apex_time = df.index[-1] + timedelta(hours=24)  # 기본값
            
        # 변동성 압축 정도 계산
        recent_volatility = df['high'].tail(20).std() / df['close'].tail(20).mean()
        overall_volatility = df['high'].std() / df['close'].mean()
        volatility_compression = 1 - (recent_volatility / overall_volatility)
        
        return TrianglePattern(
            pattern_type=pattern_type,
            apex_time=apex_time,
            upper_line=upper,
            lower_line=lower,
            convergence_angle=convergence_angle,
            volatility_compression=volatility_compression
        )
    
    def _check_breakouts(self, df: pd.DataFrame, trend_lines: List[TrendLine],
                        sr_levels: Dict) -> List[BreakoutSignal]:
        """
        돌파 신호 확인
        
        Args:
            df: OHLCV 데이터프레임
            trend_lines: 추세선 리스트
            sr_levels: Support/Resistance 레벨
            
        Returns:
            돌파 신호 리스트
        """
        signals = []
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # 추세선 돌파 확인
        for line in trend_lines:
            # 현재 추세선 가격 계산
            current_line_price = self._get_line_price(line, df.index[-1])
            prev_line_price = self._get_line_price(line, df.index[-2])
            
            # 상향 돌파
            if prev['close'] <= prev_line_price and latest['close'] > current_line_price:
                signal = self._create_breakout_signal(
                    latest, 'up', line.line_type, df
                )
                if signal:
                    signals.append(signal)
                    
            # 하향 돌파
            elif prev['close'] >= prev_line_price and latest['close'] < current_line_price:
                signal = self._create_breakout_signal(
                    latest, 'down', line.line_type, df
                )
                if signal:
                    signals.append(signal)
        
        # S/R 레벨 돌파 확인
        for resistance in sr_levels.get('resistance', []):
            if prev['close'] <= resistance and latest['close'] > resistance:
                signal = self._create_breakout_signal(
                    latest, 'up', 'resistance', df
                )
                if signal:
                    signals.append(signal)
                    
        for support in sr_levels.get('support', []):
            if prev['close'] >= support and latest['close'] < support:
                signal = self._create_breakout_signal(
                    latest, 'down', 'support', df
                )
                if signal:
                    signals.append(signal)
                    
        return signals
    
    def _get_line_price(self, line: TrendLine, timestamp: pd.Timestamp) -> float:
        """
        특정 시점의 추세선 가격 계산
        
        Args:
            line: 추세선
            timestamp: 시간
            
        Returns:
            추세선 가격
        """
        # 시간을 인덱스로 변환
        time_diff = (timestamp - line.start_time).total_seconds() / 60 / 15  # 15분 단위
        return line.slope * time_diff + line.intercept
    
    def _create_breakout_signal(self, candle: pd.Series, direction: str,
                               pattern_type: str, df: pd.DataFrame) -> Optional[BreakoutSignal]:
        """
        돌파 신호 생성
        
        Args:
            candle: 현재 캔들
            direction: 돌파 방향
            pattern_type: 패턴 타입
            df: 전체 데이터프레임
            
        Returns:
            돌파 신호 또는 None
        """
        # 거래량 확인 (평균 대비 1.5배 이상)
        avg_volume = df['volume'].tail(20).mean()
        volume_confirmation = candle['volume'] > avg_volume * 1.5
        
        # 돌파 강도 계산 (가격 변화율)
        price_change = abs(candle['close'] - df.iloc[-2]['close']) / df.iloc[-2]['close']
        strength = min(price_change * 10, 1.0)  # 0-1 정규화
        
        # False breakout 필터링
        if strength < 0.3 and not volume_confirmation:
            return None
            
        return BreakoutSignal(
            timestamp=candle.name,
            price=float(candle['close']),
            volume=float(candle['volume']),
            direction=direction,
            strength=float(strength),
            volume_confirmation=bool(volume_confirmation),
            pattern_type=pattern_type
        )
    
    def get_trend_direction(self, df: pd.DataFrame) -> str:
        """
        현재 추세 방향 판단
        
        Args:
            df: OHLCV 데이터프레임
            
        Returns:
            'up', 'down', 'sideways'
        """
        if len(df) < 50:
            return 'sideways'
            
        # 이동평균 기반 추세 판단
        ma20 = df['close'].tail(20).mean()
        ma50 = df['close'].tail(50).mean()
        current_price = df['close'].iloc[-1]
        
        if current_price > ma20 > ma50:
            return 'up'
        elif current_price < ma20 < ma50:
            return 'down'
        else:
            return 'sideways'
    
    def calculate_breakout_probability(self, pattern: TrianglePattern, 
                                      df: pd.DataFrame) -> float:
        """
        패턴 돌파 확률 계산
        
        Args:
            pattern: 삼각패턴
            df: OHLCV 데이터프레임
            
        Returns:
            돌파 확률 (0-1)
        """
        # 수렴 정도 (apex에 가까울수록 높음)
        time_to_apex = (pattern.apex_time - df.index[-1]).total_seconds() / 3600
        apex_proximity = max(0, 1 - time_to_apex / 24)  # 24시간 기준
        
        # 변동성 압축 정도
        volatility_score = pattern.volatility_compression
        
        # 패턴 타입별 기본 확률
        pattern_probs = {
            'ascending': 0.65,  # 상승 돌파 경향
            'descending': 0.35,  # 하락 돌파 경향
            'symmetric': 0.50   # 중립
        }
        base_prob = pattern_probs.get(pattern.pattern_type, 0.5)
        
        # 최종 확률 계산
        probability = base_prob * 0.5 + apex_proximity * 0.3 + volatility_score * 0.2
        
        return min(max(probability, 0), 1)