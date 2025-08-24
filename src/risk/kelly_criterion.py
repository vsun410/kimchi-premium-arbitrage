"""
Kelly Criterion Position Sizing
켈리 기준 포지션 사이징 - 최적 자본 배분
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class KellyRecommendation:
    """켈리 기준 추천"""
    strategy_name: str
    optimal_fraction: float  # 최적 자본 비율
    adjusted_fraction: float  # 조정된 비율 (보수적)
    position_size_krw: float  # KRW 포지션 크기
    position_size_btc: float  # BTC 포지션 크기
    expected_return: float  # 기대 수익률
    win_probability: float  # 승률
    risk_reward_ratio: float  # 리스크 대비 보상
    confidence_level: float  # 신뢰도 (0-1)


class KellyCriterion:
    """
    Kelly Criterion for Position Sizing
    
    Kelly Formula: f = (p*b - q) / b
    where:
    - f = fraction of capital to bet
    - p = probability of winning
    - q = probability of losing (1-p)
    - b = odds (win/loss ratio)
    
    Modified for trading:
    f = (μ - r) / σ²
    where:
    - μ = expected return
    - r = risk-free rate
    - σ² = variance of returns
    """
    
    def __init__(
        self,
        max_kelly_fraction: float = 0.25,  # 최대 25% (Kelly/4)
        risk_free_rate: float = 0.035,  # 연 3.5%
        min_win_rate: float = 0.55,  # 최소 승률 55%
        confidence_threshold: float = 0.7
    ):
        self.max_kelly_fraction = max_kelly_fraction
        self.risk_free_rate = risk_free_rate / 365  # Daily rate
        self.min_win_rate = min_win_rate
        self.confidence_threshold = confidence_threshold
        
        # 포지션 한도
        self.position_limits = {
            'min_position_krw': 100_000,  # 최소 10만원
            'max_position_krw': 10_000_000,  # 최대 1000만원
            'max_position_btc': 0.1,  # 최대 0.1 BTC
        }
        
        # 전략별 조정 계수
        self.strategy_adjustments = {
            'kimchi_premium': 0.5,  # 보수적 (전송 리스크)
            'scalping': 0.8,  # 중간
            'triangular': 0.3,  # 매우 보수적 (복잡도 높음)
            'ml_prediction': 0.6,  # ML 불확실성 고려
        }
    
    def calculate_kelly_fraction(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        strategy: str = 'kimchi_premium'
    ) -> float:
        """
        기본 켈리 비율 계산
        
        Args:
            win_rate: 승률 (0-1)
            avg_win: 평균 수익
            avg_loss: 평균 손실 (양수)
            strategy: 전략 이름
            
        Returns:
            켈리 비율
        """
        if win_rate <= 0 or win_rate >= 1:
            return 0.0
        
        if avg_loss <= 0:
            return 0.0
        
        # 기본 켈리 공식
        p = win_rate
        q = 1 - win_rate
        b = avg_win / avg_loss  # Odds
        
        kelly_f = (p * b - q) / b
        
        # 음수면 베팅하지 않음
        if kelly_f <= 0:
            return 0.0
        
        # 전략별 조정
        adjustment = self.strategy_adjustments.get(strategy, 0.5)
        adjusted_kelly = kelly_f * adjustment
        
        # 최대값 제한
        return min(adjusted_kelly, self.max_kelly_fraction)
    
    def calculate_from_returns(
        self,
        returns: pd.Series,
        strategy: str = 'kimchi_premium'
    ) -> KellyRecommendation:
        """
        과거 수익률 데이터로부터 켈리 비율 계산
        
        Args:
            returns: 과거 수익률 시리즈
            strategy: 전략 이름
            
        Returns:
            켈리 추천
        """
        if len(returns) < 30:
            logger.warning("Insufficient data for Kelly calculation")
            return self._default_recommendation(strategy)
        
        # 기본 통계
        mean_return = returns.mean()
        std_return = returns.std()
        variance = std_return ** 2
        
        # 승률 계산
        win_rate = (returns > 0).mean()
        
        # 평균 수익/손실
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        
        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = abs(losses.mean()) if len(losses) > 0 else 0
        
        # 샤프 비율 기반 켈리
        if variance > 0:
            sharpe_kelly = (mean_return - self.risk_free_rate) / variance
            sharpe_kelly = max(0, min(sharpe_kelly, 1))  # 0-1 범위
        else:
            sharpe_kelly = 0
        
        # 기본 켈리
        basic_kelly = self.calculate_kelly_fraction(
            win_rate, avg_win, avg_loss, strategy
        )
        
        # 두 방법의 평균
        optimal_fraction = (sharpe_kelly + basic_kelly) / 2
        
        # 신뢰도 계산
        confidence = self._calculate_confidence(
            win_rate, len(returns), std_return
        )
        
        # 신뢰도 기반 조정
        adjusted_fraction = optimal_fraction * confidence
        
        # 1% 룰 적용 (추가 안전장치)
        adjusted_fraction = min(adjusted_fraction, 0.01)
        
        return KellyRecommendation(
            strategy_name=strategy,
            optimal_fraction=optimal_fraction,
            adjusted_fraction=adjusted_fraction,
            position_size_krw=0,  # 나중에 계산
            position_size_btc=0,  # 나중에 계산
            expected_return=mean_return,
            win_probability=win_rate,
            risk_reward_ratio=avg_win / avg_loss if avg_loss > 0 else 0,
            confidence_level=confidence
        )
    
    def calculate_position_size(
        self,
        recommendation: KellyRecommendation,
        total_capital: float,
        btc_price: float
    ) -> KellyRecommendation:
        """
        실제 포지션 크기 계산
        
        Args:
            recommendation: 켈리 추천
            total_capital: 총 자본 (KRW)
            btc_price: BTC 가격 (KRW)
            
        Returns:
            포지션 크기가 계산된 추천
        """
        # KRW 포지션 크기
        position_krw = total_capital * recommendation.adjusted_fraction
        
        # 한도 적용
        position_krw = max(
            self.position_limits['min_position_krw'],
            min(position_krw, self.position_limits['max_position_krw'])
        )
        
        # BTC 포지션 크기
        position_btc = position_krw / btc_price
        position_btc = min(position_btc, self.position_limits['max_position_btc'])
        
        # 다시 KRW로 변환 (BTC 한도 적용 후)
        position_krw = position_btc * btc_price
        
        recommendation.position_size_krw = position_krw
        recommendation.position_size_btc = position_btc
        
        return recommendation
    
    def _calculate_confidence(
        self,
        win_rate: float,
        sample_size: int,
        std_return: float
    ) -> float:
        """
        신뢰도 계산
        
        Args:
            win_rate: 승률
            sample_size: 샘플 크기
            std_return: 수익률 표준편차
            
        Returns:
            신뢰도 (0-1)
        """
        confidence_factors = []
        
        # 1. 승률 신뢰도
        if win_rate >= self.min_win_rate:
            win_confidence = min((win_rate - 0.5) * 2, 1.0)
        else:
            win_confidence = 0.0
        confidence_factors.append(win_confidence * 0.4)
        
        # 2. 샘플 크기 신뢰도
        sample_confidence = min(sample_size / 100, 1.0)  # 100개 이상이면 최대
        confidence_factors.append(sample_confidence * 0.3)
        
        # 3. 안정성 신뢰도 (낮은 변동성)
        if std_return > 0:
            stability_confidence = max(0, 1 - std_return / 0.1)  # 10% 이상이면 0
        else:
            stability_confidence = 0
        confidence_factors.append(stability_confidence * 0.3)
        
        return sum(confidence_factors)
    
    def _default_recommendation(self, strategy: str) -> KellyRecommendation:
        """
        기본 추천 (데이터 부족시)
        """
        return KellyRecommendation(
            strategy_name=strategy,
            optimal_fraction=0.0,
            adjusted_fraction=0.0,
            position_size_krw=self.position_limits['min_position_krw'],
            position_size_btc=0.001,
            expected_return=0.0,
            win_probability=0.0,
            risk_reward_ratio=0.0,
            confidence_level=0.0
        )
    
    def dynamic_kelly_adjustment(
        self,
        base_kelly: float,
        market_conditions: Dict
    ) -> float:
        """
        시장 상황에 따른 동적 켈리 조정
        
        Args:
            base_kelly: 기본 켈리 비율
            market_conditions: 시장 상황 딕셔너리
            
        Returns:
            조정된 켈리 비율
        """
        adjustment_factor = 1.0
        
        # 1. 변동성 조정
        volatility = market_conditions.get('volatility', 0.02)
        if volatility > 0.05:  # 5% 이상 고변동성
            adjustment_factor *= 0.5
        elif volatility > 0.03:  # 3% 이상 중변동성
            adjustment_factor *= 0.75
        
        # 2. 스프레드 조정
        spread = market_conditions.get('spread', 0.001)
        if spread > 0.005:  # 0.5% 이상 스프레드
            adjustment_factor *= 0.7
        
        # 3. 거래량 조정
        volume_ratio = market_conditions.get('volume_ratio', 1.0)  # 현재/평균
        if volume_ratio < 0.5:  # 거래량 부족
            adjustment_factor *= 0.6
        
        # 4. 김치 프리미엄 극단성
        kimchi_premium = market_conditions.get('kimchi_premium', 0)
        if abs(kimchi_premium) > 3:  # 3% 이상 극단적
            adjustment_factor *= 0.5  # 위험 신호
        
        # 5. 연속 손실 조정
        consecutive_losses = market_conditions.get('consecutive_losses', 0)
        if consecutive_losses > 0:
            adjustment_factor *= (0.8 ** consecutive_losses)  # 지수적 감소
        
        return base_kelly * adjustment_factor
    
    def portfolio_kelly(
        self,
        strategies: List[Dict],
        correlations: np.ndarray,
        total_capital: float
    ) -> Dict[str, float]:
        """
        포트폴리오 켈리 - 여러 전략 동시 운용시
        
        Args:
            strategies: 전략 리스트 (각 전략의 통계 포함)
            correlations: 전략간 상관관계 행렬
            total_capital: 총 자본
            
        Returns:
            각 전략별 최적 자본 배분
        """
        n = len(strategies)
        
        # 각 전략의 기본 켈리 계산
        kelly_fractions = []
        for strategy in strategies:
            kelly = self.calculate_kelly_fraction(
                strategy['win_rate'],
                strategy['avg_win'],
                strategy['avg_loss'],
                strategy['name']
            )
            kelly_fractions.append(kelly)
        
        # 상관관계 고려한 조정
        # 높은 상관관계는 리스크 증가
        correlation_penalty = np.mean(np.abs(correlations[np.triu_indices(n, k=1)]))
        adjustment = 1 - correlation_penalty * 0.5
        
        # 조정된 켈리 비율
        adjusted_kellys = [k * adjustment for k in kelly_fractions]
        
        # 정규화 (합이 max_kelly_fraction을 넘지 않도록)
        total_kelly = sum(adjusted_kellys)
        if total_kelly > self.max_kelly_fraction:
            scale = self.max_kelly_fraction / total_kelly
            adjusted_kellys = [k * scale for k in adjusted_kellys]
        
        # 자본 배분
        allocations = {}
        for i, strategy in enumerate(strategies):
            allocations[strategy['name']] = {
                'fraction': adjusted_kellys[i],
                'capital': total_capital * adjusted_kellys[i]
            }
        
        return allocations


def analyze_kelly_sizing():
    """
    켈리 기준 포지션 사이징 분석
    """
    print("\n" + "=" * 60)
    print("  KELLY CRITERION POSITION SIZING ANALYSIS")
    print("=" * 60)
    
    kelly = KellyCriterion()
    
    # 시뮬레이션 데이터
    np.random.seed(42)
    
    # 1. 다양한 전략 시나리오
    scenarios = [
        {
            'name': 'Conservative Scalping',
            'win_rate': 0.65,
            'avg_win': 0.003,  # 0.3%
            'avg_loss': 0.002,  # 0.2%
            'strategy': 'scalping'
        },
        {
            'name': 'Aggressive Kimchi',
            'win_rate': 0.55,
            'avg_win': 0.01,  # 1%
            'avg_loss': 0.005,  # 0.5%
            'strategy': 'kimchi_premium'
        },
        {
            'name': 'ML Prediction',
            'win_rate': 0.60,
            'avg_win': 0.005,  # 0.5%
            'avg_loss': 0.003,  # 0.3%
            'strategy': 'ml_prediction'
        },
        {
            'name': 'Triangular Arbitrage',
            'win_rate': 0.70,
            'avg_win': 0.002,  # 0.2%
            'avg_loss': 0.001,  # 0.1%
            'strategy': 'triangular'
        }
    ]
    
    print("\n[Individual Strategy Analysis]")
    for scenario in scenarios:
        kelly_f = kelly.calculate_kelly_fraction(
            scenario['win_rate'],
            scenario['avg_win'],
            scenario['avg_loss'],
            scenario['strategy']
        )
        
        # 4000만원 기준 포지션 크기
        position_krw = 40_000_000 * kelly_f
        position_btc = position_krw / 159_000_000  # BTC 가격
        
        print(f"\n{scenario['name']}:")
        print(f"  Win rate: {scenario['win_rate']*100:.1f}%")
        print(f"  Risk/Reward: {scenario['avg_win']/scenario['avg_loss']:.2f}")
        print(f"  Kelly fraction: {kelly_f*100:.2f}%")
        print(f"  Position size: {position_krw:,.0f} KRW")
        print(f"  Position size: {position_btc:.4f} BTC")
    
    # 2. 과거 수익률 기반 분석
    print("\n[Historical Returns Based Analysis]")
    
    # 시뮬레이션 수익률 생성
    returns = pd.Series(np.random.normal(0.001, 0.005, 100))  # 평균 0.1%, 표준편차 0.5%
    returns[returns > 0] *= 1.2  # 수익은 좀 더 크게
    returns[returns < 0] *= 0.8  # 손실은 좀 더 작게
    
    recommendation = kelly.calculate_from_returns(returns, 'scalping')
    recommendation = kelly.calculate_position_size(
        recommendation,
        40_000_000,
        159_000_000
    )
    
    print(f"\nBased on 100 historical trades:")
    print(f"  Expected return: {recommendation.expected_return*100:.3f}%")
    print(f"  Win probability: {recommendation.win_probability*100:.1f}%")
    print(f"  Risk/Reward ratio: {recommendation.risk_reward_ratio:.2f}")
    print(f"  Confidence level: {recommendation.confidence_level:.2f}")
    print(f"  Optimal Kelly: {recommendation.optimal_fraction*100:.2f}%")
    print(f"  Adjusted Kelly: {recommendation.adjusted_fraction*100:.2f}%")
    print(f"  Position size: {recommendation.position_size_krw:,.0f} KRW")
    print(f"  Position size: {recommendation.position_size_btc:.4f} BTC")
    
    # 3. 시장 상황별 조정
    print("\n[Market Condition Adjustments]")
    
    market_conditions = [
        {'name': 'Normal', 'volatility': 0.02, 'spread': 0.001, 'kimchi_premium': 0.5},
        {'name': 'High Volatility', 'volatility': 0.06, 'spread': 0.002, 'kimchi_premium': 1.0},
        {'name': 'Wide Spread', 'volatility': 0.03, 'spread': 0.008, 'kimchi_premium': 0.3},
        {'name': 'Extreme Premium', 'volatility': 0.04, 'spread': 0.003, 'kimchi_premium': 4.0},
    ]
    
    base_kelly = 0.05  # 5% 기본
    
    for condition in market_conditions:
        adjusted = kelly.dynamic_kelly_adjustment(base_kelly, condition)
        print(f"\n{condition['name']}:")
        print(f"  Base Kelly: {base_kelly*100:.1f}%")
        print(f"  Adjusted Kelly: {adjusted*100:.2f}%")
        print(f"  Position: {40_000_000 * adjusted:,.0f} KRW")
    
    # 4. 포트폴리오 켈리
    print("\n[Portfolio Kelly Allocation]")
    
    # 상관관계 행렬 (전략간)
    correlations = np.array([
        [1.0, 0.3, 0.2, 0.1],
        [0.3, 1.0, 0.4, 0.2],
        [0.2, 0.4, 1.0, 0.3],
        [0.1, 0.2, 0.3, 1.0]
    ])
    
    portfolio_strategies = [
        {'name': 'scalping', 'win_rate': 0.65, 'avg_win': 0.003, 'avg_loss': 0.002},
        {'name': 'kimchi_premium', 'win_rate': 0.55, 'avg_win': 0.01, 'avg_loss': 0.005},
        {'name': 'ml_prediction', 'win_rate': 0.60, 'avg_win': 0.005, 'avg_loss': 0.003},
        {'name': 'triangular', 'win_rate': 0.70, 'avg_win': 0.002, 'avg_loss': 0.001},
    ]
    
    allocations = kelly.portfolio_kelly(
        portfolio_strategies,
        correlations,
        40_000_000
    )
    
    print("\nOptimal portfolio allocation:")
    total_allocated = 0
    for strategy_name, allocation in allocations.items():
        print(f"  {strategy_name}: {allocation['fraction']*100:.2f}% = {allocation['capital']:,.0f} KRW")
        total_allocated += allocation['capital']
    
    print(f"\nTotal allocated: {total_allocated:,.0f} KRW ({total_allocated/40_000_000*100:.1f}%)")
    print(f"Cash reserve: {40_000_000 - total_allocated:,.0f} KRW")
    
    # 결론
    print("\n[Key Insights]")
    print("1. Never bet more than Kelly suggests (over-betting guarantees ruin)")
    print("2. Use 1/4 Kelly (25% of optimal) for safety")
    print("3. Adjust dynamically based on market conditions")
    print("4. Keep cash reserve for unexpected opportunities")
    print("5. Monitor and update Kelly regularly with new data")


if __name__ == "__main__":
    analyze_kelly_sizing()