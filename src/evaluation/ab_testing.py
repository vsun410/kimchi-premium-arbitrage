"""
A/B 테스트 프레임워크
모델 간 통계적 유의성 검증 및 비교 분석
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


@dataclass
class ABTestResult:
    """A/B 테스트 결과"""
    model_a: str
    model_b: str
    metric: str
    
    # 기본 통계
    mean_a: float
    mean_b: float
    std_a: float
    std_b: float
    
    # 통계 검정
    t_statistic: float
    p_value: float
    is_significant: bool
    confidence_level: float
    
    # 효과 크기
    effect_size: float  # Cohen's d
    improvement: float  # % improvement
    
    # 신뢰구간
    ci_lower: float
    ci_upper: float
    
    # 추가 정보
    sample_size_a: int
    sample_size_b: int
    test_type: str
    winner: Optional[str] = None


class ABTestFramework:
    """
    A/B 테스트 프레임워크
    
    모델 성능을 통계적으로 비교하고 유의미한 차이를 검증
    """
    
    def __init__(self, confidence_level: float = 0.95):
        """
        프레임워크 초기화
        
        Args:
            confidence_level: 신뢰수준 (기본 95%)
        """
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        self.test_results = []
        
    def compare_models(
        self,
        returns_a: np.ndarray,
        returns_b: np.ndarray,
        model_a_name: str = "Model A",
        model_b_name: str = "Model B",
        metric_name: str = "returns"
    ) -> ABTestResult:
        """
        두 모델의 수익률 비교
        
        Args:
            returns_a: 모델 A의 수익률
            returns_b: 모델 B의 수익률
            model_a_name: 모델 A 이름
            model_b_name: 모델 B 이름
            metric_name: 메트릭 이름
            
        Returns:
            A/B 테스트 결과
        """
        # 기본 통계
        mean_a = np.mean(returns_a)
        mean_b = np.mean(returns_b)
        std_a = np.std(returns_a, ddof=1)
        std_b = np.std(returns_b, ddof=1)
        n_a = len(returns_a)
        n_b = len(returns_b)
        
        # T-test (독립 표본)
        t_stat, p_value = stats.ttest_ind(returns_a, returns_b, equal_var=False)
        
        # 효과 크기 (Cohen's d)
        pooled_std = np.sqrt(((n_a - 1) * std_a**2 + (n_b - 1) * std_b**2) / (n_a + n_b - 2))
        effect_size = (mean_a - mean_b) / pooled_std if pooled_std > 0 else 0
        
        # 개선도
        improvement = ((mean_a - mean_b) / abs(mean_b) * 100) if mean_b != 0 else 0
        
        # 신뢰구간 (차이의)
        se_diff = np.sqrt(std_a**2/n_a + std_b**2/n_b)
        t_critical = stats.t.ppf(1 - self.alpha/2, df=min(n_a, n_b) - 1)
        ci_lower = (mean_a - mean_b) - t_critical * se_diff
        ci_upper = (mean_a - mean_b) + t_critical * se_diff
        
        # 유의성 판단
        is_significant = bool(p_value < self.alpha)
        
        # 승자 결정
        if is_significant:
            winner = model_a_name if mean_a > mean_b else model_b_name
        else:
            winner = None
        
        result = ABTestResult(
            model_a=model_a_name,
            model_b=model_b_name,
            metric=metric_name,
            mean_a=mean_a,
            mean_b=mean_b,
            std_a=std_a,
            std_b=std_b,
            t_statistic=t_stat,
            p_value=p_value,
            is_significant=is_significant,
            confidence_level=self.confidence_level,
            effect_size=effect_size,
            improvement=improvement,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            sample_size_a=n_a,
            sample_size_b=n_b,
            test_type="t-test",
            winner=winner
        )
        
        self.test_results.append(result)
        return result
    
    def mann_whitney_test(
        self,
        data_a: np.ndarray,
        data_b: np.ndarray,
        model_a_name: str = "Model A",
        model_b_name: str = "Model B"
    ) -> ABTestResult:
        """
        Mann-Whitney U 검정 (비모수 검정)
        
        정규분포를 따르지 않는 데이터에 사용
        """
        # 기본 통계
        median_a = np.median(data_a)
        median_b = np.median(data_b)
        
        # Mann-Whitney U test
        u_stat, p_value = stats.mannwhitneyu(data_a, data_b, alternative='two-sided')
        
        # 효과 크기 (rank-biserial correlation)
        n_a, n_b = len(data_a), len(data_b)
        effect_size = 1 - (2 * u_stat) / (n_a * n_b)
        
        # 개선도
        improvement = ((median_a - median_b) / abs(median_b) * 100) if median_b != 0 else 0
        
        # 유의성
        is_significant = p_value < self.alpha
        winner = model_a_name if median_a > median_b and is_significant else (
            model_b_name if median_b > median_a and is_significant else None
        )
        
        result = ABTestResult(
            model_a=model_a_name,
            model_b=model_b_name,
            metric="median_returns",
            mean_a=median_a,
            mean_b=median_b,
            std_a=np.std(data_a),
            std_b=np.std(data_b),
            t_statistic=u_stat,
            p_value=p_value,
            is_significant=is_significant,
            confidence_level=self.confidence_level,
            effect_size=effect_size,
            improvement=improvement,
            ci_lower=0,  # Bootstrap으로 계산 필요
            ci_upper=0,
            sample_size_a=n_a,
            sample_size_b=n_b,
            test_type="mann-whitney",
            winner=winner
        )
        
        return result
    
    def bootstrap_confidence_interval(
        self,
        data_a: np.ndarray,
        data_b: np.ndarray,
        n_bootstrap: int = 10000,
        metric_func: callable = np.mean
    ) -> Tuple[float, float, float]:
        """
        부트스트랩 신뢰구간 계산
        
        Args:
            data_a: 모델 A 데이터
            data_b: 모델 B 데이터
            n_bootstrap: 부트스트랩 샘플 수
            metric_func: 계산할 메트릭 함수
            
        Returns:
            (차이의 점추정, 하한, 상한)
        """
        differences = []
        
        for _ in range(n_bootstrap):
            # 재샘플링
            sample_a = np.random.choice(data_a, size=len(data_a), replace=True)
            sample_b = np.random.choice(data_b, size=len(data_b), replace=True)
            
            # 메트릭 차이 계산
            diff = metric_func(sample_a) - metric_func(sample_b)
            differences.append(diff)
        
        differences = np.array(differences)
        
        # 신뢰구간
        alpha = 1 - self.confidence_level
        lower = np.percentile(differences, alpha/2 * 100)
        upper = np.percentile(differences, (1 - alpha/2) * 100)
        point_estimate = np.mean(differences)
        
        return point_estimate, lower, upper
    
    def bayesian_ab_test(
        self,
        successes_a: int,
        trials_a: int,
        successes_b: int,
        trials_b: int,
        n_simulations: int = 100000
    ) -> Dict[str, float]:
        """
        베이지안 A/B 테스트
        
        이항 데이터에 대한 베이지안 분석
        
        Args:
            successes_a: 모델 A 성공 횟수
            trials_a: 모델 A 시도 횟수
            successes_b: 모델 B 성공 횟수
            trials_b: 모델 B 시도 횟수
            n_simulations: 시뮬레이션 횟수
            
        Returns:
            베이지안 분석 결과
        """
        # Beta 분포에서 샘플링
        # Prior: Beta(1,1) = Uniform
        posterior_a = np.random.beta(successes_a + 1, trials_a - successes_a + 1, n_simulations)
        posterior_b = np.random.beta(successes_b + 1, trials_b - successes_b + 1, n_simulations)
        
        # A가 B보다 좋을 확률
        prob_a_better = np.mean(posterior_a > posterior_b)
        
        # 예상 개선도
        expected_improvement = np.mean(posterior_a - posterior_b)
        
        # 신뢰구간
        diff = posterior_a - posterior_b
        ci_lower = np.percentile(diff, 2.5)
        ci_upper = np.percentile(diff, 97.5)
        
        return {
            'prob_a_better': prob_a_better,
            'prob_b_better': 1 - prob_a_better,
            'expected_improvement': expected_improvement,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'mean_a': np.mean(posterior_a),
            'mean_b': np.mean(posterior_b)
        }
    
    def power_analysis(
        self,
        effect_size: float,
        alpha: float = None,
        power: float = 0.8
    ) -> int:
        """
        검정력 분석 - 필요한 샘플 크기 계산
        
        Args:
            effect_size: 예상 효과 크기
            alpha: 유의수준
            power: 원하는 검정력
            
        Returns:
            필요한 샘플 크기
        """
        from statsmodels.stats.power import ttest_power
        
        if alpha is None:
            alpha = self.alpha
        
        # 샘플 크기 계산
        from statsmodels.stats.power import tt_ind_solve_power
        
        sample_size = tt_ind_solve_power(
            effect_size=effect_size,
            alpha=alpha,
            power=power,
            ratio=1,
            alternative='two-sided'
        )
        
        return int(np.ceil(sample_size))
    
    def multiple_comparison_correction(
        self,
        p_values: List[float],
        method: str = 'bonferroni'
    ) -> List[float]:
        """
        다중 비교 보정
        
        Args:
            p_values: p-value 리스트
            method: 보정 방법 ('bonferroni', 'holm', 'fdr')
            
        Returns:
            보정된 p-value
        """
        n = len(p_values)
        
        if method == 'bonferroni':
            # Bonferroni correction
            adjusted = [min(p * n, 1.0) for p in p_values]
            
        elif method == 'holm':
            # Holm-Bonferroni method
            sorted_indices = np.argsort(p_values)
            adjusted = np.zeros(n)
            
            for i, idx in enumerate(sorted_indices):
                adjusted[idx] = min(p_values[idx] * (n - i), 1.0)
                if i > 0:
                    adjusted[idx] = max(adjusted[idx], adjusted[sorted_indices[i-1]])
                    
        elif method == 'fdr':
            # Benjamini-Hochberg FDR
            from statsmodels.stats.multitest import multipletests
            _, adjusted, _, _ = multipletests(p_values, alpha=self.alpha, method='fdr_bh')
            adjusted = list(adjusted)
            
        else:
            adjusted = p_values
            
        return adjusted
    
    def plot_ab_results(self, result: ABTestResult):
        """A/B 테스트 결과 시각화"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 1. 평균 비교
        ax1 = axes[0]
        models = [result.model_a, result.model_b]
        means = [result.mean_a, result.mean_b]
        stds = [result.std_a, result.std_b]
        
        bars = ax1.bar(models, means, yerr=stds, capsize=10)
        if result.winner:
            winner_idx = 0 if result.winner == result.model_a else 1
            bars[winner_idx].set_color('green')
            bars[1-winner_idx].set_color('red')
        
        ax1.set_ylabel(result.metric)
        ax1.set_title('Mean Comparison')
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # 2. 신뢰구간
        ax2 = axes[1]
        diff = result.mean_a - result.mean_b
        ax2.barh(0, diff, xerr=[[diff-result.ci_lower], [result.ci_upper-diff]], 
                 capsize=10, height=0.5)
        ax2.axvline(x=0, color='red', linestyle='--')
        ax2.set_xlabel('Difference (A - B)')
        ax2.set_title(f'{result.confidence_level*100}% Confidence Interval')
        ax2.set_ylim(-0.5, 0.5)
        ax2.set_yticks([])
        
        # 3. P-value와 효과 크기
        ax3 = axes[2]
        info_text = f"P-value: {result.p_value:.4f}\n"
        info_text += f"Effect Size: {result.effect_size:.3f}\n"
        info_text += f"Improvement: {result.improvement:.1f}%\n"
        info_text += f"Significant: {'Yes' if result.is_significant else 'No'}\n"
        if result.winner:
            info_text += f"Winner: {result.winner}"
        
        ax3.text(0.5, 0.5, info_text, transform=ax3.transAxes,
                fontsize=12, verticalalignment='center',
                horizontalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax3.set_title('Statistical Summary')
        ax3.axis('off')
        
        plt.suptitle(f'A/B Test: {result.model_a} vs {result.model_b}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def generate_report(self) -> pd.DataFrame:
        """테스트 결과 리포트 생성"""
        if not self.test_results:
            return pd.DataFrame()
        
        report_data = []
        for result in self.test_results:
            report_data.append({
                'Model A': result.model_a,
                'Model B': result.model_b,
                'Metric': result.metric,
                'Mean A': f"{result.mean_a:.4f}",
                'Mean B': f"{result.mean_b:.4f}",
                'P-value': f"{result.p_value:.4f}",
                'Significant': '✓' if result.is_significant else '✗',
                'Effect Size': f"{result.effect_size:.3f}",
                'Improvement (%)': f"{result.improvement:.1f}",
                'Winner': result.winner or 'None',
                'Test Type': result.test_type
            })
        
        return pd.DataFrame(report_data)