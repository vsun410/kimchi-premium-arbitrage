"""
평가 리포트 생성기
HTML/PDF 형식의 종합 평가 리포트 생성
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
import json
from jinja2 import Template
import base64
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')


class ReportGenerator:
    """
    모델 평가 리포트 생성기
    
    HTML 형식의 상세한 평가 리포트 생성
    """
    
    def __init__(self, save_dir: str = "./evaluation_reports"):
        """
        리포트 생성기 초기화
        
        Args:
            save_dir: 리포트 저장 디렉토리
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.figures = {}
        
    def generate_report(
        self,
        evaluation_results: Dict,
        ab_test_results: Optional[List] = None,
        model_comparison: Optional[pd.DataFrame] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        종합 평가 리포트 생성
        
        Args:
            evaluation_results: 모델별 평가 결과
            ab_test_results: A/B 테스트 결과
            model_comparison: 모델 비교 테이블
            metadata: 추가 메타데이터
            
        Returns:
            HTML 리포트 경로
        """
        # 차트 생성
        self._create_charts(evaluation_results)
        
        # HTML 템플릿
        html_template = self._get_html_template()
        
        # 데이터 준비
        report_data = {
            'title': 'Model Evaluation Report',
            'generated_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'evaluation_results': self._format_evaluation_results(evaluation_results),
            'model_comparison': model_comparison.to_html(classes='table table-striped') if model_comparison is not None else '',
            'ab_test_results': self._format_ab_test_results(ab_test_results),
            'charts': self.figures,
            'metadata': metadata or {},
            'summary': self._generate_summary(evaluation_results)
        }
        
        # HTML 생성
        template = Template(html_template)
        html_content = template.render(**report_data)
        
        # 파일 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.save_dir / f"evaluation_report_{timestamp}.html"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Report generated: {report_path}")
        return str(report_path)
    
    def _create_charts(self, evaluation_results: Dict):
        """차트 생성"""
        # 1. 샤프비율 비교
        self._create_sharpe_comparison(evaluation_results)
        
        # 2. 수익률 비교
        self._create_return_comparison(evaluation_results)
        
        # 3. 리스크 메트릭 비교
        self._create_risk_comparison(evaluation_results)
        
        # 4. 거래 효율성 비교
        self._create_efficiency_comparison(evaluation_results)
    
    def _create_sharpe_comparison(self, results: Dict):
        """샤프비율 비교 차트"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        models = list(results.keys())
        sharpe_ratios = [r.sharpe_ratio if hasattr(r, 'sharpe_ratio') else 0 for r in results.values()]
        
        bars = ax.bar(models, sharpe_ratios)
        
        # 색상 설정
        colors = ['green' if s > 0 else 'red' for s in sharpe_ratios]
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        ax.set_ylabel('Sharpe Ratio')
        ax.set_title('Sharpe Ratio Comparison')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.axhline(y=1.5, color='blue', linestyle='--', linewidth=0.5, label='Target (1.5)')
        ax.legend()
        
        self.figures['sharpe_comparison'] = self._fig_to_base64(fig)
        plt.close()
    
    def _create_return_comparison(self, results: Dict):
        """수익률 비교 차트"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        models = list(results.keys())
        
        # Total Return
        ax = axes[0, 0]
        total_returns = [r.total_return if hasattr(r, 'total_return') else 0 for r in results.values()]
        ax.bar(models, total_returns)
        ax.set_ylabel('Total Return (%)')
        ax.set_title('Total Return')
        
        # Annual Return
        ax = axes[0, 1]
        annual_returns = [r.annual_return if hasattr(r, 'annual_return') else 0 for r in results.values()]
        ax.bar(models, annual_returns)
        ax.set_ylabel('Annual Return (%)')
        ax.set_title('Annual Return')
        
        # Max Drawdown
        ax = axes[1, 0]
        max_drawdowns = [r.max_drawdown if hasattr(r, 'max_drawdown') else 0 for r in results.values()]
        ax.bar(models, max_drawdowns, color='red')
        ax.set_ylabel('Max Drawdown (%)')
        ax.set_title('Maximum Drawdown')
        
        # Win Rate
        ax = axes[1, 1]
        win_rates = [r.win_rate if hasattr(r, 'win_rate') else 0 for r in results.values()]
        ax.bar(models, win_rates, color='green')
        ax.set_ylabel('Win Rate (%)')
        ax.set_title('Win Rate')
        ax.axhline(y=50, color='black', linestyle='--', linewidth=0.5)
        
        plt.suptitle('Return Metrics Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        self.figures['return_comparison'] = self._fig_to_base64(fig)
        plt.close()
    
    def _create_risk_comparison(self, results: Dict):
        """리스크 메트릭 비교"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        models = list(results.keys())
        metrics = ['volatility', 'var_95', 'cvar_95']
        
        x = np.arange(len(models))
        width = 0.25
        
        for i, metric in enumerate(metrics):
            values = [getattr(r, metric, 0) for r in results.values()]
            ax.bar(x + i * width, values, width, label=metric.upper())
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Risk (%)')
        ax.set_title('Risk Metrics Comparison')
        ax.set_xticks(x + width)
        ax.set_xticklabels(models)
        ax.legend()
        
        self.figures['risk_comparison'] = self._fig_to_base64(fig)
        plt.close()
    
    def _create_efficiency_comparison(self, results: Dict):
        """거래 효율성 비교"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        models = list(results.keys())
        
        # Profit Factor
        ax = axes[0]
        profit_factors = [r.profit_factor if hasattr(r, 'profit_factor') else 0 for r in results.values()]
        ax.bar(models, profit_factors)
        ax.set_ylabel('Profit Factor')
        ax.set_title('Profit Factor')
        ax.axhline(y=1, color='red', linestyle='--', linewidth=0.5)
        
        # Calmar Ratio
        ax = axes[1]
        calmar_ratios = [r.calmar_ratio if hasattr(r, 'calmar_ratio') else 0 for r in results.values()]
        ax.bar(models, calmar_ratios)
        ax.set_ylabel('Calmar Ratio')
        ax.set_title('Calmar Ratio')
        
        plt.suptitle('Trading Efficiency Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        self.figures['efficiency_comparison'] = self._fig_to_base64(fig)
        plt.close()
    
    def _fig_to_base64(self, fig) -> str:
        """matplotlib figure를 base64 문자열로 변환"""
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        return f"data:image/png;base64,{img_base64}"
    
    def _format_evaluation_results(self, results: Dict) -> str:
        """평가 결과 포맷팅"""
        if not results:
            return ""
        
        html = "<div class='evaluation-results'>"
        
        for model_name, result in results.items():
            html += f"""
            <div class='model-result'>
                <h3>{model_name}</h3>
                <table class='table table-sm'>
                    <tr><td>Total Return</td><td>{getattr(result, 'total_return', 0):.2f}%</td></tr>
                    <tr><td>Annual Return</td><td>{getattr(result, 'annual_return', 0):.2f}%</td></tr>
                    <tr><td>Sharpe Ratio</td><td>{getattr(result, 'sharpe_ratio', 0):.3f}</td></tr>
                    <tr><td>Max Drawdown</td><td>{getattr(result, 'max_drawdown', 0):.2f}%</td></tr>
                    <tr><td>Win Rate</td><td>{getattr(result, 'win_rate', 0):.1f}%</td></tr>
                </table>
            </div>
            """
        
        html += "</div>"
        return html
    
    def _format_ab_test_results(self, results: Optional[List]) -> str:
        """A/B 테스트 결과 포맷팅"""
        if not results:
            return ""
        
        html = "<div class='ab-test-results'><h3>A/B Test Results</h3>"
        
        for result in results:
            html += f"""
            <div class='test-result'>
                <h4>{result.model_a} vs {result.model_b}</h4>
                <p>P-value: {result.p_value:.4f}</p>
                <p>Significant: {'Yes' if result.is_significant else 'No'}</p>
                <p>Effect Size: {result.effect_size:.3f}</p>
                <p>Winner: {result.winner or 'None'}</p>
            </div>
            """
        
        html += "</div>"
        return html
    
    def _generate_summary(self, results: Dict) -> str:
        """요약 생성"""
        if not results:
            return "No evaluation results available."
        
        # 최고 성능 모델 찾기
        best_sharpe = -float('inf')
        best_model = None
        
        for name, result in results.items():
            if hasattr(result, 'sharpe_ratio') and result.sharpe_ratio > best_sharpe:
                best_sharpe = result.sharpe_ratio
                best_model = name
        
        summary = f"""
        <div class='summary'>
            <h3>Executive Summary</h3>
            <ul>
                <li><strong>Models Evaluated:</strong> {len(results)}</li>
                <li><strong>Best Model (Sharpe):</strong> {best_model} ({best_sharpe:.3f})</li>
                <li><strong>Evaluation Date:</strong> {datetime.now().strftime('%Y-%m-%d')}</li>
            </ul>
        </div>
        """
        
        return summary
    
    def _get_html_template(self) -> str:
        """HTML 템플릿 반환"""
        return """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{{ title }}</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }
                .container {
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0 0 10px rgba(0,0,0,0.1);
                }
                h1 {
                    color: #333;
                    border-bottom: 2px solid #4CAF50;
                    padding-bottom: 10px;
                }
                h2 {
                    color: #555;
                    margin-top: 30px;
                }
                h3 {
                    color: #666;
                }
                .table {
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }
                .table th, .table td {
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }
                .table th {
                    background-color: #4CAF50;
                    color: white;
                }
                .table-striped tbody tr:nth-of-type(odd) {
                    background-color: #f9f9f9;
                }
                .chart-container {
                    margin: 20px 0;
                    text-align: center;
                }
                .chart-container img {
                    max-width: 100%;
                    height: auto;
                }
                .model-result {
                    margin: 20px 0;
                    padding: 15px;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                }
                .summary {
                    background-color: #e7f5e7;
                    padding: 15px;
                    border-radius: 5px;
                    margin: 20px 0;
                }
                .metadata {
                    color: #888;
                    font-size: 0.9em;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>{{ title }}</h1>
                <p class="metadata">Generated: {{ generated_date }}</p>
                
                {{ summary | safe }}
                
                <h2>Model Comparison</h2>
                {{ model_comparison | safe }}
                
                <h2>Performance Charts</h2>
                <div class="chart-container">
                    <h3>Sharpe Ratio Comparison</h3>
                    <img src="{{ charts.sharpe_comparison }}" alt="Sharpe Ratio">
                </div>
                
                <div class="chart-container">
                    <h3>Return Metrics</h3>
                    <img src="{{ charts.return_comparison }}" alt="Returns">
                </div>
                
                <div class="chart-container">
                    <h3>Risk Metrics</h3>
                    <img src="{{ charts.risk_comparison }}" alt="Risk">
                </div>
                
                <div class="chart-container">
                    <h3>Trading Efficiency</h3>
                    <img src="{{ charts.efficiency_comparison }}" alt="Efficiency">
                </div>
                
                <h2>Detailed Results</h2>
                {{ evaluation_results | safe }}
                
                {{ ab_test_results | safe }}
                
                <footer>
                    <p class="metadata">Kimchi Premium Arbitrage System - Model Evaluation Report</p>
                </footer>
            </div>
        </body>
        </html>
        """