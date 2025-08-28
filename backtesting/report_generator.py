"""
Report Generator for Backtesting Results
백테스팅 결과 리포트 생성
"""

import pandas as pd
import json
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    리포트 생성기
    - HTML 대시보드 생성
    - JSON 상세 결과
    - 거래 내역 CSV
    """
    
    def __init__(self, output_dir: str = "reports/backtesting"):
        """
        Args:
            output_dir: 리포트 출력 디렉토리
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_full_report(self, performance_summary: Dict,
                           portfolio_history: List,
                           trades: List,
                           signals: List) -> str:
        """
        전체 리포트 생성
        
        Args:
            performance_summary: 성과 요약
            portfolio_history: 포트폴리오 히스토리
            trades: 거래 기록
            signals: 신호 기록
            
        Returns:
            리포트 파일 경로
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_name = f"backtest_report_{timestamp}"
        
        # JSON 리포트
        json_path = self.save_json_report(
            performance_summary, 
            f"{report_name}.json"
        )
        
        # CSV 거래 내역
        csv_path = self.save_trades_csv(
            trades,
            f"{report_name}_trades.csv"
        )
        
        # HTML 리포트
        html_path = self.generate_html_report(
            performance_summary,
            portfolio_history,
            trades,
            signals,
            f"{report_name}.html"
        )
        
        logger.info(f"Reports generated: {report_name}")
        return str(html_path)
    
    def save_json_report(self, data: Dict, filename: str) -> Path:
        """JSON 리포트 저장"""
        filepath = self.output_dir / filename
        
        # datetime 객체 처리
        def json_serializer(obj):
            if isinstance(obj, datetime):
                return obj.strftime('%Y-%m-%d %H:%M:%S')
            raise TypeError(f"Type {type(obj)} not serializable")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=json_serializer)
        
        return filepath
    
    def save_trades_csv(self, trades: List, filename: str) -> Path:
        """거래 내역 CSV 저장"""
        filepath = self.output_dir / filename
        
        # 거래 데이터를 DataFrame으로 변환
        trade_data = []
        for trade in trades:
            trade_data.append({
                'timestamp': trade.timestamp,
                'exchange': trade.exchange,
                'symbol': trade.symbol,
                'side': trade.side.value,
                'price': trade.price,
                'amount': trade.amount,
                'value': trade.value,
                'fee': trade.fee,
                'position_side': trade.position_side.value
            })
        
        df = pd.DataFrame(trade_data)
        df.to_csv(filepath, index=False)
        
        return filepath
    
    def generate_html_report(self, performance_summary: Dict,
                           portfolio_history: List,
                           trades: List,
                           signals: List,
                           filename: str) -> Path:
        """HTML 리포트 생성"""
        filepath = self.output_dir / filename
        
        # 포트폴리오 가치 차트 데이터
        portfolio_data = []
        for p in portfolio_history:
            portfolio_data.append({
                'timestamp': p.timestamp.strftime('%Y-%m-%d %H:%M'),
                'value': p.total_value,
                'cash_krw': p.cash.get('KRW', 0),
                'cash_usd': p.cash.get('USD', 0),
                'unrealized_pnl': p.unrealized_pnl,
                'realized_pnl': p.realized_pnl
            })
        
        # HTML 템플릿
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>백테스팅 리포트 - Kimchi Premium Arbitrage</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-bottom: 1px solid #ecf0f1;
            padding-bottom: 5px;
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #3498db;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #2c3e50;
        }}
        .metric-label {{
            font-size: 12px;
            color: #7f8c8d;
            text-transform: uppercase;
            margin-top: 5px;
        }}
        .positive {{
            color: #27ae60;
        }}
        .negative {{
            color: #e74c3c;
        }}
        .chart-container {{
            margin: 30px 0;
            border: 1px solid #ecf0f1;
            border-radius: 8px;
            padding: 10px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th {{
            background: #3498db;
            color: white;
            padding: 10px;
            text-align: left;
        }}
        td {{
            padding: 8px;
            border-bottom: 1px solid #ecf0f1;
        }}
        tr:hover {{
            background: #f8f9fa;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 백테스팅 리포트 - Dynamic Hedge Strategy</h1>
        
        <h2>📈 성과 요약</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-value {self._get_color_class(performance_summary.get('total_return', 0))}">
                    {performance_summary.get('total_return', 0):.2f}%
                </div>
                <div class="metric-label">총 수익률</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-value">
                    ₩{performance_summary.get('total_return_krw', 0):,.0f}
                </div>
                <div class="metric-label">총 수익 (KRW)</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-value {self._get_color_class(performance_summary.get('monthly_return', 0))}">
                    {performance_summary.get('monthly_return', 0):.2f}%
                </div>
                <div class="metric-label">월 평균 수익률</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-value">
                    ₩{performance_summary.get('monthly_return_krw', 0):,.0f}
                </div>
                <div class="metric-label">월 평균 수익 (KRW)</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-value">
                    {performance_summary.get('sharpe_ratio', 0):.2f}
                </div>
                <div class="metric-label">Sharpe Ratio</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-value">
                    {performance_summary.get('calmar_ratio', 0):.2f}
                </div>
                <div class="metric-label">Calmar Ratio</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-value negative">
                    -{performance_summary.get('max_drawdown', 0):.2f}%
                </div>
                <div class="metric-label">최대 낙폭</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-value">
                    {performance_summary.get('win_rate', 0):.1f}%
                </div>
                <div class="metric-label">승률</div>
            </div>
        </div>
        
        <h2>📊 포트폴리오 가치 추이</h2>
        <div class="chart-container">
            <div id="portfolio-chart"></div>
        </div>
        
        <h2>📋 거래 통계</h2>
        <table>
            <tr>
                <th>항목</th>
                <th>값</th>
            </tr>
            <tr>
                <td>총 거래 횟수</td>
                <td>{performance_summary.get('total_trades', 0)}</td>
            </tr>
            <tr>
                <td>업비트 거래</td>
                <td>{performance_summary.get('upbit_trades', 0)}</td>
            </tr>
            <tr>
                <td>바이낸스 거래</td>
                <td>{performance_summary.get('binance_trades', 0)}</td>
            </tr>
            <tr>
                <td>총 거래 수수료</td>
                <td>₩{performance_summary.get('total_fees', 0):,.0f}</td>
            </tr>
            <tr>
                <td>Profit Factor</td>
                <td>{performance_summary.get('profit_factor', 0):.2f}</td>
            </tr>
            <tr>
                <td>거래 기간</td>
                <td>{performance_summary.get('trading_days', 0)}일</td>
            </tr>
        </table>
        
        <h2>📅 기간</h2>
        <p>
            <strong>시작:</strong> {performance_summary.get('start_date', 'N/A')}<br>
            <strong>종료:</strong> {performance_summary.get('end_date', 'N/A')}
        </p>
        
        <script>
            // 포트폴리오 차트
            var portfolioData = {json.dumps(portfolio_data)};
            
            var trace1 = {{
                x: portfolioData.map(d => d.timestamp),
                y: portfolioData.map(d => d.value),
                type: 'scatter',
                mode: 'lines',
                name: '포트폴리오 가치',
                line: {{color: '#3498db', width: 2}}
            }};
            
            var layout = {{
                title: '포트폴리오 가치 변화',
                xaxis: {{title: '시간'}},
                yaxis: {{title: '가치 (KRW)', tickformat: ',.0f'}},
                hovermode: 'x unified'
            }};
            
            Plotly.newPlot('portfolio-chart', [trace1], layout);
        </script>
    </div>
</body>
</html>
"""
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return filepath
    
    def _get_color_class(self, value: float) -> str:
        """값에 따른 색상 클래스 반환"""
        return 'positive' if value >= 0 else 'negative'