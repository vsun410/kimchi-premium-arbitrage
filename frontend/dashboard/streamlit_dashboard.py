"""
Streamlit Real-time Dashboard
실시간 모니터링 대시보드
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import asyncio
from datetime import datetime, timedelta
import json
from pathlib import Path
import time

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.core.exchange_manager import ExchangeManager, ExchangeConfig
from src.utils.exchange_rate_manager import get_exchange_rate_manager

# 페이지 설정
st.set_page_config(
    page_title="Kimchi Premium Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일
st.markdown("""
<style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        margin: 5px;
    }
    .positive {
        color: #00cc00;
    }
    .negative {
        color: #ff0000;
    }
</style>
""", unsafe_allow_html=True)


class Dashboard:
    def __init__(self):
        self.exchange_manager = None
        self.rate_manager = get_exchange_rate_manager()
        self.kimchi_history = []
        self.price_history = []
        self.ma_history = []
        self.trades = []
        
    async def initialize(self):
        """초기화"""
        if not self.exchange_manager:
            self.exchange_manager = ExchangeManager()
            
            # Upbit 추가
            await self.exchange_manager.add_exchange(
                ExchangeConfig(name='upbit', options={'defaultType': 'spot'})
            )
            
            # Binance 추가
            await self.exchange_manager.add_exchange(
                ExchangeConfig(name='binance', options={'defaultType': 'spot'})
            )
    
    async def fetch_market_data(self):
        """시장 데이터 수집"""
        try:
            # Upbit
            upbit_ticker = await self.exchange_manager.get_ticker('upbit', 'BTC/KRW')
            upbit_orderbook = await self.exchange_manager.get_orderbook('upbit', 'BTC/KRW', 5)
            
            # Binance
            binance_ticker = await self.exchange_manager.get_ticker('binance', 'BTC/USDT')
            
            # 환율
            usd_krw = self.rate_manager.current_rate
            
            # 김프 계산
            kimchi = self.rate_manager.calculate_kimchi_premium(
                upbit_ticker['last'],
                binance_ticker['last'],
                datetime.now()
            )
            
            return {
                'timestamp': datetime.now(),
                'upbit_price': upbit_ticker['last'],
                'binance_price': binance_ticker['last'],
                'usd_krw': usd_krw,
                'kimchi_premium': kimchi,
                'upbit_bid': upbit_orderbook['bids'][0][0] if upbit_orderbook['bids'] else 0,
                'upbit_ask': upbit_orderbook['asks'][0][0] if upbit_orderbook['asks'] else 0
            }
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            return None
    
    def load_paper_trading_results(self):
        """Paper Trading 결과 로드"""
        log_dir = Path("paper_trading_logs")
        if not log_dir.exists():
            return None
        
        # 최신 리포트 찾기
        reports = list(log_dir.glob("report_*.json"))
        if not reports:
            return None
        
        latest_report = max(reports, key=lambda p: p.stat().st_mtime)
        
        with open(latest_report, 'r') as f:
            return json.load(f)
    
    def render_header(self):
        """헤더 렌더링"""
        st.title("🎯 Kimchi Premium Trading Dashboard")
        st.markdown("---")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Strategy", "Mean Reversion", "48h MA")
        with col2:
            st.metric("Capital", "40,000,000 KRW", "30% Active")
        with col3:
            st.metric("Target Profit", "80,000 KRW", "0.2%")
        with col4:
            status = st.empty()
            status.metric("Status", "🟢 Running", "Live")
    
    def render_realtime_data(self, data):
        """실시간 데이터 표시"""
        if not data:
            st.warning("No data available")
            return
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("📊 Prices")
            st.metric(
                "BTC/KRW (Upbit)",
                f"{data['upbit_price']:,.0f} KRW",
                f"{(data['upbit_price'] - data.get('prev_upbit', data['upbit_price']))/data['upbit_price']*100:.2f}%"
            )
            st.metric(
                "BTC/USDT (Binance)",
                f"{data['binance_price']:,.2f} USDT",
                f"{(data['binance_price'] - data.get('prev_binance', data['binance_price']))/data['binance_price']*100:.2f}%"
            )
        
        with col2:
            st.subheader("💱 Exchange Rate")
            st.metric(
                "USD/KRW",
                f"{data['usd_krw']:,.2f}",
                "Fixed"
            )
            spread = ((data['upbit_ask'] - data['upbit_bid']) / data['upbit_bid'] * 100) if data['upbit_bid'] > 0 else 0
            st.metric(
                "Upbit Spread",
                f"{spread:.3f}%",
                "Good" if spread < 0.1 else "Wide"
            )
        
        with col3:
            st.subheader("🔥 Kimchi Premium")
            kimchi_color = "🟢" if data['kimchi_premium'] < 0 else "🔴"
            st.metric(
                "Current Premium",
                f"{kimchi_color} {data['kimchi_premium']:.3f}%",
                "Entry Zone" if data['kimchi_premium'] < -0.02 else "Wait"
            )
            # MA 계산 (간단 버전)
            if len(self.kimchi_history) > 0:
                ma = sum(self.kimchi_history[-48:]) / min(len(self.kimchi_history), 48)
                st.metric(
                    "48h MA",
                    f"{ma:.3f}%",
                    f"Dev: {data['kimchi_premium'] - ma:.3f}%"
                )
    
    def render_chart(self):
        """차트 렌더링"""
        st.subheader("📈 Kimchi Premium Chart")
        
        if len(self.kimchi_history) < 2:
            st.info("Collecting data... Please wait")
            return
        
        # 차트 생성
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Kimchi Premium vs MA', 'BTC Price'),
            vertical_spacing=0.1,
            row_heights=[0.6, 0.4]
        )
        
        # 시간축
        timestamps = [datetime.now() - timedelta(minutes=len(self.kimchi_history)-i) 
                     for i in range(len(self.kimchi_history))]
        
        # 김프 차트
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=self.kimchi_history,
                mode='lines',
                name='Kimchi Premium',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # MA 라인
        if len(self.ma_history) > 0:
            fig.add_trace(
                go.Scatter(
                    x=timestamps[-len(self.ma_history):],
                    y=self.ma_history,
                    mode='lines',
                    name='48h MA',
                    line=dict(color='orange', width=1, dash='dash')
                ),
                row=1, col=1
            )
        
        # Entry threshold
        fig.add_hline(
            y=-0.02, 
            line_dash="dot", 
            line_color="green",
            annotation_text="Entry Threshold",
            row=1, col=1
        )
        
        # 가격 차트
        if len(self.price_history) > 0:
            fig.add_trace(
                go.Scatter(
                    x=timestamps[-len(self.price_history):],
                    y=self.price_history,
                    mode='lines',
                    name='BTC/KRW',
                    line=dict(color='purple', width=1)
                ),
                row=2, col=1
            )
        
        # 레이아웃 업데이트
        fig.update_layout(
            height=600,
            showlegend=True,
            template='plotly_white',
            hovermode='x unified'
        )
        
        fig.update_xaxes(title_text="Time", row=2, col=1)
        fig.update_yaxes(title_text="Premium (%)", row=1, col=1)
        fig.update_yaxes(title_text="Price (KRW)", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_positions(self):
        """포지션 표시"""
        st.subheader("📦 Active Positions")
        
        # Paper Trading 결과 로드
        report = self.load_paper_trading_results()
        
        if report and 'closed_positions' in report:
            positions_df = pd.DataFrame(report['closed_positions'])
            if not positions_df.empty:
                st.dataframe(
                    positions_df[['id', 'symbol', 'entry_price', 'exit_price', 'pnl', 'pnl_percent']],
                    use_container_width=True
                )
            else:
                st.info("No closed positions yet")
        else:
            st.info("No positions data available")
    
    def render_performance(self):
        """성과 표시"""
        st.subheader("📊 Performance Metrics")
        
        report = self.load_paper_trading_results()
        
        if report:
            col1, col2, col3, col4 = st.columns(4)
            
            stats = report.get('statistics', {})
            perf = report.get('performance', {})
            
            with col1:
                st.metric("Total Trades", perf.get('total_trades', 0))
            with col2:
                win_rate = perf.get('win_rate', 0) * 100
                st.metric("Win Rate", f"{win_rate:.1f}%")
            with col3:
                st.metric("Total PnL", f"{stats.get('total_pnl', 0):,.0f} KRW")
            with col4:
                st.metric("Total Fees", f"{stats.get('total_fees', 0):,.0f} KRW")
        else:
            st.info("Performance data will appear after trades are executed")
    
    async def run(self):
        """메인 실행"""
        await self.initialize()
        
        # 헤더
        self.render_header()
        
        # 메인 컨테이너
        data_container = st.container()
        chart_container = st.container()
        
        # 사이드바
        with st.sidebar:
            st.header("⚙️ Settings")
            
            refresh_rate = st.slider(
                "Refresh Rate (seconds)",
                min_value=1,
                max_value=60,
                value=10
            )
            
            st.markdown("---")
            
            st.header("📋 Strategy Config")
            st.text("Lookback: 48 hours")
            st.text("Entry: MA - 0.02%")
            st.text("Target: 0.2%")
            st.text("Stop Loss: -0.1%")
            st.text("Order Type: Maker Only")
            
            st.markdown("---")
            
            if st.button("🔄 Clear Data"):
                self.kimchi_history.clear()
                self.price_history.clear()
                self.ma_history.clear()
                st.success("Data cleared")
        
        # 실시간 업데이트 루프
        placeholder = st.empty()
        
        while True:
            with placeholder.container():
                # 데이터 수집
                data = await self.fetch_market_data()
                
                if data:
                    # 히스토리 업데이트
                    self.kimchi_history.append(data['kimchi_premium'])
                    self.price_history.append(data['upbit_price'])
                    
                    # MA 계산
                    if len(self.kimchi_history) >= 10:
                        ma = sum(self.kimchi_history[-48:]) / min(len(self.kimchi_history), 48)
                        self.ma_history.append(ma)
                    
                    # 렌더링
                    with data_container:
                        self.render_realtime_data(data)
                    
                    with chart_container:
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            self.render_chart()
                        
                        with col2:
                            self.render_positions()
                            self.render_performance()
                
                # 대기
                await asyncio.sleep(refresh_rate)


async def main():
    dashboard = Dashboard()
    await dashboard.run()


if __name__ == "__main__":
    # Streamlit 실행
    st.write("Starting dashboard...")
    asyncio.run(main())