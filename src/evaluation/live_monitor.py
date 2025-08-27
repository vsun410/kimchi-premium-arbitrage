"""
실시간 모니터링 시스템
모델 성능과 거래를 실시간으로 추적하고 대시보드에 표시
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import threading
import time
import json
import asyncio
import websocket
from collections import deque
import streamlit as st
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

from src.evaluation.performance_metrics import PerformanceMetrics


@dataclass
class LiveMetrics:
    """실시간 메트릭 데이터"""
    timestamp: datetime
    portfolio_value: float
    position_value: float
    cash_balance: float
    premium_rate: float
    
    # 수익성
    pnl: float
    pnl_percent: float
    cumulative_pnl: float
    
    # 포지션
    position_size: float
    position_side: str  # 'long', 'short', 'neutral'
    leverage: float
    
    # 리스크
    current_drawdown: float
    var_95: float
    exposure: float
    
    # 거래
    last_trade_time: Optional[datetime] = None
    last_trade_action: Optional[str] = None
    last_trade_size: Optional[float] = None
    total_trades_today: int = 0
    
    # 모델 신호
    model_signal: Optional[float] = None
    model_confidence: Optional[float] = None
    model_name: Optional[str] = None


class LiveMonitor:
    """
    실시간 모니터링 시스템
    
    모델 성능과 거래를 추적하고 대시보드에 표시
    """
    
    def __init__(
        self,
        initial_capital: float = 10000000,
        update_interval: int = 1,
        history_size: int = 1000
    ):
        """
        모니터 초기화
        
        Args:
            initial_capital: 초기 자본금
            update_interval: 업데이트 주기 (초)
            history_size: 히스토리 크기
        """
        self.initial_capital = initial_capital
        self.update_interval = update_interval
        self.history_size = history_size
        
        # 메트릭 히스토리
        self.metrics_history = deque(maxlen=history_size)
        self.trade_history = deque(maxlen=100)
        
        # 현재 상태
        self.current_metrics = None
        self.is_running = False
        self.monitor_thread = None
        
        # 콜백
        self.callbacks = []
        
        # 알림 설정
        self.alert_thresholds = {
            'max_drawdown': -10.0,  # %
            'daily_loss': -5.0,     # %
            'position_size': 0.5,    # 50% of capital
            'consecutive_losses': 3
        }
        
        # 성과 추적
        self.performance_tracker = {}
        self.daily_stats = {}
        
    def start(self):
        """모니터링 시작"""
        if not self.is_running:
            self.is_running = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop)
            self.monitor_thread.start()
            print("Live monitoring started")
    
    def stop(self):
        """모니터링 중지"""
        self.is_running = False
        if self.monitor_thread:
            self.monitor_thread.join()
        print("Live monitoring stopped")
    
    def _monitor_loop(self):
        """모니터링 루프"""
        while self.is_running:
            try:
                # 메트릭 업데이트
                self._update_metrics()
                
                # 알림 체크
                self._check_alerts()
                
                # 콜백 실행
                self._run_callbacks()
                
                # 대기
                time.sleep(self.update_interval)
                
            except Exception as e:
                print(f"Monitor error: {e}")
    
    def update_from_trade(self, trade_data: Dict):
        """
        거래 데이터로 업데이트
        
        Args:
            trade_data: 거래 정보
        """
        current_time = datetime.now()
        
        # 거래 기록
        self.trade_history.append({
            'timestamp': current_time,
            'action': trade_data.get('action'),
            'size': trade_data.get('size'),
            'price': trade_data.get('price'),
            'premium_rate': trade_data.get('premium_rate'),
            'model': trade_data.get('model')
        })
        
        # 메트릭 업데이트
        if self.current_metrics:
            self.current_metrics.last_trade_time = current_time
            self.current_metrics.last_trade_action = trade_data.get('action')
            self.current_metrics.last_trade_size = trade_data.get('size')
            self.current_metrics.total_trades_today += 1
    
    def update_portfolio(self, portfolio_data: Dict):
        """
        포트폴리오 데이터 업데이트
        
        Args:
            portfolio_data: 포트폴리오 정보
        """
        current_time = datetime.now()
        
        # PnL 계산
        portfolio_value = portfolio_data.get('total_value', self.initial_capital)
        pnl = portfolio_value - self.initial_capital
        pnl_percent = (pnl / self.initial_capital) * 100
        
        # 드로다운 계산
        if self.metrics_history:
            max_value = max(m.portfolio_value for m in self.metrics_history)
            current_drawdown = ((portfolio_value - max_value) / max_value) * 100
        else:
            current_drawdown = 0
        
        # 메트릭 생성
        metrics = LiveMetrics(
            timestamp=current_time,
            portfolio_value=portfolio_value,
            position_value=portfolio_data.get('position_value', 0),
            cash_balance=portfolio_data.get('cash_balance', self.initial_capital),
            premium_rate=portfolio_data.get('premium_rate', 0),
            pnl=pnl,
            pnl_percent=pnl_percent,
            cumulative_pnl=pnl,
            position_size=portfolio_data.get('position_size', 0),
            position_side=portfolio_data.get('position_side', 'neutral'),
            leverage=portfolio_data.get('leverage', 1.0),
            current_drawdown=current_drawdown,
            var_95=portfolio_data.get('var_95', 0),
            exposure=portfolio_data.get('exposure', 0),
            model_signal=portfolio_data.get('model_signal'),
            model_confidence=portfolio_data.get('model_confidence'),
            model_name=portfolio_data.get('model_name')
        )
        
        # 이전 메트릭 정보 복사
        if self.current_metrics:
            metrics.last_trade_time = self.current_metrics.last_trade_time
            metrics.last_trade_action = self.current_metrics.last_trade_action
            metrics.last_trade_size = self.current_metrics.last_trade_size
            metrics.total_trades_today = self.current_metrics.total_trades_today
        
        # 저장
        self.current_metrics = metrics
        self.metrics_history.append(metrics)
    
    def _update_metrics(self):
        """내부 메트릭 업데이트"""
        if not self.current_metrics:
            return
        
        # 일일 통계 업데이트
        today = datetime.now().date()
        if today not in self.daily_stats:
            self.daily_stats[today] = {
                'start_value': self.current_metrics.portfolio_value,
                'high_value': self.current_metrics.portfolio_value,
                'low_value': self.current_metrics.portfolio_value,
                'trades': 0
            }
        
        stats = self.daily_stats[today]
        stats['high_value'] = max(stats['high_value'], self.current_metrics.portfolio_value)
        stats['low_value'] = min(stats['low_value'], self.current_metrics.portfolio_value)
        stats['trades'] = self.current_metrics.total_trades_today
        
        # 성과 메트릭 계산
        if len(self.metrics_history) > 1:
            values = [m.portfolio_value for m in self.metrics_history]
            metrics = PerformanceMetrics(values)
            
            self.performance_tracker = {
                'sharpe_ratio': metrics.sharpe_ratio(),
                'win_rate': metrics.win_rate(),
                'profit_factor': metrics.profit_factor(),
                'max_drawdown': metrics.max_drawdown()
            }
    
    def _check_alerts(self):
        """알림 체크"""
        if not self.current_metrics:
            return
        
        alerts = []
        
        # 드로다운 체크
        if self.current_metrics.current_drawdown < self.alert_thresholds['max_drawdown']:
            alerts.append(f"⚠️ Max drawdown exceeded: {self.current_metrics.current_drawdown:.2f}%")
        
        # 일일 손실 체크
        today = datetime.now().date()
        if today in self.daily_stats:
            daily_return = ((self.current_metrics.portfolio_value - self.daily_stats[today]['start_value']) 
                          / self.daily_stats[today]['start_value']) * 100
            if daily_return < self.alert_thresholds['daily_loss']:
                alerts.append(f"⚠️ Daily loss limit exceeded: {daily_return:.2f}%")
        
        # 포지션 크기 체크
        position_ratio = abs(self.current_metrics.position_size) / self.initial_capital
        if position_ratio > self.alert_thresholds['position_size']:
            alerts.append(f"⚠️ Position size too large: {position_ratio:.1%}")
        
        # 연속 손실 체크
        consecutive_losses = self._count_consecutive_losses()
        if consecutive_losses >= self.alert_thresholds['consecutive_losses']:
            alerts.append(f"⚠️ Consecutive losses: {consecutive_losses}")
        
        # 알림 발송
        for alert in alerts:
            self._send_alert(alert)
    
    def _count_consecutive_losses(self) -> int:
        """연속 손실 횟수 계산"""
        if not self.trade_history:
            return 0
        
        count = 0
        for trade in reversed(self.trade_history):
            if trade.get('pnl', 0) < 0:
                count += 1
            else:
                break
        
        return count
    
    def _send_alert(self, message: str):
        """알림 발송"""
        print(f"[ALERT] {message}")
        # TODO: 이메일, Slack, Discord 등 알림 발송
    
    def _run_callbacks(self):
        """콜백 실행"""
        for callback in self.callbacks:
            try:
                callback(self.current_metrics)
            except Exception as e:
                print(f"Callback error: {e}")
    
    def add_callback(self, callback: Callable):
        """콜백 추가"""
        self.callbacks.append(callback)
    
    def get_dashboard_data(self) -> Dict:
        """
        대시보드용 데이터 반환
        
        Returns:
            대시보드 데이터
        """
        if not self.metrics_history:
            return {}
        
        # 시계열 데이터
        timestamps = [m.timestamp for m in self.metrics_history]
        portfolio_values = [m.portfolio_value for m in self.metrics_history]
        pnl_values = [m.pnl for m in self.metrics_history]
        drawdowns = [m.current_drawdown for m in self.metrics_history]
        premium_rates = [m.premium_rate for m in self.metrics_history]
        
        # 현재 상태
        current = self.current_metrics
        
        return {
            'timestamps': timestamps,
            'portfolio_values': portfolio_values,
            'pnl_values': pnl_values,
            'drawdowns': drawdowns,
            'premium_rates': premium_rates,
            'current_metrics': current,
            'performance': self.performance_tracker,
            'daily_stats': self.daily_stats,
            'trades': list(self.trade_history)
        }
    
    def create_streamlit_dashboard(self):
        """Streamlit 대시보드 생성"""
        st.title("🚀 Kimchi Premium Arbitrage - Live Monitor")
        
        # 데이터 가져오기
        data = self.get_dashboard_data()
        
        if not data:
            st.warning("No data available yet")
            return
        
        current = data['current_metrics']
        
        # 주요 지표
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Portfolio Value",
                f"${current.portfolio_value:,.0f}",
                f"{current.pnl_percent:.2f}%"
            )
        
        with col2:
            st.metric(
                "PnL",
                f"${current.pnl:,.0f}",
                f"{current.pnl_percent:.2f}%"
            )
        
        with col3:
            st.metric(
                "Drawdown",
                f"{current.current_drawdown:.2f}%",
                delta_color="inverse"
            )
        
        with col4:
            st.metric(
                "Premium Rate",
                f"{current.premium_rate:.2f}%"
            )
        
        # 차트
        st.subheader("📊 Performance Charts")
        
        # Portfolio Value Chart
        fig_portfolio = go.Figure()
        fig_portfolio.add_trace(go.Scatter(
            x=data['timestamps'],
            y=data['portfolio_values'],
            mode='lines',
            name='Portfolio Value',
            line=dict(color='blue', width=2)
        ))
        fig_portfolio.update_layout(
            title="Portfolio Value Over Time",
            xaxis_title="Time",
            yaxis_title="Value ($)",
            height=400
        )
        st.plotly_chart(fig_portfolio, use_container_width=True)
        
        # PnL and Drawdown
        col1, col2 = st.columns(2)
        
        with col1:
            fig_pnl = go.Figure()
            fig_pnl.add_trace(go.Scatter(
                x=data['timestamps'],
                y=data['pnl_values'],
                mode='lines',
                name='PnL',
                line=dict(color='green' if current.pnl >= 0 else 'red')
            ))
            fig_pnl.update_layout(
                title="Profit & Loss",
                xaxis_title="Time",
                yaxis_title="PnL ($)",
                height=300
            )
            st.plotly_chart(fig_pnl, use_container_width=True)
        
        with col2:
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(
                x=data['timestamps'],
                y=data['drawdowns'],
                mode='lines',
                name='Drawdown',
                line=dict(color='red'),
                fill='tozeroy'
            ))
            fig_dd.update_layout(
                title="Drawdown",
                xaxis_title="Time",
                yaxis_title="Drawdown (%)",
                height=300
            )
            st.plotly_chart(fig_dd, use_container_width=True)
        
        # 성과 메트릭
        st.subheader("📈 Performance Metrics")
        
        perf = data['performance']
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Sharpe Ratio", f"{perf.get('sharpe_ratio', 0):.3f}")
        
        with col2:
            st.metric("Win Rate", f"{perf.get('win_rate', 0):.1f}%")
        
        with col3:
            st.metric("Profit Factor", f"{perf.get('profit_factor', 0):.2f}")
        
        with col4:
            st.metric("Max Drawdown", f"{perf.get('max_drawdown', 0):.2f}%")
        
        # 최근 거래
        st.subheader("🔄 Recent Trades")
        
        if data['trades']:
            trades_df = pd.DataFrame(data['trades'][-10:])
            st.dataframe(trades_df, use_container_width=True)
        else:
            st.info("No trades executed yet")
        
        # 포지션 정보
        st.subheader("📍 Current Position")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Position Size", f"${current.position_size:,.0f}")
        
        with col2:
            st.metric("Position Side", current.position_side.upper())
        
        with col3:
            st.metric("Leverage", f"{current.leverage:.1f}x")
        
        # 모델 신호
        if current.model_signal is not None:
            st.subheader("🤖 Model Signal")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Model", current.model_name or "Unknown")
            
            with col2:
                signal_str = "BUY" if current.model_signal > 0 else "SELL" if current.model_signal < 0 else "HOLD"
                st.metric("Signal", signal_str)
            
            with col3:
                if current.model_confidence:
                    st.metric("Confidence", f"{current.model_confidence:.1%}")
        
        # 자동 새로고침
        st.button("🔄 Refresh")
        
        # 메타데이터
        st.caption(f"Last updated: {current.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")