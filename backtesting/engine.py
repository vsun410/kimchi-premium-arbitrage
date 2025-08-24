"""
백테스팅 엔진 핵심 구현
Walk-forward analysis와 실제 거래 비용을 반영한 정확한 시뮬레이션
"""

import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
import json
import logging

# 프로젝트 루트 경로 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from strategies.base_strategy import BaseStrategy
from src.utils.exchange_rate_manager import ExchangeRateManager

# 로거 설정
logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """거래 기록"""
    timestamp: datetime
    side: str  # 'buy' or 'sell'
    price: float
    amount: float
    fee: float
    slippage: float
    pnl: float = 0.0
    cumulative_pnl: float = 0.0


@dataclass
class BacktestResult:
    """백테스트 결과"""
    # 기본 메트릭
    total_return: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    
    # 수익 메트릭
    average_win: float = 0.0
    average_loss: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    
    # 리스크 메트릭
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    
    # 거래 내역
    trades: List[Trade] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=pd.Series)
    
    # 추가 정보
    start_date: datetime = None
    end_date: datetime = None
    initial_capital: float = 0.0
    final_capital: float = 0.0


class BacktestingEngine:
    """
    백테스팅 엔진
    과거 데이터로 전략을 검증하고 성능을 분석
    """
    
    def __init__(
        self,
        strategy: BaseStrategy,
        initial_capital: float = 40_000_000,  # 4천만원
        maker_fee: float = 0.0002,  # 0.02% 메이커 수수료
        taker_fee: float = 0.0015,  # 0.15% 테이커 수수료
        slippage_bps: float = 10,  # 10 bps (0.1%) 슬리피지
        use_maker_only: bool = True  # 메이커 주문만 사용
    ):
        """
        백테스팅 엔진 초기화
        
        Args:
            strategy: 테스트할 전략
            initial_capital: 초기 자본금
            maker_fee: 메이커 수수료율
            taker_fee: 테이커 수수료율  
            slippage_bps: 슬리피지 (basis points)
            use_maker_only: 메이커 주문만 사용 여부
        """
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.slippage_rate = slippage_bps / 10000  # bps to rate
        self.use_maker_only = use_maker_only
        
        # 거래 상태
        self.capital = initial_capital
        self.position = 0.0  # 현재 포지션
        self.trades: List[Trade] = []
        self.equity_curve = []
        
        # 환율 관리자
        self.exchange_rate_manager = ExchangeRateManager()
        
        logger.info(f"BacktestingEngine initialized with capital: {initial_capital:,.0f} KRW")
    
    def run(
        self,
        data: pd.DataFrame,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        walk_forward_window: Optional[int] = None
    ) -> BacktestResult:
        """
        백테스트 실행
        
        Args:
            data: 과거 가격 데이터 (OHLCV + 김프 등)
            start_date: 백테스트 시작일
            end_date: 백테스트 종료일
            walk_forward_window: Walk-forward 윈도우 크기 (일)
            
        Returns:
            BacktestResult: 백테스트 결과
        """
        # 데이터 필터링
        if start_date:
            data = data[data.index >= start_date]
        if end_date:
            data = data[data.index <= end_date]
            
        if len(data) < 100:
            raise ValueError("백테스트를 위한 데이터가 부족합니다 (최소 100개 필요)")
        
        logger.info(f"Running backtest from {data.index[0]} to {data.index[-1]}")
        
        # 초기화
        self.capital = self.initial_capital
        self.position = 0.0
        self.trades = []
        self.equity_curve = []
        
        # Walk-forward analysis 적용
        if walk_forward_window:
            results = self._run_walk_forward(data, walk_forward_window)
        else:
            results = self._run_simple_backtest(data)
            
        return results
    
    def _run_simple_backtest(self, data: pd.DataFrame) -> BacktestResult:
        """
        단순 백테스트 실행
        
        Args:
            data: 과거 데이터
            
        Returns:
            BacktestResult: 백테스트 결과
        """
        for i in range(len(data)):
            current_data = data.iloc[:i+1]
            current_price = data.iloc[i]['close']
            current_time = data.index[i]
            
            # 전략 신호 생성
            signal = self.strategy.generate_signal(current_data)
            
            if signal:
                self._execute_trade(signal, current_price, current_time)
            
            # 자산 가치 기록
            equity = self._calculate_equity(current_price)
            self.equity_curve.append({
                'timestamp': current_time,
                'equity': equity,
                'capital': self.capital,
                'position_value': self.position * current_price
            })
        
        # 결과 계산
        return self._calculate_results()
    
    def _run_walk_forward(self, data: pd.DataFrame, window_days: int) -> BacktestResult:
        """
        Walk-forward analysis 실행
        
        Args:
            data: 과거 데이터
            window_days: 학습 윈도우 크기 (일)
            
        Returns:
            BacktestResult: 백테스트 결과
        """
        window_size = window_days * 24  # 시간 단위로 변환 (1시간 봉 기준)
        test_size = window_size // 4  # 테스트 기간은 학습 기간의 1/4
        
        all_trades = []
        all_equity = []
        
        for start_idx in range(0, len(data) - window_size - test_size, test_size):
            # 학습 데이터
            train_data = data.iloc[start_idx:start_idx + window_size]
            
            # 테스트 데이터
            test_start = start_idx + window_size
            test_end = min(test_start + test_size, len(data))
            test_data = data.iloc[test_start:test_end]
            
            # 전략 재학습 (필요한 경우)
            if hasattr(self.strategy, 'train'):
                self.strategy.train(train_data)
            
            # 테스트 기간 백테스트
            for i in range(len(test_data)):
                current_data = data.iloc[:test_start + i + 1]
                current_price = test_data.iloc[i]['close']
                current_time = test_data.index[i]
                
                signal = self.strategy.generate_signal(current_data)
                
                if signal:
                    self._execute_trade(signal, current_price, current_time)
                
                equity = self._calculate_equity(current_price)
                self.equity_curve.append({
                    'timestamp': current_time,
                    'equity': equity,
                    'capital': self.capital,
                    'position_value': self.position * current_price
                })
        
        return self._calculate_results()
    
    def _execute_trade(self, signal: Dict, price: float, timestamp: datetime):
        """
        거래 실행 시뮬레이션
        
        Args:
            signal: 거래 신호
            price: 현재 가격
            timestamp: 거래 시간
        """
        side = signal.get('side')
        amount = signal.get('amount', 0.1)  # 기본 0.1 BTC
        
        # 슬리피지 적용
        if side == 'buy':
            execution_price = price * (1 + self.slippage_rate)
        else:
            execution_price = price * (1 - self.slippage_rate)
        
        # 수수료 계산
        if self.use_maker_only:
            fee_rate = self.maker_fee
        else:
            fee_rate = self.taker_fee
            
        trade_value = execution_price * amount
        fee = trade_value * fee_rate
        
        # 자본금 확인
        total_cost = trade_value + fee
        if side == 'buy' and total_cost > self.capital:
            logger.warning(f"Insufficient capital for trade: {total_cost:,.0f} > {self.capital:,.0f}")
            return
        
        # 거래 실행
        if side == 'buy':
            self.capital -= total_cost
            self.position += amount
        else:
            if self.position < amount:
                logger.warning(f"Insufficient position for sell: {amount} > {self.position}")
                return
            self.capital += trade_value - fee
            self.position -= amount
        
        # 거래 기록
        trade = Trade(
            timestamp=timestamp,
            side=side,
            price=execution_price,
            amount=amount,
            fee=fee,
            slippage=(execution_price - price) * amount,
            pnl=0.0  # 나중에 계산
        )
        
        self.trades.append(trade)
        
        logger.debug(f"Trade executed: {side} {amount} @ {execution_price:,.0f} (fee: {fee:,.0f})")
    
    def _calculate_equity(self, current_price: float) -> float:
        """
        현재 자산 가치 계산
        
        Args:
            current_price: 현재 가격
            
        Returns:
            float: 총 자산 가치
        """
        return self.capital + (self.position * current_price)
    
    def _calculate_results(self) -> BacktestResult:
        """
        백테스트 결과 계산
        
        Returns:
            BacktestResult: 계산된 결과
        """
        if not self.equity_curve:
            return BacktestResult(initial_capital=self.initial_capital)
        
        # Equity curve를 DataFrame으로 변환
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df.set_index('timestamp', inplace=True)
        
        # 수익률 계산
        final_equity = equity_df['equity'].iloc[-1]
        total_return = (final_equity - self.initial_capital) / self.initial_capital
        
        # 거래 통계
        if self.trades:
            # PnL 계산
            buy_trades = [t for t in self.trades if t.side == 'buy']
            sell_trades = [t for t in self.trades if t.side == 'sell']
            
            # 매칭된 거래들의 PnL 계산
            pnls = []
            for i in range(min(len(buy_trades), len(sell_trades))):
                buy = buy_trades[i]
                sell = sell_trades[i]
                pnl = (sell.price - buy.price) * buy.amount - buy.fee - sell.fee
                pnls.append(pnl)
            
            winning_trades = [p for p in pnls if p > 0]
            losing_trades = [p for p in pnls if p <= 0]
            
            win_rate = len(winning_trades) / len(pnls) if pnls else 0
            avg_win = np.mean(winning_trades) if winning_trades else 0
            avg_loss = np.mean(losing_trades) if losing_trades else 0
            
            # Profit factor
            total_wins = sum(winning_trades) if winning_trades else 0
            total_losses = abs(sum(losing_trades)) if losing_trades else 1
            profit_factor = total_wins / total_losses if total_losses > 0 else 0
        else:
            win_rate = avg_win = avg_loss = profit_factor = 0
            winning_trades = []
            losing_trades = []
        
        # Drawdown 계산
        equity_series = equity_df['equity']
        cummax = equity_series.cummax()
        drawdown = (equity_series - cummax) / cummax
        max_drawdown = drawdown.min()
        
        # Drawdown duration
        drawdown_start = None
        max_duration = 0
        current_duration = 0
        
        for i, dd in enumerate(drawdown):
            if dd < 0:
                if drawdown_start is None:
                    drawdown_start = i
                current_duration = i - drawdown_start
            else:
                if current_duration > max_duration:
                    max_duration = current_duration
                drawdown_start = None
                current_duration = 0
        
        # 리스크 메트릭 계산
        returns = equity_series.pct_change().dropna()
        
        # Sharpe Ratio (연율화)
        if returns.std() > 0:
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(365 * 24)
        else:
            sharpe_ratio = 0
        
        # Sortino Ratio
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            sortino_ratio = (returns.mean() / downside_returns.std()) * np.sqrt(365 * 24)
        else:
            sortino_ratio = 0
        
        # Calmar Ratio
        if max_drawdown < 0:
            calmar_ratio = total_return / abs(max_drawdown)
        else:
            calmar_ratio = 0
        
        # 결과 생성
        result = BacktestResult(
            total_return=total_return,
            total_trades=len(self.trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            win_rate=win_rate,
            average_win=avg_win,
            average_loss=avg_loss,
            profit_factor=profit_factor,
            max_drawdown=max_drawdown,
            max_drawdown_duration=max_duration,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            trades=self.trades,
            equity_curve=equity_series,
            start_date=equity_df.index[0],
            end_date=equity_df.index[-1],
            initial_capital=self.initial_capital,
            final_capital=final_equity
        )
        
        return result
    
    def generate_report(self, result: BacktestResult, output_path: str = None):
        """
        백테스트 리포트 생성
        
        Args:
            result: 백테스트 결과
            output_path: 리포트 저장 경로
        """
        report = {
            "백테스트 요약": {
                "기간": f"{result.start_date} ~ {result.end_date}",
                "초기 자본": f"{result.initial_capital:,.0f} KRW",
                "최종 자본": f"{result.final_capital:,.0f} KRW",
                "총 수익률": f"{result.total_return:.2%}",
            },
            "거래 통계": {
                "총 거래 횟수": result.total_trades,
                "승리 거래": result.winning_trades,
                "패배 거래": result.losing_trades,
                "승률": f"{result.win_rate:.2%}",
                "평균 수익": f"{result.average_win:,.0f} KRW",
                "평균 손실": f"{result.average_loss:,.0f} KRW",
                "Profit Factor": f"{result.profit_factor:.2f}",
            },
            "리스크 메트릭": {
                "최대 낙폭": f"{result.max_drawdown:.2%}",
                "최대 낙폭 기간": f"{result.max_drawdown_duration} 시간",
                "Sharpe Ratio": f"{result.sharpe_ratio:.2f}",
                "Sortino Ratio": f"{result.sortino_ratio:.2f}",
                "Calmar Ratio": f"{result.calmar_ratio:.2f}",
            },
            "수수료 설정": {
                "메이커 수수료": f"{self.maker_fee:.2%}",
                "테이커 수수료": f"{self.taker_fee:.2%}",
                "슬리피지": f"{self.slippage_rate:.2%}",
                "메이커 전용": self.use_maker_only,
            }
        }
        
        # 콘솔 출력
        print("\n" + "=" * 60)
        print("백테스트 리포트")
        print("=" * 60)
        
        for section, metrics in report.items():
            print(f"\n[{section}]")
            for key, value in metrics.items():
                print(f"  {key}: {value}")
        
        # 파일 저장
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # JSON 리포트
            json_path = output_path.replace('.txt', '.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2, default=str)
            
            # 텍스트 리포트
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("백테스트 리포트\n")
                f.write("=" * 60 + "\n")
                for section, metrics in report.items():
                    f.write(f"\n[{section}]\n")
                    for key, value in metrics.items():
                        f.write(f"  {key}: {value}\n")
            
            print(f"\n리포트 저장: {output_path}")
            print(f"JSON 리포트: {json_path}")
        
        return report