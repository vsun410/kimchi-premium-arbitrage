"""
Backtesting Engine
Phase 3: 김치 프리미엄 백테스팅 엔진
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from src.utils.logger import logger
from src.utils.exchange_rate_manager import get_exchange_rate_manager


@dataclass
class Position:
    """포지션 정보"""
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_price_upbit: float
    entry_price_binance: float
    exit_price_upbit: Optional[float] = None
    exit_price_binance: Optional[float] = None
    size: float = 0.01  # BTC
    is_open: bool = True
    pnl: float = 0.0
    kimchi_premium_entry: float = 0.0
    kimchi_premium_exit: float = 0.0


@dataclass
class TradingCosts:
    """거래 비용 설정"""
    upbit_fee: float = 0.0005  # 0.05%
    binance_fee: float = 0.001  # 0.1%
    slippage: float = 0.0001  # 0.01%
    funding_rate: float = 0.0001  # 0.01% per 8h
    

class BacktestEngine:
    """
    백테스팅 엔진
    
    Features:
    - 실제 거래 비용 시뮬레이션
    - 슬리피지 계산
    - 자금 관리
    - 포지션 추적
    """
    
    def __init__(
        self,
        initial_capital: float = 40_000_000,  # 4천만원
        max_position_size: float = 0.1,  # 최대 0.1 BTC
        costs: Optional[TradingCosts] = None
    ):
        """
        초기화
        
        Args:
            initial_capital: 초기 자본금 (KRW)
            max_position_size: 최대 포지션 크기 (BTC)
            costs: 거래 비용 설정
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_position_size = max_position_size
        self.costs = costs or TradingCosts()
        
        # 포지션 관리
        self.positions: List[Position] = []
        self.current_position: Optional[Position] = None
        
        # 성과 기록
        self.equity_curve = []
        self.trades = []
        
        # 환율 관리자 초기화
        self.rate_manager = get_exchange_rate_manager()
        
        logger.info(f"Backtest engine initialized with {initial_capital:,} KRW")
    
    def calculate_kimchi_premium(
        self,
        upbit_price: float,
        binance_price: float,
        timestamp: Optional[datetime] = None
    ) -> float:
        """
        김치 프리미엄 계산 (실제 환율 사용)
        
        Args:
            upbit_price: 업비트 가격 (KRW)
            binance_price: 바이낸스 가격 (USD)
            timestamp: 시간 (백테스트 시점)
            
        Returns:
            김치 프리미엄 (%)
        """
        # 실제 환율 사용!
        return self.rate_manager.calculate_kimchi_premium(
            upbit_price, binance_price, timestamp
        )
    
    def calculate_position_size(
        self,
        confidence: float,
        current_premium: float
    ) -> float:
        """
        포지션 크기 계산 (Kelly Criterion 기반)
        
        Args:
            confidence: 신호 신뢰도 (0-1)
            current_premium: 현재 김프 (%)
            
        Returns:
            포지션 크기 (BTC)
        """
        # 기본 크기
        base_size = 0.01  # 0.01 BTC
        
        # 신뢰도 기반 조정
        size_multiplier = 1 + (confidence - 0.5) * 2  # 0.5-1.0 → 0-2
        
        # 김프 기반 조정
        if abs(current_premium) > 5:  # 5% 이상이면 크게
            size_multiplier *= 1.5
        elif abs(current_premium) < 2:  # 2% 이하면 작게
            size_multiplier *= 0.5
        
        position_size = base_size * size_multiplier
        
        # 최대 크기 제한
        position_size = min(position_size, self.max_position_size)
        
        # 자본 대비 제한 (리스크 관리)
        max_by_capital = (self.current_capital * 0.1) / (100_000_000)  # 자본의 10%
        position_size = min(position_size, max_by_capital)
        
        return position_size
    
    def open_position(
        self,
        timestamp: datetime,
        upbit_price: float,
        binance_price: float,
        kimchi_premium: float,
        confidence: float = 0.5
    ) -> Position:
        """
        포지션 진입
        
        Args:
            timestamp: 진입 시간
            upbit_price: 업비트 가격
            binance_price: 바이낸스 가격
            kimchi_premium: 김치 프리미엄
            confidence: 신호 신뢰도
            
        Returns:
            생성된 포지션
        """
        # 포지션 크기 계산
        size = self.calculate_position_size(confidence, kimchi_premium)
        
        # 거래 비용 계산 (실제 환율 사용)
        exchange_rate = self.rate_manager.get_rate_at_time(timestamp) if timestamp else self.rate_manager.current_rate
        upbit_cost = upbit_price * size * self.costs.upbit_fee
        binance_cost = binance_price * size * self.costs.binance_fee * exchange_rate  # 실제 환율
        slippage_cost = (upbit_price + binance_price * exchange_rate) * size * self.costs.slippage
        
        total_cost = upbit_cost + binance_cost + slippage_cost
        
        # 포지션 생성
        position = Position(
            entry_time=timestamp,
            exit_time=None,
            entry_price_upbit=upbit_price,
            entry_price_binance=binance_price,
            size=size,
            is_open=True,
            kimchi_premium_entry=kimchi_premium
        )
        
        # 자본 차감
        self.current_capital -= total_cost
        
        self.current_position = position
        self.positions.append(position)
        
        logger.debug(f"Opened position at {timestamp}: size={size:.4f} BTC, premium={kimchi_premium:.2f}%")
        
        return position
    
    def close_position(
        self,
        timestamp: datetime,
        upbit_price: float,
        binance_price: float,
        kimchi_premium: float
    ) -> float:
        """
        포지션 청산
        
        Args:
            timestamp: 청산 시간
            upbit_price: 업비트 가격
            binance_price: 바이낸스 가격
            kimchi_premium: 김치 프리미엄
            
        Returns:
            실현 손익 (KRW)
        """
        if not self.current_position or not self.current_position.is_open:
            return 0.0
        
        position = self.current_position
        
        # 거래 비용 계산 (실제 환율 사용)
        exchange_rate = self.rate_manager.get_rate_at_time(timestamp) if timestamp else self.rate_manager.current_rate
        upbit_cost = upbit_price * position.size * self.costs.upbit_fee
        binance_cost = binance_price * position.size * self.costs.binance_fee * exchange_rate
        slippage_cost = (upbit_price + binance_price * exchange_rate) * position.size * self.costs.slippage
        
        # Funding 비용 (시간 기반)
        hours_held = (timestamp - position.entry_time).total_seconds() / 3600
        funding_periods = hours_held / 8  # 8시간마다
        funding_cost = binance_price * position.size * self.costs.funding_rate * funding_periods * exchange_rate
        
        total_cost = upbit_cost + binance_cost + slippage_cost + funding_cost
        
        # PnL 계산
        # 업비트: 매도 - 매수 (현물)
        upbit_pnl = (upbit_price - position.entry_price_upbit) * position.size
        
        # 바이낸스: 매수 - 매도 (선물 숏) - 진입 시점과 청산 시점 환율 사용
        entry_rate = self.rate_manager.get_rate_at_time(position.entry_time)
        exit_rate = exchange_rate
        binance_pnl = (position.entry_price_binance * entry_rate - binance_price * exit_rate) * position.size
        
        gross_pnl = upbit_pnl + binance_pnl
        net_pnl = gross_pnl - total_cost
        
        # 포지션 업데이트
        position.exit_time = timestamp
        position.exit_price_upbit = upbit_price
        position.exit_price_binance = binance_price
        position.is_open = False
        position.pnl = net_pnl
        position.kimchi_premium_exit = kimchi_premium
        
        # 자본 업데이트
        self.current_capital += gross_pnl - total_cost
        
        self.current_position = None
        self.trades.append({
            'entry_time': position.entry_time,
            'exit_time': timestamp,
            'pnl': net_pnl,
            'return': net_pnl / self.initial_capital * 100,
            'premium_entry': position.kimchi_premium_entry,
            'premium_exit': kimchi_premium
        })
        
        logger.debug(f"Closed position at {timestamp}: PnL={net_pnl:,.0f} KRW")
        
        return net_pnl
    
    def update_equity(self, timestamp: datetime):
        """
        자산 가치 업데이트
        
        Args:
            timestamp: 현재 시간
        """
        self.equity_curve.append({
            'timestamp': timestamp,
            'equity': self.current_capital,
            'return': (self.current_capital - self.initial_capital) / self.initial_capital * 100
        })
    
    def run(
        self,
        data: pd.DataFrame,
        strategy,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict:
        """
        백테스트 실행
        
        Args:
            data: 가격 데이터 (index=timestamp)
            strategy: 거래 전략 객체
            start_date: 시작일
            end_date: 종료일
            
        Returns:
            백테스트 결과
        """
        logger.info("Starting backtest...")
        
        # 날짜 필터링
        if start_date:
            data = data[data.index >= start_date]
        if end_date:
            data = data[data.index <= end_date]
        
        # 백테스트 루프
        for timestamp, row in data.iterrows():
            # 김프 계산 (시간 정보 포함)
            kimchi_premium = self.calculate_kimchi_premium(
                row['upbit_close'],
                row['binance_close'],
                timestamp=timestamp
            )
            
            # 전략 신호
            signal = strategy.generate_signal(
                timestamp=timestamp,
                kimchi_premium=kimchi_premium,
                row=row
            )
            
            # 포지션 관리
            if signal['action'] == 'ENTER' and not self.current_position:
                self.open_position(
                    timestamp=timestamp,
                    upbit_price=row['upbit_close'],
                    binance_price=row['binance_close'],
                    kimchi_premium=kimchi_premium,
                    confidence=signal.get('confidence', 0.5)
                )
            
            elif signal['action'] == 'EXIT' and self.current_position:
                self.close_position(
                    timestamp=timestamp,
                    upbit_price=row['upbit_close'],
                    binance_price=row['binance_close'],
                    kimchi_premium=kimchi_premium
                )
            
            # 자산 가치 업데이트
            self.update_equity(timestamp)
        
        # 마지막 포지션 청산
        if self.current_position and self.current_position.is_open:
            last_row = data.iloc[-1]
            self.close_position(
                timestamp=data.index[-1],
                upbit_price=last_row['upbit_close'],
                binance_price=last_row['binance_close'],
                kimchi_premium=self.calculate_kimchi_premium(
                    last_row['upbit_close'],
                    last_row['binance_close'],
                    timestamp=data.index[-1]
                )
            )
        
        logger.info(f"Backtest complete: {len(self.trades)} trades executed")
        
        return self.get_results()
    
    def get_results(self) -> Dict:
        """
        백테스트 결과 반환
        
        Returns:
            성과 지표
        """
        if not self.trades:
            return {
                'total_trades': 0,
                'total_return': 0,
                'win_rate': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0
            }
        
        trades_df = pd.DataFrame(self.trades)
        equity_df = pd.DataFrame(self.equity_curve)
        
        # 기본 통계
        total_return = (self.current_capital - self.initial_capital) / self.initial_capital * 100
        win_rate = (trades_df['pnl'] > 0).mean() * 100
        
        # Sharpe Ratio
        if len(equity_df) > 1:
            returns = equity_df['return'].diff().dropna()
            sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        else:
            sharpe = 0
        
        # Max Drawdown
        equity_df['cummax'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['cummax']) / equity_df['cummax'] * 100
        max_drawdown = equity_df['drawdown'].min()
        
        return {
            'total_trades': len(self.trades),
            'total_return': total_return,
            'final_capital': self.current_capital,
            'win_rate': win_rate,
            'avg_trade_return': trades_df['return'].mean() if len(trades_df) > 0 else 0,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'best_trade': trades_df['pnl'].max() if len(trades_df) > 0 else 0,
            'worst_trade': trades_df['pnl'].min() if len(trades_df) > 0 else 0,
            'trades': self.trades,
            'equity_curve': self.equity_curve
        }