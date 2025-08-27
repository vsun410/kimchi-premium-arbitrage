"""
멀티 전략 시스템 데모
여러 전략을 동시에 실행하고 성과를 비교하는 데모
"""

import asyncio
import random
from datetime import datetime, timedelta
import time
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich.live import Live
from rich.layout import Layout

from strategies.multi_strategy import (
    MarketData,
    ThresholdStrategy,
    MovingAverageStrategy,
    BollingerBandsStrategy,
    StrategyManager,
    AllocationMethod,
    SignalAggregation
)

console = Console()


def generate_market_data(base_premium: float = 2.0) -> MarketData:
    """
    시뮬레이션용 시장 데이터 생성
    
    Args:
        base_premium: 기본 김치 프리미엄
        
    Returns:
        시장 데이터
    """
    # 랜덤 변동 추가
    variation = random.uniform(-0.5, 0.5)
    kimchi_premium = base_premium + variation
    
    # 가격 설정
    binance_price = 70000 + random.uniform(-500, 500)
    exchange_rate = 1400 + random.uniform(-10, 10)
    upbit_price = binance_price * exchange_rate * (1 + kimchi_premium / 100)
    
    return MarketData(
        timestamp=datetime.now(),
        upbit_price=upbit_price,
        binance_price=binance_price,
        exchange_rate=exchange_rate,
        kimchi_premium=kimchi_premium,
        volume_upbit=random.uniform(50, 200),
        volume_binance=random.uniform(100, 300),
        bid_ask_spread_upbit=random.uniform(0.05, 0.15),
        bid_ask_spread_binance=random.uniform(0.02, 0.08)
    )


def create_strategies_table(manager: StrategyManager) -> Table:
    """
    전략 상태 테이블 생성
    
    Args:
        manager: 전략 매니저
        
    Returns:
        Rich Table
    """
    table = Table(title="🎯 전략별 상태", show_header=True, header_style="bold magenta")
    table.add_column("전략", style="cyan", no_wrap=True)
    table.add_column("상태", style="green")
    table.add_column("포지션", justify="right")
    table.add_column("자본금", justify="right")
    table.add_column("PnL", justify="right")
    table.add_column("승률", justify="right")
    table.add_column("신호", justify="center")
    
    for status in manager.get_all_strategies_status():
        # 색상 설정
        pnl = status['total_pnl']
        pnl_color = "green" if pnl >= 0 else "red"
        
        table.add_row(
            status['strategy_name'],
            status['status'],
            f"{status['current_position']:.4f}" if status['current_position'] != 0 else "-",
            f"{status['allocated_capital']:,.0f}",
            f"[{pnl_color}]{pnl:+,.0f}[/{pnl_color}]",
            status['win_rate'],
            status.get('weight', '-')
        )
    
    return table


def create_portfolio_panel(manager: StrategyManager) -> Panel:
    """
    포트폴리오 상태 패널 생성
    
    Args:
        manager: 전략 매니저
        
    Returns:
        Rich Panel
    """
    portfolio = manager.get_portfolio_status()
    
    content = f"""
💰 총 자본: {portfolio['total_capital']:,.0f} KRW
📊 할당 자본: {portfolio['allocated_capital']:,.0f} KRW
💵 여유 자본: {portfolio['free_capital']:,.0f} KRW
📈 총 포지션: {portfolio['total_positions']}개
💹 총 수익: {portfolio['total_pnl']:+,.0f} KRW ({portfolio['total_pnl_pct']})
🎯 활성 전략: {portfolio['active_strategies']}개
"""
    
    return Panel(content, title="📊 포트폴리오 현황", border_style="blue")


def create_market_panel(market_data: MarketData) -> Panel:
    """
    시장 데이터 패널 생성
    
    Args:
        market_data: 시장 데이터
        
    Returns:
        Rich Panel
    """
    premium_color = "green" if market_data.kimchi_premium > 2 else "yellow" if market_data.kimchi_premium > 0 else "red"
    
    content = f"""
🇰🇷 업비트: {market_data.upbit_price:,.0f} KRW
🌍 바이낸스: {market_data.binance_price:,.2f} USDT
💱 환율: {market_data.exchange_rate:,.2f} KRW/USD
[{premium_color}]🔥 김프: {market_data.kimchi_premium:+.2f}%[/{premium_color}]
📊 거래량(업비트): {market_data.volume_upbit:.2f} BTC
📊 거래량(바이낸스): {market_data.volume_binance:.2f} BTC
"""
    
    return Panel(content, title="📈 시장 현황", border_style="green")


async def run_simulation():
    """
    시뮬레이션 실행
    """
    console.print(Panel.fit("🚀 멀티 전략 시스템 데모 시작", style="bold green"))
    
    # 전략 매니저 생성
    manager = StrategyManager(
        initial_capital=10_000_000,
        config={
            'allocation_method': AllocationMethod.EQUAL,
            'signal_aggregation': SignalAggregation.WEIGHTED,
            'max_concurrent_positions': 3
        }
    )
    
    # 전략 추가
    strategies = [
        ThresholdStrategy(
            name="Threshold-Conservative",
            config={'entry_threshold': 3.5, 'exit_threshold': 2.0}
        ),
        ThresholdStrategy(
            name="Threshold-Aggressive",
            config={'entry_threshold': 2.5, 'exit_threshold': 1.0}
        ),
        MovingAverageStrategy(
            name="MA-Fast",
            config={'short_window': 5, 'long_window': 15}
        ),
        MovingAverageStrategy(
            name="MA-Slow",
            config={'short_window': 10, 'long_window': 30}
        ),
        BollingerBandsStrategy(
            name="BB-Standard",
            config={'bb_period': 20, 'bb_std': 2.0}
        )
    ]
    
    for strategy in strategies:
        manager.add_strategy(strategy)
    
    console.print(f"✅ {len(strategies)}개 전략 추가 완료\n")
    
    # 시뮬레이션 설정
    simulation_rounds = 100
    base_premium = 2.5
    premium_trend = 0
    
    # Layout 설정
    layout = Layout()
    layout.split_column(
        Layout(name="market", size=9),
        Layout(name="portfolio", size=8),
        Layout(name="strategies", size=15)
    )
    
    with Live(layout, refresh_per_second=2) as live:
        for round_num in range(simulation_rounds):
            # 김프 트렌드 시뮬레이션
            if round_num % 20 == 0:
                premium_trend = random.uniform(-0.5, 0.5)
            base_premium += premium_trend * 0.1
            base_premium = max(0.5, min(5.0, base_premium))  # 0.5% ~ 5% 제한
            
            # 시장 데이터 생성
            market_data = generate_market_data(base_premium)
            
            # 전략 분석 (비동기)
            signals = await manager.analyze_market(market_data)
            
            # 신호 통합
            if signals:
                aggregated_signal = manager.aggregate_signals(signals)
                
                if aggregated_signal:
                    # 거래 실행 시뮬레이션
                    for strategy in manager.strategies.values():
                        if strategy.position == 0 and aggregated_signal.signal_type.value == "BUY":
                            # 진입 시뮬레이션
                            strategy.execute_trade(aggregated_signal, market_data.upbit_price)
                        elif strategy.position != 0:
                            # 청산 체크
                            if strategy.should_close_position(market_data):
                                close_signal = strategy._create_close_signal(market_data)
                                strategy.execute_trade(close_signal, market_data.upbit_price)
            
            # UI 업데이트
            layout["market"].update(create_market_panel(market_data))
            layout["portfolio"].update(create_portfolio_panel(manager))
            layout["strategies"].update(create_strategies_table(manager))
            
            # 딜레이
            await asyncio.sleep(0.5)
    
    # 최종 결과
    console.print("\n" + "="*60)
    console.print(Panel.fit("📊 시뮬레이션 완료", style="bold green"))
    
    # 최종 성과
    final_table = Table(title="🏆 최종 성과", show_header=True, header_style="bold yellow")
    final_table.add_column("전략", style="cyan")
    final_table.add_column("거래 횟수", justify="right")
    final_table.add_column("승률", justify="right")
    final_table.add_column("총 PnL", justify="right")
    final_table.add_column("최고 수익", justify="right")
    final_table.add_column("최대 손실", justify="right")
    
    for status in manager.get_all_strategies_status():
        strategy = manager.strategies[status['strategy_name']]
        perf = strategy.performance
        
        pnl_color = "green" if perf.total_pnl >= 0 else "red"
        
        final_table.add_row(
            status['strategy_name'],
            str(perf.total_trades),
            f"{perf.win_rate:.1%}",
            f"[{pnl_color}]{perf.total_pnl:+,.0f}[/{pnl_color}]",
            f"{perf.best_trade:+,.0f}",
            f"{perf.worst_trade:+,.0f}"
        )
    
    console.print(final_table)
    
    # 포트폴리오 최종 상태
    portfolio = manager.get_portfolio_status()
    total_pnl = portfolio['total_pnl']
    pnl_color = "green" if total_pnl >= 0 else "red"
    
    console.print(f"\n💼 포트폴리오 총 수익: [{pnl_color}]{total_pnl:+,.0f} KRW ({portfolio['total_pnl_pct']})[/{pnl_color}]")


def main():
    """메인 함수"""
    try:
        # 이벤트 루프 실행
        asyncio.run(run_simulation())
    except KeyboardInterrupt:
        console.print("\n[yellow]시뮬레이션 중단됨[/yellow]")
    except Exception as e:
        console.print(f"\n[red]오류 발생: {e}[/red]")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()