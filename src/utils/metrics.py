"""
성능 메트릭 수집 및 모니터링
Prometheus 형식 메트릭 및 시스템 모니터링
"""

import json
import threading
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil

# Prometheus 클라이언트 (선택적)
try:
    from prometheus_client import Counter, Gauge, Histogram, Summary, start_http_server

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

    # 더미 클래스들
    class Counter:
        pass

    class Gauge:
        pass

    class Histogram:
        pass

    class Summary:
        pass


from src.utils.logger import logger


@dataclass
class SystemMetrics:
    """시스템 메트릭"""

    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_mb: float
    disk_percent: float
    network_sent_mb: float
    network_recv_mb: float
    open_files: int
    threads: int


@dataclass
class TradingMetrics:
    """거래 메트릭"""

    timestamp: datetime
    total_trades: int
    successful_trades: int
    failed_trades: int
    total_volume: float
    total_pnl: float
    win_rate: float
    avg_latency_ms: float
    kimchi_premium: float


class MetricsCollector:
    """메트릭 수집기"""

    def __init__(self, export_interval: int = 60):
        """
        초기화

        Args:
            export_interval: 메트릭 내보내기 간격 (초)
        """
        self.export_interval = export_interval
        self.metrics_dir = Path("logs/metrics")
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        # 메트릭 저장소
        self.system_metrics = deque(maxlen=1440)  # 24시간 (분당)
        self.trading_metrics = deque(maxlen=1440)
        self.api_latencies = defaultdict(list)
        self.error_counts = defaultdict(int)

        # Prometheus 메트릭 (가능한 경우)
        if PROMETHEUS_AVAILABLE:
            self._setup_prometheus_metrics()

        # 백그라운드 수집 스레드
        self._stop_event = threading.Event()
        self._collector_thread = None

    def _setup_prometheus_metrics(self):
        """Prometheus 메트릭 설정"""
        # 카운터
        self.trade_counter = Counter(
            "trades_total", "Total number of trades", ["exchange", "symbol", "side"]
        )
        self.error_counter = Counter("errors_total", "Total number of errors", ["type", "exchange"])

        # 게이지
        self.kimchi_premium_gauge = Gauge("kimchi_premium", "Current Kimchi Premium (%)")
        self.balance_gauge = Gauge("balance", "Account balance", ["exchange", "currency"])
        self.position_gauge = Gauge("position_size", "Position size", ["symbol"])

        # 히스토그램
        self.latency_histogram = Histogram(
            "api_latency_seconds", "API latency", ["exchange", "operation"]
        )
        self.trade_size_histogram = Histogram("trade_size", "Trade size distribution", ["symbol"])

        # 요약
        self.pnl_summary = Summary("pnl", "Profit and Loss")

        logger.info("Prometheus metrics initialized")

    def collect_system_metrics(self) -> SystemMetrics:
        """시스템 메트릭 수집"""
        try:
            # CPU 및 메모리
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()

            # 디스크
            disk = psutil.disk_usage("/")

            # 네트워크
            net_io = psutil.net_io_counters()

            # 프로세스 정보
            process = psutil.Process()

            metrics = SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_mb=memory.used / 1024 / 1024,
                disk_percent=disk.percent,
                network_sent_mb=net_io.bytes_sent / 1024 / 1024,
                network_recv_mb=net_io.bytes_recv / 1024 / 1024,
                open_files=len(process.open_files()),
                threads=process.num_threads(),
            )

            self.system_metrics.append(metrics)

            # Prometheus 업데이트
            if PROMETHEUS_AVAILABLE and hasattr(self, "balance_gauge"):
                # 시스템 메트릭을 게이지로 (예시)
                pass

            return metrics

        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
            return None

    def record_api_latency(self, exchange: str, operation: str, latency_ms: float):
        """API 지연시간 기록"""
        key = f"{exchange}_{operation}"
        self.api_latencies[key].append(latency_ms)

        # 최근 100개만 유지
        if len(self.api_latencies[key]) > 100:
            self.api_latencies[key] = self.api_latencies[key][-100:]

        # Prometheus 업데이트
        if PROMETHEUS_AVAILABLE and hasattr(self, "latency_histogram"):
            self.latency_histogram.labels(exchange=exchange, operation=operation).observe(
                latency_ms / 1000
            )  # 초 단위로 변환

        logger.debug(f"API latency recorded: {exchange} {operation} = {latency_ms:.2f}ms")

    def record_trade(
        self,
        exchange: str,
        symbol: str,
        side: str,
        amount: float,
        price: float,
        success: bool = True,
    ):
        """거래 기록"""
        if PROMETHEUS_AVAILABLE and hasattr(self, "trade_counter"):
            self.trade_counter.labels(exchange=exchange, symbol=symbol, side=side).inc()

            self.trade_size_histogram.labels(symbol=symbol).observe(amount)

        trade_data = {
            "timestamp": datetime.now().isoformat(),
            "exchange": exchange,
            "symbol": symbol,
            "side": side,
            "amount": amount,
            "price": price,
            "success": success,
        }

        # 파일로 저장 (일별)
        today = datetime.now().strftime("%Y%m%d")
        trade_file = self.metrics_dir / f"trades_{today}.jsonl"

        with open(trade_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(trade_data, ensure_ascii=False) + "\n")

    def record_error(
        self, error_type: str, exchange: Optional[str] = None, details: Optional[str] = None
    ):
        """에러 기록"""
        key = f"{error_type}_{exchange}" if exchange else error_type
        self.error_counts[key] += 1

        if PROMETHEUS_AVAILABLE and hasattr(self, "error_counter"):
            self.error_counter.labels(type=error_type, exchange=exchange or "unknown").inc()

        error_data = {
            "timestamp": datetime.now().isoformat(),
            "type": error_type,
            "exchange": exchange,
            "details": details,
            "count": self.error_counts[key],
        }

        # 에러 로그 파일
        error_file = self.metrics_dir / "errors.jsonl"
        with open(error_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(error_data, ensure_ascii=False) + "\n")

    def update_kimchi_premium(self, premium: float):
        """김치 프리미엄 업데이트"""
        if PROMETHEUS_AVAILABLE and hasattr(self, "kimchi_premium_gauge"):
            self.kimchi_premium_gauge.set(premium)

        # 시계열 저장
        premium_data = {"timestamp": datetime.now().isoformat(), "premium": premium}

        # 파일로 저장
        today = datetime.now().strftime("%Y%m%d")
        premium_file = self.metrics_dir / f"kimchi_premium_{today}.jsonl"

        with open(premium_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(premium_data) + "\n")

    def update_balance(self, exchange: str, currency: str, balance: float):
        """잔고 업데이트"""
        if PROMETHEUS_AVAILABLE and hasattr(self, "balance_gauge"):
            self.balance_gauge.labels(exchange=exchange, currency=currency).set(balance)

    def update_position(self, symbol: str, size: float):
        """포지션 업데이트"""
        if PROMETHEUS_AVAILABLE and hasattr(self, "position_gauge"):
            self.position_gauge.labels(symbol=symbol).set(size)

    def record_pnl(self, pnl: float):
        """손익 기록"""
        if PROMETHEUS_AVAILABLE and hasattr(self, "pnl_summary"):
            self.pnl_summary.observe(pnl)

        pnl_data = {"timestamp": datetime.now().isoformat(), "pnl": pnl}

        # 파일로 저장
        today = datetime.now().strftime("%Y%m%d")
        pnl_file = self.metrics_dir / f"pnl_{today}.jsonl"

        with open(pnl_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(pnl_data) + "\n")

    def get_average_latency(self, exchange: str, operation: str) -> Optional[float]:
        """평균 지연시간 조회"""
        key = f"{exchange}_{operation}"
        latencies = self.api_latencies.get(key, [])

        if latencies:
            return sum(latencies) / len(latencies)
        return None

    def get_system_status(self) -> Dict[str, Any]:
        """시스템 상태 요약"""
        if self.system_metrics:
            latest = self.system_metrics[-1]

            # 최근 5분 평균
            recent_metrics = list(self.system_metrics)[-5:]
            avg_cpu = sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)
            avg_memory = sum(m.memory_percent for m in recent_metrics) / len(recent_metrics)

            return {
                "timestamp": latest.timestamp.isoformat(),
                "current": {
                    "cpu_percent": latest.cpu_percent,
                    "memory_percent": latest.memory_percent,
                    "memory_mb": latest.memory_mb,
                    "disk_percent": latest.disk_percent,
                },
                "average_5min": {
                    "cpu_percent": avg_cpu,
                    "memory_percent": avg_memory,
                },
                "network": {
                    "sent_mb": latest.network_sent_mb,
                    "recv_mb": latest.network_recv_mb,
                },
                "process": {
                    "open_files": latest.open_files,
                    "threads": latest.threads,
                },
            }
        return {}

    def export_metrics(self):
        """메트릭 내보내기 (파일로)"""
        try:
            # 시스템 메트릭
            system_status = self.get_system_status()
            if system_status:
                export_file = self.metrics_dir / "system_metrics.json"
                with open(export_file, "w", encoding="utf-8") as f:
                    json.dump(system_status, f, indent=2, ensure_ascii=False)

            # API 지연시간
            latency_summary = {}
            for key, values in self.api_latencies.items():
                if values:
                    latency_summary[key] = {
                        "avg": sum(values) / len(values),
                        "min": min(values),
                        "max": max(values),
                        "count": len(values),
                    }

            if latency_summary:
                latency_file = self.metrics_dir / "api_latency.json"
                with open(latency_file, "w", encoding="utf-8") as f:
                    json.dump(latency_summary, f, indent=2)

            # 에러 요약
            if self.error_counts:
                error_summary_file = self.metrics_dir / "error_summary.json"
                with open(error_summary_file, "w", encoding="utf-8") as f:
                    json.dump(dict(self.error_counts), f, indent=2)

            logger.debug("Metrics exported successfully")

        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")

    def start_collector(self):
        """백그라운드 수집 시작"""
        if self._collector_thread is None or not self._collector_thread.is_alive():
            self._stop_event.clear()
            self._collector_thread = threading.Thread(target=self._collector_loop)
            self._collector_thread.daemon = True
            self._collector_thread.start()
            logger.info("Metrics collector started")

    def stop_collector(self):
        """백그라운드 수집 중지"""
        if self._collector_thread and self._collector_thread.is_alive():
            self._stop_event.set()
            self._collector_thread.join(timeout=5)
            logger.info("Metrics collector stopped")

    def _collector_loop(self):
        """수집 루프"""
        next_export = time.time() + self.export_interval

        while not self._stop_event.is_set():
            try:
                # 시스템 메트릭 수집 (1분마다)
                self.collect_system_metrics()

                # 메트릭 내보내기
                if time.time() >= next_export:
                    self.export_metrics()
                    next_export = time.time() + self.export_interval

                # 1분 대기
                self._stop_event.wait(60)

            except Exception as e:
                logger.error(f"Collector loop error: {e}")
                time.sleep(60)


# 전역 메트릭 수집기
metrics_collector = MetricsCollector()


# 컨텍스트 매니저
class MetricsTimer:
    """실행 시간 측정 컨텍스트 매니저"""

    def __init__(self, exchange: str, operation: str):
        self.exchange = exchange
        self.operation = operation
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            elapsed_ms = (time.time() - self.start_time) * 1000
            metrics_collector.record_api_latency(self.exchange, self.operation, elapsed_ms)


if __name__ == "__main__":
    # 메트릭 수집기 테스트
    print("메트릭 수집기 테스트")
    print("-" * 40)

    # 시스템 메트릭 수집
    system_metrics = metrics_collector.collect_system_metrics()
    if system_metrics:
        print(f"CPU: {system_metrics.cpu_percent}%")
        print(f"Memory: {system_metrics.memory_percent}%")
        print(f"Disk: {system_metrics.disk_percent}%")

    # API 지연시간 기록
    metrics_collector.record_api_latency("upbit", "fetch_ticker", 25.5)
    metrics_collector.record_api_latency("binance", "fetch_ticker", 15.3)

    # 거래 기록
    metrics_collector.record_trade(
        exchange="upbit", symbol="BTC/KRW", side="buy", amount=0.01, price=100000000
    )

    # 김프 업데이트
    metrics_collector.update_kimchi_premium(4.5)

    # 타이머 테스트
    with MetricsTimer("upbit", "fetch_balance"):
        time.sleep(0.1)  # API 호출 시뮬레이션

    # 메트릭 내보내기
    metrics_collector.export_metrics()

    print(f"\n메트릭 파일 확인: {metrics_collector.metrics_dir}")
    print("테스트 완료!")
