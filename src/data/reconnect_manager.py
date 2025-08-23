"""
WebSocket 재연결 관리자
자동 재연결, 데이터 갭 처리, 연결 모니터링
"""

import asyncio
import random
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from src.utils.alerts import AlertLevel, alert_manager
from src.utils.logger import logger


class ConnectionState(Enum):
    """연결 상태"""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    FAILED = "failed"


class ReconnectStrategy(Enum):
    """재연결 전략"""

    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIXED_DELAY = "fixed_delay"


class ReconnectManager:
    """재연결 관리자"""

    def __init__(
        self,
        max_attempts: int = 10,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        strategy: ReconnectStrategy = ReconnectStrategy.EXPONENTIAL_BACKOFF,
    ):
        """
        초기화

        Args:
            max_attempts: 최대 재연결 시도 횟수
            base_delay: 기본 지연 시간 (초)
            max_delay: 최대 지연 시간 (초)
            strategy: 재연결 전략
        """
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.strategy = strategy

        # 연결 상태
        self.states = {}
        self.attempt_counts = {}
        self.last_connect_times = {}
        self.disconnect_times = {}

        # 데이터 갭 추적
        self.last_data_timestamps = {}
        self.data_gaps = []

        # 콜백
        self.on_reconnect_callbacks = []
        self.on_disconnect_callbacks = []
        self.on_failed_callbacks = []

        logger.info(f"ReconnectManager initialized with strategy: {strategy.value}")

    def register_connection(self, connection_id: str):
        """연결 등록"""
        self.states[connection_id] = ConnectionState.DISCONNECTED
        self.attempt_counts[connection_id] = 0
        self.last_connect_times[connection_id] = None
        self.disconnect_times[connection_id] = None
        logger.debug(f"Connection registered: {connection_id}")

    def on_connected(self, connection_id: str):
        """연결 성공 시 호출"""
        if connection_id not in self.states:
            self.register_connection(connection_id)

        prev_state = self.states[connection_id]
        self.states[connection_id] = ConnectionState.CONNECTED
        self.attempt_counts[connection_id] = 0
        self.last_connect_times[connection_id] = datetime.now()

        # 데이터 갭 체크
        if self.disconnect_times[connection_id]:
            gap_duration = (datetime.now() - self.disconnect_times[connection_id]).total_seconds()
            if gap_duration > 5:  # 5초 이상 끊김
                self.data_gaps.append(
                    {
                        "connection_id": connection_id,
                        "start": self.disconnect_times[connection_id],
                        "end": datetime.now(),
                        "duration": gap_duration,
                    }
                )
                logger.warning(f"Data gap detected on {connection_id}: {gap_duration:.1f}s")

        self.disconnect_times[connection_id] = None

        # 재연결 성공 알림
        if prev_state == ConnectionState.RECONNECTING:
            logger.info(f"Reconnected successfully: {connection_id}")
            asyncio.create_task(self._notify_reconnect(connection_id))

    def on_disconnected(self, connection_id: str, error: Optional[Exception] = None):
        """연결 끊김 시 호출"""
        if connection_id not in self.states:
            self.register_connection(connection_id)

        self.states[connection_id] = ConnectionState.DISCONNECTED
        self.disconnect_times[connection_id] = datetime.now()

        if error:
            logger.error(f"Connection lost: {connection_id} - {error}")
        else:
            logger.warning(f"Connection lost: {connection_id}")

        asyncio.create_task(self._notify_disconnect(connection_id, error))

    async def reconnect(self, connection_id: str, connect_func: Callable) -> bool:
        """
        재연결 시도

        Args:
            connection_id: 연결 ID
            connect_func: 연결 함수

        Returns:
            성공 여부
        """
        if connection_id not in self.states:
            self.register_connection(connection_id)

        if self.states[connection_id] == ConnectionState.CONNECTED:
            logger.debug(f"Already connected: {connection_id}")
            return True

        self.states[connection_id] = ConnectionState.RECONNECTING

        while self.attempt_counts[connection_id] < self.max_attempts:
            self.attempt_counts[connection_id] += 1
            attempt = self.attempt_counts[connection_id]

            # 지연 시간 계산
            delay = self._calculate_delay(attempt)

            logger.info(
                f"Reconnection attempt {attempt}/{self.max_attempts} for {connection_id} "
                f"(delay: {delay:.1f}s)"
            )

            # 지연
            await asyncio.sleep(delay)

            try:
                # 연결 시도
                self.states[connection_id] = ConnectionState.CONNECTING
                await connect_func()

                # 성공
                self.on_connected(connection_id)
                return True

            except Exception as e:
                logger.error(f"Reconnection attempt {attempt} failed for {connection_id}: {e}")

                if attempt >= self.max_attempts:
                    self.states[connection_id] = ConnectionState.FAILED
                    logger.critical(f"Max reconnection attempts reached for {connection_id}")
                    asyncio.create_task(self._notify_failed(connection_id))
                    return False

        return False

    def _calculate_delay(self, attempt: int) -> float:
        """재연결 지연 시간 계산"""
        if self.strategy == ReconnectStrategy.EXPONENTIAL_BACKOFF:
            # 2^n * base_delay (+ jitter)
            delay = min(self.base_delay * (2 ** (attempt - 1)), self.max_delay)
            jitter = random.uniform(0, delay * 0.1)  # 10% jitter
            return delay + jitter

        elif self.strategy == ReconnectStrategy.LINEAR_BACKOFF:
            # n * base_delay
            return min(self.base_delay * attempt, self.max_delay)

        else:  # FIXED_DELAY
            return self.base_delay

    def update_data_timestamp(self, connection_id: str):
        """데이터 타임스탬프 업데이트"""
        self.last_data_timestamps[connection_id] = datetime.now()

    def check_data_gaps(self, connection_id: str) -> Optional[float]:
        """
        데이터 갭 체크

        Returns:
            갭 시간 (초) 또는 None
        """
        if connection_id not in self.last_data_timestamps:
            return None

        last_timestamp = self.last_data_timestamps[connection_id]
        gap = (datetime.now() - last_timestamp).total_seconds()

        return gap if gap > 5 else None  # 5초 이상을 갭으로 간주

    def get_connection_stats(self, connection_id: str) -> Dict[str, Any]:
        """연결 통계 조회"""
        if connection_id not in self.states:
            return {}

        uptime = None
        if (
            self.last_connect_times[connection_id]
            and self.states[connection_id] == ConnectionState.CONNECTED
        ):
            uptime = (datetime.now() - self.last_connect_times[connection_id]).total_seconds()

        return {
            "state": self.states[connection_id].value,
            "attempts": self.attempt_counts[connection_id],
            "last_connect": self.last_connect_times[connection_id],
            "disconnect_time": self.disconnect_times[connection_id],
            "uptime_seconds": uptime,
            "data_gaps": len([g for g in self.data_gaps if g["connection_id"] == connection_id]),
        }

    def get_all_stats(self) -> Dict[str, Any]:
        """전체 연결 통계"""
        stats = {}
        for connection_id in self.states:
            stats[connection_id] = self.get_connection_stats(connection_id)

        stats["summary"] = {
            "total_connections": len(self.states),
            "connected": sum(1 for s in self.states.values() if s == ConnectionState.CONNECTED),
            "disconnected": sum(
                1 for s in self.states.values() if s == ConnectionState.DISCONNECTED
            ),
            "reconnecting": sum(
                1 for s in self.states.values() if s == ConnectionState.RECONNECTING
            ),
            "failed": sum(1 for s in self.states.values() if s == ConnectionState.FAILED),
            "total_gaps": len(self.data_gaps),
        }

        return stats

    def reset_connection(self, connection_id: str):
        """연결 상태 초기화"""
        if connection_id in self.states:
            self.states[connection_id] = ConnectionState.DISCONNECTED
            self.attempt_counts[connection_id] = 0
            logger.info(f"Connection reset: {connection_id}")

    def on_reconnect(self, callback: Callable):
        """재연결 콜백 등록"""
        self.on_reconnect_callbacks.append(callback)

    def on_disconnect(self, callback: Callable):
        """연결 끊김 콜백 등록"""
        self.on_disconnect_callbacks.append(callback)

    def on_failed(self, callback: Callable):
        """연결 실패 콜백 등록"""
        self.on_failed_callbacks.append(callback)

    async def _notify_reconnect(self, connection_id: str):
        """재연결 알림"""
        for callback in self.on_reconnect_callbacks:
            try:
                await callback(connection_id)
            except Exception as e:
                logger.error(f"Error in reconnect callback: {e}")

        # 알림 전송
        await alert_manager.send_alert(
            message=f"WebSocket reconnected: {connection_id}",
            level=AlertLevel.INFO,
            title="Connection Restored",
            details={"connection": connection_id, "attempts": self.attempt_counts[connection_id]},
        )

    async def _notify_disconnect(self, connection_id: str, error: Optional[Exception]):
        """연결 끊김 알림"""
        for callback in self.on_disconnect_callbacks:
            try:
                await callback(connection_id, error)
            except Exception as e:
                logger.error(f"Error in disconnect callback: {e}")

        # 경고 알림 (첫 번째 끊김만)
        if self.attempt_counts[connection_id] == 0:
            await alert_manager.send_alert(
                message=f"WebSocket disconnected: {connection_id}",
                level=AlertLevel.WARNING,
                title="Connection Lost",
                details={"connection": connection_id, "error": str(error) if error else None},
            )

    async def _notify_failed(self, connection_id: str):
        """연결 실패 알림"""
        for callback in self.on_failed_callbacks:
            try:
                await callback(connection_id)
            except Exception as e:
                logger.error(f"Error in failed callback: {e}")

        # 치명적 알림
        await alert_manager.send_critical_alert(
            message=f"WebSocket connection failed permanently: {connection_id}",
            connection=connection_id,
            max_attempts=self.max_attempts,
        )


class ConnectionMonitor:
    """연결 모니터"""

    def __init__(self, reconnect_manager: ReconnectManager, check_interval: int = 30):
        """
        초기화

        Args:
            reconnect_manager: 재연결 관리자
            check_interval: 체크 간격 (초)
        """
        self.reconnect_manager = reconnect_manager
        self.check_interval = check_interval
        self.is_running = False
        self._monitor_task = None

    async def start(self):
        """모니터 시작"""
        if self.is_running:
            return

        self.is_running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("Connection monitor started")

    async def stop(self):
        """모니터 중지"""
        self.is_running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Connection monitor stopped")

    async def _monitor_loop(self):
        """모니터링 루프"""
        while self.is_running:
            try:
                await asyncio.sleep(self.check_interval)

                # 연결 상태 체크
                stats = self.reconnect_manager.get_all_stats()

                # 실패한 연결 체크
                failed_connections = [
                    cid
                    for cid, stat in stats.items()
                    if stat.get("state") == ConnectionState.FAILED.value
                ]

                if failed_connections:
                    logger.critical(f"Failed connections detected: {failed_connections}")

                # 데이터 갭 체크
                for connection_id in self.reconnect_manager.states:
                    gap = self.reconnect_manager.check_data_gaps(connection_id)
                    if gap and gap > 60:  # 60초 이상 데이터 없음
                        logger.warning(f"No data from {connection_id} for {gap:.0f}s")

                # 통계 로깅
                if stats["summary"]["disconnected"] > 0 or stats["summary"]["reconnecting"] > 0:
                    logger.info(f"Connection status: {stats['summary']}")

            except Exception as e:
                logger.error(f"Error in connection monitor: {e}")


# 전역 재연결 관리자
reconnect_manager = ReconnectManager()
connection_monitor = ConnectionMonitor(reconnect_manager)


if __name__ == "__main__":
    # 재연결 관리자 테스트
    async def test_connect():
        """테스트 연결 함수"""
        import random

        if random.random() > 0.3:  # 70% 성공률
            return True
        raise Exception("Connection failed")

    async def main():
        print("Reconnect Manager Test")
        print("-" * 40)

        # 연결 등록
        reconnect_manager.register_connection("test_exchange")

        # 재연결 테스트
        success = await reconnect_manager.reconnect("test_exchange", test_connect)

        if success:
            print("Reconnection successful!")
        else:
            print("Reconnection failed!")

        # 통계 출력
        stats = reconnect_manager.get_all_stats()
        print(f"\nConnection stats: {stats}")

    asyncio.run(main())
