"""
알림 시스템
Slack, Discord, Email 알림 지원
"""

import asyncio
import json
import os
import smtplib
import ssl
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from typing import Any, Dict, List, Optional

import aiohttp

from src.utils.logger import logger


class AlertLevel(Enum):
    """알림 레벨"""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    TRADE = "trade"  # 거래 알림


class AlertChannel(Enum):
    """알림 채널"""

    SLACK = "slack"
    DISCORD = "discord"
    EMAIL = "email"
    ALL = "all"


class AlertManager:
    """알림 관리자"""

    def __init__(self):
        """초기화"""
        # Webhook URLs
        self.slack_webhook = os.getenv("SLACK_WEBHOOK_URL")
        self.discord_webhook = os.getenv("DISCORD_WEBHOOK_URL")

        # Email 설정
        self.email_enabled = os.getenv("EMAIL_ALERTS_ENABLED", "false").lower() == "true"
        self.smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.email_from = os.getenv("EMAIL_FROM")
        self.email_password = os.getenv("EMAIL_PASSWORD")
        self.email_to = os.getenv("EMAIL_TO", "").split(",")

        # 알림 설정
        self.min_alert_level = AlertLevel[os.getenv("MIN_ALERT_LEVEL", "WARNING")]
        self.rate_limit = {}  # 중복 알림 방지

        logger.info("Alert manager initialized")

    async def send_alert(
        self,
        message: str,
        level: AlertLevel = AlertLevel.INFO,
        channel: AlertChannel = AlertChannel.ALL,
        title: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """
        알림 전송

        Args:
            message: 알림 메시지
            level: 알림 레벨
            channel: 알림 채널
            title: 알림 제목
            details: 추가 상세 정보
        """
        # 레벨 체크
        if self._should_skip_alert(level):
            return

        # 중복 방지
        if self._is_duplicate(message, level):
            return

        # 제목 설정
        if not title:
            title = f"[{level.value.upper()}] Kimchi Arbitrage Alert"

        # 각 채널로 전송
        tasks = []

        if channel in [AlertChannel.SLACK, AlertChannel.ALL] and self.slack_webhook:
            tasks.append(self._send_slack(title, message, level, details))

        if channel in [AlertChannel.DISCORD, AlertChannel.ALL] and self.discord_webhook:
            tasks.append(self._send_discord(title, message, level, details))

        if channel in [AlertChannel.EMAIL, AlertChannel.ALL] and self.email_enabled:
            tasks.append(self._send_email(title, message, level, details))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
            logger.info(f"Alert sent: {title} - {message[:50]}...")

    async def _send_slack(
        self, title: str, message: str, level: AlertLevel, details: Optional[Dict] = None
    ):
        """Slack 알림 전송"""
        try:
            # 색상 설정
            color_map = {
                AlertLevel.INFO: "#36a64f",
                AlertLevel.WARNING: "#ff9900",
                AlertLevel.ERROR: "#ff0000",
                AlertLevel.CRITICAL: "#8b0000",
                AlertLevel.TRADE: "#0099ff",
            }

            # 메시지 구성
            payload = {
                "text": title,
                "attachments": [
                    {
                        "color": color_map.get(level, "#808080"),
                        "fields": [
                            {"title": "Message", "value": message, "short": False},
                            {"title": "Level", "value": level.value, "short": True},
                            {
                                "title": "Time",
                                "value": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "short": True,
                            },
                        ],
                    }
                ],
            }

            # 상세 정보 추가
            if details:
                for key, value in details.items():
                    payload["attachments"][0]["fields"].append(
                        {"title": key, "value": str(value), "short": True}
                    )

            # 전송
            async with aiohttp.ClientSession() as session:
                async with session.post(self.slack_webhook, json=payload) as response:
                    if response.status != 200:
                        logger.error(f"Slack alert failed: {response.status}")

        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")

    async def _send_discord(
        self, title: str, message: str, level: AlertLevel, details: Optional[Dict] = None
    ):
        """Discord 알림 전송"""
        try:
            # 이모지 설정
            emoji_map = {
                AlertLevel.INFO: "ℹ️",
                AlertLevel.WARNING: "⚠️",
                AlertLevel.ERROR: "❌",
                AlertLevel.CRITICAL: "🚨",
                AlertLevel.TRADE: "💹",
            }

            # Embed 구성
            embed = {
                "title": f"{emoji_map.get(level, '📢')} {title}",
                "description": message,
                "color": {
                    AlertLevel.INFO: 0x36A64F,
                    AlertLevel.WARNING: 0xFF9900,
                    AlertLevel.ERROR: 0xFF0000,
                    AlertLevel.CRITICAL: 0x8B0000,
                    AlertLevel.TRADE: 0x0099FF,
                }.get(level, 0x808080),
                "timestamp": datetime.utcnow().isoformat(),
                "fields": [],
            }

            # 상세 정보 추가
            if details:
                for key, value in details.items():
                    embed["fields"].append({"name": key, "value": str(value), "inline": True})

            payload = {"embeds": [embed]}

            # 전송
            async with aiohttp.ClientSession() as session:
                async with session.post(self.discord_webhook, json=payload) as response:
                    if response.status not in [200, 204]:
                        logger.error(f"Discord alert failed: {response.status}")

        except Exception as e:
            logger.error(f"Failed to send Discord alert: {e}")

    async def _send_email(
        self, title: str, message: str, level: AlertLevel, details: Optional[Dict] = None
    ):
        """이메일 알림 전송"""
        try:
            # HTML 메시지 구성
            html_body = f"""
            <html>
                <body>
                    <h2>{title}</h2>
                    <p><strong>Level:</strong> {level.value}</p>
                    <p><strong>Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <hr>
                    <p>{message}</p>
            """

            if details:
                html_body += "<hr><h3>Details:</h3><ul>"
                for key, value in details.items():
                    html_body += f"<li><strong>{key}:</strong> {value}</li>"
                html_body += "</ul>"

            html_body += "</body></html>"

            # 이메일 메시지 생성
            msg = MIMEMultipart("alternative")
            msg["Subject"] = title
            msg["From"] = self.email_from
            msg["To"] = ", ".join(self.email_to)

            # HTML 파트 추가
            html_part = MIMEText(html_body, "html")
            msg.attach(html_part)

            # 이메일 전송
            context = ssl.create_default_context()
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls(context=context)
                server.login(self.email_from, self.email_password)
                server.send_message(msg)

        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")

    def _should_skip_alert(self, level: AlertLevel) -> bool:
        """알림 스킵 여부 확인"""
        level_priority = {
            AlertLevel.INFO: 0,
            AlertLevel.WARNING: 1,
            AlertLevel.ERROR: 2,
            AlertLevel.CRITICAL: 3,
            AlertLevel.TRADE: 1,  # WARNING과 동일 레벨
        }

        return level_priority.get(level, 0) < level_priority.get(self.min_alert_level, 0)

    def _is_duplicate(self, message: str, level: AlertLevel) -> bool:
        """중복 알림 체크 (5분 이내 동일 메시지)"""
        key = f"{level.value}:{message[:100]}"
        now = datetime.now().timestamp()

        if key in self.rate_limit:
            if now - self.rate_limit[key] < 300:  # 5분
                return True

        self.rate_limit[key] = now

        # 오래된 항목 정리
        if len(self.rate_limit) > 100:
            cutoff = now - 300
            self.rate_limit = {k: v for k, v in self.rate_limit.items() if v > cutoff}

        return False

    async def send_trade_alert(
        self, action: str, exchange: str, symbol: str, amount: float, price: float, **kwargs
    ):
        """거래 알림 전송"""
        message = f"Trade executed: {action} {amount} {symbol} @ {price} on {exchange}"

        details = {
            "Action": action,
            "Exchange": exchange,
            "Symbol": symbol,
            "Amount": amount,
            "Price": price,
            **kwargs,
        }

        await self.send_alert(
            message=message, level=AlertLevel.TRADE, title="Trade Execution", details=details
        )

    async def send_risk_alert(self, risk_type: str, message: str, **kwargs):
        """리스크 알림 전송"""
        await self.send_alert(
            message=message,
            level=AlertLevel.WARNING,
            title=f"Risk Alert: {risk_type}",
            details=kwargs,
        )

    async def send_error_alert(self, error: Exception, context: str = ""):
        """에러 알림 전송"""
        message = f"Error in {context}: {str(error)}"

        details = {"Error Type": type(error).__name__, "Context": context, "Error": str(error)}

        await self.send_alert(
            message=message, level=AlertLevel.ERROR, title="System Error", details=details
        )

    async def send_critical_alert(self, message: str, **kwargs):
        """치명적 알림 전송"""
        await self.send_alert(
            message=message,
            level=AlertLevel.CRITICAL,
            title="CRITICAL SYSTEM ALERT",
            details=kwargs,
        )


# 전역 알림 관리자
alert_manager = AlertManager()


# 동기 래퍼 함수들
def send_alert_sync(message: str, level: AlertLevel = AlertLevel.INFO, **kwargs):
    """동기 알림 전송"""
    asyncio.create_task(alert_manager.send_alert(message, level, **kwargs))


def send_trade_alert_sync(
    action: str, exchange: str, symbol: str, amount: float, price: float, **kwargs
):
    """동기 거래 알림 전송"""
    asyncio.create_task(
        alert_manager.send_trade_alert(action, exchange, symbol, amount, price, **kwargs)
    )


def send_risk_alert_sync(risk_type: str, message: str, **kwargs):
    """동기 리스크 알림 전송"""
    asyncio.create_task(alert_manager.send_risk_alert(risk_type, message, **kwargs))


def send_error_alert_sync(error: Exception, context: str = ""):
    """동기 에러 알림 전송"""
    asyncio.create_task(alert_manager.send_error_alert(error, context))


if __name__ == "__main__":
    # 테스트
    import asyncio

    async def test_alerts():
        print("알림 시스템 테스트")
        print("-" * 40)

        # 테스트 알림 전송
        await alert_manager.send_alert(
            message="테스트 알림입니다", level=AlertLevel.INFO, title="Test Alert"
        )

        # 거래 알림 테스트
        await alert_manager.send_trade_alert(
            action="BUY", exchange="upbit", symbol="BTC/KRW", amount=0.01, price=100000000
        )

        # 리스크 알림 테스트
        await alert_manager.send_risk_alert(
            risk_type="High Volatility",
            message="Market volatility exceeds threshold",
            volatility=0.05,
            threshold=0.03,
        )

        print("알림 전송 완료!")

    # 실행
    asyncio.run(test_alerts())
