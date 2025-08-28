"""
Celery tasks for notification operations
"""
from celery import Task
from typing import Dict, Any, List, Optional
import asyncio
from datetime import datetime
import json
import aiohttp
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib

from app.core.celery_app import celery_app
from app.core.config import settings
from app.core.database import AsyncSessionLocal
from app.models.system import Alert, Notification
from sqlalchemy import select, and_

@celery_app.task(name='send_email')
def send_email(
    to: str,
    subject: str,
    body: str,
    html: bool = False
) -> Dict[str, Any]:
    """
    Send email notification
    
    Args:
        to: Recipient email address
        subject: Email subject
        body: Email body
        html: Whether body is HTML
        
    Returns:
        Dict with send status
    """
    try:
        # Create message
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = settings.SMTP_FROM
        msg['To'] = to
        
        # Add body
        if html:
            part = MIMEText(body, 'html')
        else:
            part = MIMEText(body, 'plain')
        msg.attach(part)
        
        # Send email
        with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT) as server:
            if settings.SMTP_TLS:
                server.starttls()
            if settings.SMTP_USER and settings.SMTP_PASSWORD:
                server.login(settings.SMTP_USER, settings.SMTP_PASSWORD)
            server.send_message(msg)
        
        return {
            "status": "sent",
            "to": to,
            "subject": subject,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "failed",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

@celery_app.task(name='send_slack_notification')
def send_slack_notification(
    channel: str,
    message: str,
    attachments: Optional[List[Dict]] = None
) -> Dict[str, Any]:
    """
    Send Slack notification
    
    Args:
        channel: Slack channel
        message: Message text
        attachments: Optional message attachments
        
    Returns:
        Dict with send status
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        result = loop.run_until_complete(
            _send_slack_notification_async(channel, message, attachments)
        )
        return result
    finally:
        loop.close()

async def _send_slack_notification_async(
    channel: str,
    message: str,
    attachments: Optional[List[Dict]]
) -> Dict[str, Any]:
    """Async implementation of Slack notification"""
    if not settings.SLACK_WEBHOOK_URL:
        return {
            "status": "skipped",
            "reason": "Slack webhook not configured"
        }
    
    payload = {
        "channel": channel,
        "text": message
    }
    
    if attachments:
        payload["attachments"] = attachments
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                settings.SLACK_WEBHOOK_URL,
                json=payload
            ) as response:
                if response.status == 200:
                    return {
                        "status": "sent",
                        "channel": channel,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                else:
                    return {
                        "status": "failed",
                        "status_code": response.status,
                        "timestamp": datetime.utcnow().isoformat()
                    }
    except Exception as e:
        return {
            "status": "failed",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

@celery_app.task(name='send_telegram_notification')
def send_telegram_notification(
    chat_id: str,
    message: str,
    parse_mode: str = "HTML"
) -> Dict[str, Any]:
    """
    Send Telegram notification
    
    Args:
        chat_id: Telegram chat ID
        message: Message text
        parse_mode: Message parse mode (HTML, Markdown)
        
    Returns:
        Dict with send status
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        result = loop.run_until_complete(
            _send_telegram_notification_async(chat_id, message, parse_mode)
        )
        return result
    finally:
        loop.close()

async def _send_telegram_notification_async(
    chat_id: str,
    message: str,
    parse_mode: str
) -> Dict[str, Any]:
    """Async implementation of Telegram notification"""
    if not settings.TELEGRAM_BOT_TOKEN:
        return {
            "status": "skipped",
            "reason": "Telegram bot token not configured"
        }
    
    url = f"https://api.telegram.org/bot{settings.TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": parse_mode
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                data = await response.json()
                
                if data.get("ok"):
                    return {
                        "status": "sent",
                        "chat_id": chat_id,
                        "message_id": data["result"]["message_id"],
                        "timestamp": datetime.utcnow().isoformat()
                    }
                else:
                    return {
                        "status": "failed",
                        "error": data.get("description", "Unknown error"),
                        "timestamp": datetime.utcnow().isoformat()
                    }
    except Exception as e:
        return {
            "status": "failed",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

@celery_app.task(name='process_alerts')
def process_alerts() -> Dict[str, Any]:
    """
    Process pending alerts and send notifications
    
    Returns:
        Dict with processing status
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        result = loop.run_until_complete(_process_alerts_async())
        return result
    finally:
        loop.close()

async def _process_alerts_async() -> Dict[str, Any]:
    """Async implementation of alert processing"""
    processed_alerts = []
    failed_alerts = []
    
    async with AsyncSessionLocal() as db:
        # Get pending alerts
        result = await db.execute(
            select(Alert)
            .where(Alert.is_active == True)
            .order_by(Alert.priority.desc(), Alert.created_at)
        )
        alerts = result.scalars().all()
        
        for alert in alerts:
            try:
                # Check if alert condition is met
                if await _check_alert_condition(alert):
                    # Send notifications based on alert configuration
                    notification_results = []
                    
                    if alert.notification_channels.get("email"):
                        result = send_email.delay(
                            to=alert.notification_channels["email"],
                            subject=f"Alert: {alert.title}",
                            body=alert.message
                        )
                        notification_results.append(("email", result.id))
                    
                    if alert.notification_channels.get("slack"):
                        result = send_slack_notification.delay(
                            channel=alert.notification_channels["slack"],
                            message=f"🚨 *Alert: {alert.title}*\n{alert.message}"
                        )
                        notification_results.append(("slack", result.id))
                    
                    if alert.notification_channels.get("telegram"):
                        result = send_telegram_notification.delay(
                            chat_id=alert.notification_channels["telegram"],
                            message=f"<b>Alert: {alert.title}</b>\n{alert.message}"
                        )
                        notification_results.append(("telegram", result.id))
                    
                    # Create notification record
                    notification = Notification(
                        alert_id=alert.id,
                        title=alert.title,
                        message=alert.message,
                        channel=json.dumps([r[0] for r in notification_results]),
                        status="sent",
                        sent_at=datetime.utcnow()
                    )
                    db.add(notification)
                    
                    # Update alert
                    alert.last_triggered = datetime.utcnow()
                    alert.trigger_count += 1
                    
                    # Deactivate if one-time alert
                    if not alert.is_recurring:
                        alert.is_active = False
                    
                    processed_alerts.append(alert.id)
                    
            except Exception as e:
                failed_alerts.append({
                    "alert_id": alert.id,
                    "error": str(e)
                })
        
        await db.commit()
    
    return {
        "status": "completed",
        "processed": processed_alerts,
        "failed": failed_alerts,
        "timestamp": datetime.utcnow().isoformat()
    }

async def _check_alert_condition(alert: Alert) -> bool:
    """
    Check if alert condition is met
    
    Args:
        alert: Alert object
        
    Returns:
        True if condition is met
    """
    # This is a simplified implementation
    # In production, this would evaluate complex conditions
    
    condition_type = alert.condition.get("type")
    
    if condition_type == "price_threshold":
        # Check if price crosses threshold
        # TODO: Implement actual price checking
        return False
    
    elif condition_type == "kimchi_premium":
        # Check if Kimchi premium exceeds threshold
        # TODO: Implement actual premium checking
        return False
    
    elif condition_type == "position_pnl":
        # Check if position PnL crosses threshold
        # TODO: Implement actual PnL checking
        return False
    
    return False

@celery_app.task(name='send_trade_notification')
def send_trade_notification(trade_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Send notification for executed trade
    
    Args:
        trade_data: Trade information
        
    Returns:
        Dict with notification status
    """
    # Format trade message
    message = f"""
    Trade Executed:
    Symbol: {trade_data.get('symbol')}
    Side: {trade_data.get('side')}
    Quantity: {trade_data.get('quantity')}
    Price: {trade_data.get('price')}
    Exchange: {trade_data.get('exchange')}
    Time: {trade_data.get('timestamp')}
    """
    
    # Send to all configured channels
    results = []
    
    if settings.NOTIFICATION_EMAIL:
        result = send_email.delay(
            to=settings.NOTIFICATION_EMAIL,
            subject="Trade Executed",
            body=message
        )
        results.append(("email", result.id))
    
    if settings.SLACK_CHANNEL:
        result = send_slack_notification.delay(
            channel=settings.SLACK_CHANNEL,
            message=message
        )
        results.append(("slack", result.id))
    
    if settings.TELEGRAM_CHAT_ID:
        result = send_telegram_notification.delay(
            chat_id=settings.TELEGRAM_CHAT_ID,
            message=message
        )
        results.append(("telegram", result.id))
    
    return {
        "status": "notifications_sent",
        "channels": [r[0] for r in results],
        "task_ids": [r[1] for r in results],
        "timestamp": datetime.utcnow().isoformat()
    }