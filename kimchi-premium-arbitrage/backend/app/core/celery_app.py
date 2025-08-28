"""
Celery configuration and initialization
"""
from celery import Celery
from app.core.config import settings
import os

# Create Celery instance
celery_app = Celery(
    "kimchi_premium_arbitrage",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=[
        "app.tasks.backtesting_tasks",
        "app.tasks.data_collection_tasks",
        "app.tasks.analysis_tasks",
        "app.tasks.notification_tasks"
    ]
)

# Configure Celery
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    
    # Task execution settings
    task_soft_time_limit=3600,  # 1 hour soft limit
    task_time_limit=7200,  # 2 hour hard limit
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    
    # Result backend settings
    result_expires=86400,  # Results expire after 24 hours
    result_backend_always_retry=True,
    result_backend_max_retries=10,
    
    # Worker settings
    worker_prefetch_multiplier=4,
    worker_max_tasks_per_child=1000,
    worker_disable_rate_limits=False,
    
    # Beat schedule for periodic tasks
    beat_schedule={
        'collect-market-data': {
            'task': 'app.tasks.data_collection_tasks.collect_market_data',
            'schedule': 60.0,  # Every minute
        },
        'update-kimchi-premium': {
            'task': 'app.tasks.data_collection_tasks.update_kimchi_premium',
            'schedule': 30.0,  # Every 30 seconds
        },
        'cleanup-old-data': {
            'task': 'app.tasks.data_collection_tasks.cleanup_old_data',
            'schedule': 3600.0,  # Every hour
        },
        'generate-daily-report': {
            'task': 'app.tasks.analysis_tasks.generate_daily_report',
            'schedule': 86400.0,  # Daily
        },
    },
    
    # Queue routing
    task_routes={
        'app.tasks.backtesting_tasks.*': {'queue': 'backtesting'},
        'app.tasks.data_collection_tasks.*': {'queue': 'data_collection'},
        'app.tasks.analysis_tasks.*': {'queue': 'analysis'},
        'app.tasks.notification_tasks.*': {'queue': 'notifications'},
    },
    
    # Task annotations for rate limiting
    task_annotations={
        'app.tasks.data_collection_tasks.collect_market_data': {
            'rate_limit': '10/m'  # 10 per minute
        },
        'app.tasks.notification_tasks.send_email': {
            'rate_limit': '30/m'  # 30 per minute
        }
    }
)

# Set Redis connection pool settings
celery_app.conf.broker_transport_options = {
    'visibility_timeout': 3600,
    'fanout_prefix': True,
    'fanout_patterns': True,
    'priority_steps': list(range(10)),
    'max_connections': 50,
    'health_check_interval': 30,
}