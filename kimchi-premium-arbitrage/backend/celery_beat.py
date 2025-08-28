"""
Celery beat scheduler entry point
Run with: celery -A celery_beat beat --loglevel=info
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.core.celery_app import celery_app

if __name__ == '__main__':
    celery_app.start()