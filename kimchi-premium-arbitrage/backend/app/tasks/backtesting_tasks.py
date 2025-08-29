"""
Celery tasks for backtesting operations
"""
from celery import Task
from typing import Dict, Any, Optional
import asyncio
from datetime import datetime
import json

from app.core.celery_app import celery_app
from app.core.database import AsyncSessionLocal
from app.services.backtest_service import BacktestService
from app.core.websocket_manager import WebSocketManager
from sqlalchemy import select
from app.models.backtesting import Backtest

class CallbackTask(Task):
    """Task with progress callback support"""
    def __init__(self):
        self.backtest_service = BacktestService()
        self.ws_manager = WebSocketManager()

@celery_app.task(bind=True, base=CallbackTask, name='run_backtest')
def run_backtest(self, backtest_id: int) -> Dict[str, Any]:
    """
    Run a backtest asynchronously
    
    Args:
        backtest_id: ID of the backtest to run
        
    Returns:
        Dict with backtest results
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        result = loop.run_until_complete(_run_backtest_async(self, backtest_id))
        return result
    finally:
        loop.close()

async def _run_backtest_async(task: CallbackTask, backtest_id: int) -> Dict[str, Any]:
    """Async implementation of backtest execution"""
    async with AsyncSessionLocal() as db:
        try:
            # Get backtest
            result = await db.execute(
                select(Backtest).where(Backtest.id == backtest_id)
            )
            backtest = result.scalar_one_or_none()
            
            if not backtest:
                raise ValueError(f"Backtest {backtest_id} not found")
            
            # Update status to running
            backtest.status = "running"
            backtest.started_at = datetime.utcnow()
            await db.commit()
            
            # Progress callback
            async def progress_callback(progress: int, message: str):
                # Update Celery task state
                task.update_state(
                    state='PROGRESS',
                    meta={
                        'current': progress,
                        'total': 100,
                        'message': message
                    }
                )
                
                # Send WebSocket update
                await task.ws_manager.broadcast(json.dumps({
                    "type": "backtest_progress",
                    "backtest_id": backtest_id,
                    "progress": progress,
                    "message": message
                }))
            
            # Run backtest with progress callback
            results = await task.backtest_service.run_backtest(
                db, backtest_id, progress_callback
            )
            
            # Update status to completed
            backtest.status = "completed"
            backtest.completed_at = datetime.utcnow()
            await db.commit()
            
            return {
                "status": "completed",
                "backtest_id": backtest_id,
                "results": results
            }
            
        except Exception as e:
            # Update status to failed
            if backtest:
                backtest.status = "failed"
                backtest.error_message = str(e)
                backtest.completed_at = datetime.utcnow()
                await db.commit()
            
            # Send error notification
            await task.ws_manager.broadcast(json.dumps({
                "type": "backtest_error",
                "backtest_id": backtest_id,
                "error": str(e)
            }))
            
            raise

@celery_app.task(name='cancel_backtest')
def cancel_backtest(backtest_id: int) -> Dict[str, Any]:
    """
    Cancel a running backtest
    
    Args:
        backtest_id: ID of the backtest to cancel
        
    Returns:
        Dict with cancellation status
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        result = loop.run_until_complete(_cancel_backtest_async(backtest_id))
        return result
    finally:
        loop.close()

async def _cancel_backtest_async(backtest_id: int) -> Dict[str, Any]:
    """Async implementation of backtest cancellation"""
    async with AsyncSessionLocal() as db:
        # Get backtest
        result = await db.execute(
            select(Backtest).where(Backtest.id == backtest_id)
        )
        backtest = result.scalar_one_or_none()
        
        if not backtest:
            raise ValueError(f"Backtest {backtest_id} not found")
        
        if backtest.status != "running":
            return {
                "status": "not_running",
                "message": f"Backtest {backtest_id} is not running"
            }
        
        # Update status to cancelled
        backtest.status = "cancelled"
        backtest.completed_at = datetime.utcnow()
        await db.commit()
        
        # Revoke the task if it's in Celery queue
        celery_app.control.revoke(f"run_backtest_{backtest_id}", terminate=True)
        
        # Send cancellation notification
        ws_manager = WebSocketManager()
        await ws_manager.broadcast(json.dumps({
            "type": "backtest_cancelled",
            "backtest_id": backtest_id
        }))
        
        return {
            "status": "cancelled",
            "backtest_id": backtest_id
        }

@celery_app.task(name='batch_backtest')
def batch_backtest(backtest_configs: list) -> Dict[str, Any]:
    """
    Run multiple backtests in batch
    
    Args:
        backtest_configs: List of backtest configurations
        
    Returns:
        Dict with batch results
    """
    results = []
    
    for config in backtest_configs:
        try:
            # Create and run each backtest
            task = run_backtest.delay(config['backtest_id'])
            results.append({
                "backtest_id": config['backtest_id'],
                "task_id": task.id,
                "status": "queued"
            })
        except Exception as e:
            results.append({
                "backtest_id": config.get('backtest_id'),
                "error": str(e),
                "status": "failed"
            })
    
    return {
        "batch_size": len(backtest_configs),
        "results": results
    }

@celery_app.task(name='optimize_strategy_parameters')
def optimize_strategy_parameters(
    strategy_id: int,
    parameter_ranges: Dict[str, Any],
    optimization_method: str = "grid_search"
) -> Dict[str, Any]:
    """
    Optimize strategy parameters using various methods
    
    Args:
        strategy_id: ID of the strategy to optimize
        parameter_ranges: Parameter ranges for optimization
        optimization_method: Method to use (grid_search, random_search, bayesian)
        
    Returns:
        Dict with optimal parameters and performance
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        result = loop.run_until_complete(
            _optimize_strategy_async(strategy_id, parameter_ranges, optimization_method)
        )
        return result
    finally:
        loop.close()

async def _optimize_strategy_async(
    strategy_id: int,
    parameter_ranges: Dict[str, Any],
    optimization_method: str
) -> Dict[str, Any]:
    """Async implementation of strategy optimization"""
    # This is a placeholder for actual optimization logic
    # In production, this would run multiple backtests with different parameters
    # and find the optimal combination
    
    best_params = {}
    best_performance = 0
    
    if optimization_method == "grid_search":
        # Implement grid search
        combinations = []  # Generate all parameter combinations
        for combo in combinations:
            # Run backtest with these parameters
            # Track best performing combination
            pass
    
    elif optimization_method == "random_search":
        # Implement random search
        n_iterations = 100
        for _ in range(n_iterations):
            # Generate random parameters within ranges
            # Run backtest
            # Track best
            pass
    
    elif optimization_method == "bayesian":
        # Implement Bayesian optimization
        # Use libraries like scikit-optimize
        pass
    
    return {
        "strategy_id": strategy_id,
        "optimization_method": optimization_method,
        "best_parameters": best_params,
        "best_performance": best_performance,
        "timestamp": datetime.utcnow().isoformat()
    }