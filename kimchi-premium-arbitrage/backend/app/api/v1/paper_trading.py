"""
Paper Trading API endpoints
"""
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func

from app.core.database import get_db
from app.models.paper_trading import PaperTradingSession, PaperOrder, PaperPosition
from app.schemas.paper_trading import (
    PaperTradingSessionCreate as PaperSessionCreate,
    PaperTradingSessionResponse as PaperSessionResponse,
    PaperSessionUpdate,
    PaperOrderCreate, 
    PaperOrderResponse,
    PaperPositionResponse, 
    PaperBalanceResponse,
    PaperSessionMetrics as PaperPerformanceMetrics
)
from app.services.paper_trading_service import PaperTradingService
from app.core.websocket_manager import WebSocketManager
import json

router = APIRouter()
paper_service = PaperTradingService()
ws_manager = WebSocketManager()

@router.post("/sessions", response_model=PaperSessionResponse)
async def create_session(
    session_data: PaperSessionCreate,
    db: AsyncSession = Depends(get_db)
):
    """Create a new paper trading session"""
    try:
        session = await paper_service.create_session(db, session_data)
        
        # WebSocket으로 새 세션 알림
        await ws_manager.broadcast(json.dumps({
            "type": "paper_session_created",
            "session_id": session.id,
            "name": session.name,
            "initial_balance": float(session.initial_balance_krw)
        }))
        
        return session
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/sessions", response_model=List[PaperSessionResponse])
async def list_sessions(
    active_only: bool = Query(False, description="Only show active sessions"),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=100),
    db: AsyncSession = Depends(get_db)
):
    """List all paper trading sessions"""
    query = select(PaperTradingSession)
    
    if active_only:
        query = query.where(PaperTradingSession.is_active == True)
    
    query = query.order_by(PaperTradingSession.created_at.desc())
    query = query.offset(skip).limit(limit)
    
    result = await db.execute(query)
    sessions = result.scalars().all()
    
    return sessions

@router.get("/sessions/{session_id}", response_model=PaperSessionResponse)
async def get_session(
    session_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Get details of a specific paper trading session"""
    session = await paper_service.get_session(db, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session

@router.patch("/sessions/{session_id}", response_model=PaperSessionResponse)
async def update_session(
    session_id: int,
    update_data: PaperSessionUpdate,
    db: AsyncSession = Depends(get_db)
):
    """Update a paper trading session"""
    session = await paper_service.get_session(db, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Update fields
    update_dict = update_data.dict(exclude_unset=True)
    for field, value in update_dict.items():
        setattr(session, field, value)
    
    await db.commit()
    await db.refresh(session)
    
    return session

@router.post("/sessions/{session_id}/orders", response_model=PaperOrderResponse)
async def create_order(
    session_id: int,
    order_data: PaperOrderCreate,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """Create a new paper trading order"""
    # Verify session exists and is active
    session = await paper_service.get_session(db, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if not session.is_active:
        raise HTTPException(status_code=400, detail="Session is not active")
    
    # Add session_id to order data
    order_dict = order_data.dict()
    order_dict['session_id'] = session_id
    order_data = PaperOrderCreate(**order_dict)
    
    try:
        # Execute order
        order = await paper_service.execute_order(db, order_data)
        
        # WebSocket으로 주문 실행 알림
        await ws_manager.broadcast(json.dumps({
            "type": "paper_order_executed",
            "session_id": session_id,
            "order_id": order.id,
            "symbol": order.symbol,
            "side": order.side,
            "quantity": float(order.quantity),
            "price": float(order.price) if order.price else None,
            "status": order.status
        }))
        
        # 백그라운드에서 포지션 업데이트
        background_tasks.add_task(
            paper_service.update_positions,
            db, session_id
        )
        
        return order
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sessions/{session_id}/orders", response_model=List[PaperOrderResponse])
async def list_orders(
    session_id: int,
    status: Optional[str] = Query(None, description="Filter by order status"),
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=100),
    db: AsyncSession = Depends(get_db)
):
    """List orders for a paper trading session"""
    query = select(PaperOrder).where(PaperOrder.session_id == session_id)
    
    if status:
        query = query.where(PaperOrder.status == status)
    if symbol:
        query = query.where(PaperOrder.symbol == symbol)
    
    query = query.order_by(PaperOrder.created_at.desc())
    query = query.offset(skip).limit(limit)
    
    result = await db.execute(query)
    orders = result.scalars().all()
    
    return orders

@router.get("/sessions/{session_id}/positions", response_model=List[PaperPositionResponse])
async def list_positions(
    session_id: int,
    open_only: bool = Query(True, description="Only show open positions"),
    db: AsyncSession = Depends(get_db)
):
    """List positions for a paper trading session"""
    query = select(PaperPosition).where(PaperPosition.session_id == session_id)
    
    if open_only:
        query = query.where(PaperPosition.quantity > 0)
    
    result = await db.execute(query)
    positions = result.scalars().all()
    
    return positions

@router.get("/sessions/{session_id}/balance", response_model=PaperBalanceResponse)
async def get_balance(
    session_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Get current balance for a paper trading session"""
    balance = await paper_service.get_balance(db, session_id)
    if not balance:
        raise HTTPException(status_code=404, detail="Session not found")
    return balance

@router.get("/sessions/{session_id}/performance", response_model=PaperPerformanceMetrics)
async def get_performance(
    session_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Get performance metrics for a paper trading session"""
    metrics = await paper_service.calculate_performance(db, session_id)
    if not metrics:
        raise HTTPException(status_code=404, detail="Session not found")
    return metrics

@router.post("/sessions/{session_id}/reset")
async def reset_session(
    session_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Reset a paper trading session to initial state"""
    session = await paper_service.get_session(db, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Close all open positions
    positions = await db.execute(
        select(PaperPosition).where(
            and_(
                PaperPosition.session_id == session_id,
                PaperPosition.quantity > 0
            )
        )
    )
    open_positions = positions.scalars().all()
    
    for position in open_positions:
        position.quantity = 0
        position.closed_at = datetime.utcnow()
    
    # Cancel all pending orders
    orders = await db.execute(
        select(PaperOrder).where(
            and_(
                PaperOrder.session_id == session_id,
                PaperOrder.status == "pending"
            )
        )
    )
    pending_orders = orders.scalars().all()
    
    for order in pending_orders:
        order.status = "cancelled"
    
    # Reset session balances
    session.current_balance_krw = session.initial_balance_krw
    session.current_balance_usd = session.initial_balance_usd
    session.total_trades = 0
    session.winning_trades = 0
    session.losing_trades = 0
    session.total_pnl = 0
    session.total_fees = 0
    
    await db.commit()
    
    # WebSocket으로 리셋 알림
    await ws_manager.broadcast(json.dumps({
        "type": "paper_session_reset",
        "session_id": session_id
    }))
    
    return {"message": f"Session {session_id} has been reset"}

@router.delete("/sessions/{session_id}")
async def delete_session(
    session_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Delete a paper trading session and all related data"""
    session = await paper_service.get_session(db, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Delete related positions
    from sqlalchemy import delete
    await db.execute(
        delete(PaperPosition).where(PaperPosition.session_id == session_id)
    )
    
    # Delete related orders
    await db.execute(
        delete(PaperOrder).where(PaperOrder.session_id == session_id)
    )
    
    # Delete session
    await db.delete(session)
    await db.commit()
    
    return {"message": f"Session {session_id} has been deleted"}

@router.post("/sessions/{session_id}/close-position")
async def close_position(
    session_id: int,
    symbol: str,
    quantity: Optional[float] = None,
    db: AsyncSession = Depends(get_db)
):
    """Close a position (partially or fully)"""
    # Get current position
    position = await db.execute(
        select(PaperPosition).where(
            and_(
                PaperPosition.session_id == session_id,
                PaperPosition.symbol == symbol,
                PaperPosition.quantity > 0
            )
        )
    )
    position = position.scalar_one_or_none()
    
    if not position:
        raise HTTPException(status_code=404, detail="Position not found")
    
    # Determine quantity to close
    close_quantity = quantity if quantity else position.quantity
    if close_quantity > position.quantity:
        raise HTTPException(status_code=400, detail="Cannot close more than current position")
    
    # Create sell order
    order_data = PaperOrderCreate(
        session_id=session_id,
        symbol=symbol,
        side="sell",
        order_type="market",
        quantity=close_quantity,
        exchange="upbit" if position.side == "buy" else "binance"
    )
    
    try:
        order = await paper_service.execute_order(db, order_data)
        
        # WebSocket으로 포지션 종료 알림
        await ws_manager.broadcast(json.dumps({
            "type": "paper_position_closed",
            "session_id": session_id,
            "symbol": symbol,
            "quantity": float(close_quantity),
            "remaining": float(position.quantity - close_quantity)
        }))
        
        return {
            "message": f"Position closed successfully",
            "order_id": order.id,
            "closed_quantity": close_quantity,
            "remaining_quantity": position.quantity - close_quantity
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sessions/{session_id}/export")
async def export_session_data(
    session_id: int,
    format: str = Query("json", regex="^(json|csv)$"),
    db: AsyncSession = Depends(get_db)
):
    """Export paper trading session data"""
    session = await paper_service.get_session(db, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Get all data
    orders = await db.execute(
        select(PaperOrder).where(PaperOrder.session_id == session_id)
    )
    orders = orders.scalars().all()
    
    positions = await db.execute(
        select(PaperPosition).where(PaperPosition.session_id == session_id)
    )
    positions = positions.scalars().all()
    
    if format == "json":
        return {
            "session": {
                "id": session.id,
                "name": session.name,
                "initial_balance_krw": float(session.initial_balance_krw),
                "current_balance_krw": float(session.current_balance_krw),
                "total_pnl": float(session.total_pnl),
                "total_trades": session.total_trades,
                "winning_trades": session.winning_trades,
                "losing_trades": session.losing_trades
            },
            "orders": [
                {
                    "id": o.id,
                    "symbol": o.symbol,
                    "side": o.side,
                    "quantity": float(o.quantity),
                    "price": float(o.price) if o.price else None,
                    "status": o.status,
                    "created_at": o.created_at.isoformat()
                }
                for o in orders
            ],
            "positions": [
                {
                    "symbol": p.symbol,
                    "side": p.side,
                    "quantity": float(p.quantity),
                    "entry_price": float(p.entry_price),
                    "current_price": float(p.current_price) if p.current_price else None,
                    "pnl": float(p.pnl) if p.pnl else 0,
                    "created_at": p.created_at.isoformat()
                }
                for p in positions
            ]
        }
    else:  # CSV format
        # For CSV, we'll return a simplified flat structure
        import csv
        import io
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write orders
        writer.writerow(["Order ID", "Symbol", "Side", "Quantity", "Price", "Status", "Created At"])
        for o in orders:
            writer.writerow([
                o.id, o.symbol, o.side, o.quantity,
                o.price if o.price else "Market",
                o.status, o.created_at
            ])
        
        content = output.getvalue()
        return {"csv_data": content}