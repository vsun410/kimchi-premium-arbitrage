"""
Dynamic Hedge Module for Kimchi Premium Arbitrage System

This module implements the trend exploitation strategy that converts
basic arbitrage into a sophisticated dynamic hedging system.
"""

from .trend_analysis import TrendAnalysisEngine
from .position_manager import DynamicPositionManager
from .pattern_detector import TrianglePatternDetector
from .reverse_premium import ReversePremiumHandler

__all__ = [
    'TrendAnalysisEngine',
    'DynamicPositionManager',
    'TrianglePatternDetector',
    'ReversePremiumHandler'
]