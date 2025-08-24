"""
Simple test to verify module import works
"""

def test_import_dynamic_hedge():
    """Test that dynamic_hedge module can be imported"""
    try:
        from dynamic_hedge import TrendAnalysisEngine
        from dynamic_hedge import DynamicPositionManager
        from dynamic_hedge import TrianglePatternDetector
        from dynamic_hedge import ReversePremiumHandler
        assert True
    except ImportError as e:
        assert False, f"Failed to import: {e}"

def test_basic_initialization():
    """Test basic class initialization"""
    from dynamic_hedge.trend_analysis import TrendAnalysisEngine
    engine = TrendAnalysisEngine()
    assert engine is not None
    assert engine.window_size == 100