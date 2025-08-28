"""
Tests for DataLoader
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

from backtesting.data_loader import DataLoader


class TestDataLoader:
    """Test suite for DataLoader"""
    
    @pytest.fixture
    def data_loader(self):
        """Create DataLoader instance"""
        return DataLoader(data_dir="data/historical")
    
    def test_initialization(self, data_loader):
        """Test DataLoader initialization"""
        assert data_loader.data_dir == Path("data/historical")
        assert data_loader.upbit_data is None
        assert data_loader.binance_data is None
        assert data_loader.premium_data is None
        assert data_loader.exchange_rate == 1350.0
    
    def test_load_upbit_data(self, data_loader):
        """Test loading Upbit data"""
        try:
            upbit_data = data_loader._load_upbit_data()
            assert isinstance(upbit_data, pd.DataFrame)
            assert len(upbit_data) > 0
            assert 'close' in upbit_data.columns
            assert 'volume' in upbit_data.columns
            assert isinstance(upbit_data.index, pd.DatetimeIndex)
        except FileNotFoundError:
            pytest.skip("No Upbit data files found")
    
    def test_load_binance_data(self, data_loader):
        """Test loading Binance data"""
        try:
            binance_data = data_loader._load_binance_data()
            assert isinstance(binance_data, pd.DataFrame)
            assert len(binance_data) > 0
            assert 'close' in binance_data.columns
            assert 'volume' in binance_data.columns
            assert isinstance(binance_data.index, pd.DatetimeIndex)
        except FileNotFoundError:
            pytest.skip("No Binance data files found")
    
    def test_load_all_data(self, data_loader):
        """Test loading all data"""
        try:
            data = data_loader.load_all_data()
            assert 'upbit' in data
            assert 'binance' in data
            assert 'premium' in data
            
            # Check synchronization
            if all(v is not None for v in data.values()):
                # All datasets should have same index after synchronization
                upbit_index = set(data['upbit'].index)
                binance_index = set(data['binance'].index)
                assert upbit_index == binance_index
        except FileNotFoundError:
            pytest.skip("Data files not found")
    
    def test_calculate_premium(self, data_loader):
        """Test premium calculation"""
        # Create sample data
        dates = pd.date_range(start='2024-01-01', periods=10, freq='1H')
        
        data_loader.upbit_data = pd.DataFrame({
            'close': [100000000] * 10  # 100M KRW
        }, index=dates)
        
        data_loader.binance_data = pd.DataFrame({
            'close': [70000] * 10  # 70k USD
        }, index=dates)
        
        premium_data = data_loader._calculate_premium()
        
        assert isinstance(premium_data, pd.DataFrame)
        assert 'kimchi_premium' in premium_data.columns
        assert len(premium_data) == 10
        
        # Check premium calculation
        expected_premium = ((100000000 - 70000 * 1350) / (70000 * 1350)) * 100
        assert abs(premium_data['kimchi_premium'].iloc[0] - expected_premium) < 0.01
    
    def test_resample(self, data_loader):
        """Test data resampling"""
        # Create sample data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1min')
        
        data_loader.upbit_data = pd.DataFrame({
            'open': np.random.randn(100) + 100,
            'high': np.random.randn(100) + 101,
            'low': np.random.randn(100) + 99,
            'close': np.random.randn(100) + 100,
            'volume': np.random.randn(100) + 10
        }, index=dates)
        
        # Resample to 5 minutes
        resampled = data_loader.resample('5T')
        
        assert 'upbit' in resampled
        assert len(resampled['upbit']) < len(data_loader.upbit_data)
        assert all(col in resampled['upbit'].columns for col in ['open', 'high', 'low', 'close', 'volume'])
    
    def test_get_data_info(self, data_loader):
        """Test data info retrieval"""
        info = data_loader.get_data_info()
        assert isinstance(info, dict)
        
        # If data is loaded, check info structure
        if data_loader.upbit_data is not None:
            assert 'upbit' in info
            assert 'records' in info['upbit']
            assert 'start' in info['upbit']
            assert 'end' in info['upbit']