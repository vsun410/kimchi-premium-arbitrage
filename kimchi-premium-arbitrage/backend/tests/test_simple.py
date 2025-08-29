"""
Simple test to verify CI environment
"""

def test_simple_addition():
    """Test that basic Python works"""
    assert 1 + 1 == 2

def test_simple_string():
    """Test string operations"""
    assert "hello" + " world" == "hello world"

def test_simple_list():
    """Test list operations"""
    my_list = [1, 2, 3]
    assert len(my_list) == 3
    assert sum(my_list) == 6