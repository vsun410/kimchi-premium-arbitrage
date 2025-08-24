"""
Python 버전 호환성 모듈
Python 3.9+ 지원을 위한 호환성 처리
"""

import sys
from typing import Union

# Python 3.10+ 타입 힌트 호환성
if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    # Python 3.9 호환
    TypeAlias = type

# asyncio 호환성
if sys.version_info >= (3, 10):
    from asyncio import TaskGroup
else:
    # Python 3.9용 대체 구현
    class TaskGroup:
        """Python 3.9 호환 TaskGroup"""
        def __init__(self):
            self.tasks = []
        
        async def __aenter__(self):
            return self
        
        async def __aexit__(self, *args):
            pass
        
        def create_task(self, coro):
            import asyncio
            task = asyncio.create_task(coro)
            self.tasks.append(task)
            return task

# dataclasses 호환성
if sys.version_info >= (3, 10):
    from dataclasses import KW_ONLY
else:
    # Python 3.9 호환
    KW_ONLY = None

__all__ = ['TypeAlias', 'TaskGroup', 'KW_ONLY']