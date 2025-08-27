"""
Executive Control System for Kimchi Premium Trading
프로젝트 비전을 지키며 코드 드리프트를 방지하는 거버넌스 시스템
"""

from .vision_guardian import VisionGuardian, ValidationResult, ProjectVision
from .task_orchestrator import TaskOrchestrator, TaskSpec, TaskStatus, CodeBlock
from .notion_governance_integration import NotionGovernanceIntegration
from .claude_code_interceptor import ClaudeCodeInterceptor

__version__ = "1.0.0"
__all__ = [
    "VisionGuardian",
    "ValidationResult",
    "ProjectVision",
    "TaskOrchestrator",
    "TaskSpec",
    "TaskStatus",
    "CodeBlock",
    "NotionGovernanceIntegration",
    "ClaudeCodeInterceptor"
]