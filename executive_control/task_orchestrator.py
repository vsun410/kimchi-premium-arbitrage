"""
Task Orchestrator - 요구사항을 검증 가능한 작업으로 분해
Claude Code의 작업을 Vision Guardian이 검증할 수 있는 형태로 구조화
"""

import json
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from dataclasses import dataclass, field
import uuid
from enum import Enum
import re

@dataclass
class TaskSpec:
    """작업 명세"""
    task_id: str
    title: str
    description: str
    component: str  # 어느 컴포넌트를 수정하는지
    business_requirements: List[str]
    technical_requirements: List[str]
    acceptance_criteria: List[str]
    estimated_effort: int  # hours
    priority: str
    dependencies: List[str] = field(default_factory=list)
    test_requirements: List[str] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)
    
@dataclass
class CodeBlock:
    """코드 블록 단위"""
    block_id: str
    component: str
    function_name: str
    description: str
    inputs: Dict[str, str]
    outputs: Dict[str, str]
    business_logic: str
    can_be_replaced: bool = True
    version: str = "1.0.0"

class TaskStatus(Enum):
    """작업 상태"""
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    IN_REVIEW = "in_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    COMPLETED = "completed"


class TaskOrchestrator:
    """
    요구사항을 실행 가능한 작업으로 분해하고 관리
    각 작업이 비전과 일치하도록 보장
    """
    
    def __init__(self, vision_guardian):
        self.vision_guardian = vision_guardian
        self.tasks = {}  # task_id -> TaskSpec
        self.code_blocks = {}  # block_id -> CodeBlock
        self.task_status = {}  # task_id -> TaskStatus
        self.execution_history = []
        
    def decompose_requirement(self, requirement: str) -> List[TaskSpec]:
        """
        고수준 요구사항을 실행 가능한 작업으로 분해
        
        Args:
            requirement: 자연어 요구사항
            
        Returns:
            TaskSpec 리스트
        """
        # 요구사항 분석
        parsed = self._parse_requirement(requirement)
        
        # 컴포넌트별로 작업 분할
        tasks = []
        for component in parsed['affected_components']:
            task = self._create_task_for_component(
                component=component,
                requirement=requirement,
                parsed_data=parsed
            )
            tasks.append(task)
            
        # 의존성 설정
        self._set_task_dependencies(tasks)
        
        # 우선순위 설정
        self._prioritize_tasks(tasks)
        
        return tasks
    
    def _parse_requirement(self, requirement: str) -> Dict:
        """요구사항 파싱"""
        parsed = {
            'affected_components': [],
            'business_goals': [],
            'technical_needs': [],
            'keywords': []
        }
        
        # 컴포넌트 식별
        component_keywords = {
            'trading_core': ['거래', '주문', 'trading', 'order', 'execution'],
            'ml_engine': ['머신러닝', 'ML', '예측', 'prediction', 'model'],
            'dashboard': ['대시보드', '화면', 'UI', 'dashboard', 'display'],
            'risk_manager': ['리스크', '위험', 'risk', 'limit', 'exposure'],
            'data_pipeline': ['데이터', '수집', 'data', 'collection', 'pipeline']
        }
        
        requirement_lower = requirement.lower()
        for component, keywords in component_keywords.items():
            if any(keyword in requirement_lower for keyword in keywords):
                parsed['affected_components'].append(component)
        
        # 비즈니스 목표 추출
        business_patterns = [
            r'목표[는:]?\s*([^\.]+)',
            r'위해[서]?\s*([^\.]+)',
            r'구현[하고자]?\s*([^\.]+)'
        ]
        
        for pattern in business_patterns:
            matches = re.findall(pattern, requirement)
            parsed['business_goals'].extend(matches)
        
        return parsed
    
    def _create_task_for_component(self, 
                                  component: str, 
                                  requirement: str,
                                  parsed_data: Dict) -> TaskSpec:
        """컴포넌트별 작업 생성"""
        
        task_id = f"{component[:2].upper()}-{str(uuid.uuid4())[:8]}"
        
        # 컴포넌트별 템플릿
        templates = {
            'trading_core': {
                'title': f"Implement trading logic for: {requirement[:50]}",
                'technical': ['WebSocket connection', 'Order management', 'State handling'],
                'tests': ['Unit tests for order logic', 'Integration tests with exchange']
            },
            'ml_engine': {
                'title': f"Develop ML model for: {requirement[:50]}",
                'technical': ['Feature engineering', 'Model training', 'Prediction pipeline'],
                'tests': ['Model accuracy tests', 'Performance benchmarks']
            },
            'dashboard': {
                'title': f"Create dashboard for: {requirement[:50]}",
                'technical': ['Real-time updates', 'Responsive design', 'WebSocket client'],
                'tests': ['UI component tests', 'E2E tests']
            }
        }
        
        template = templates.get(component, {})
        
        return TaskSpec(
            task_id=task_id,
            title=template.get('title', f"Task for {component}"),
            description=requirement,
            component=component,
            business_requirements=parsed_data['business_goals'],
            technical_requirements=template.get('technical', []),
            acceptance_criteria=[
                f"Code passes Vision Guardian validation",
                f"All tests pass",
                f"Documentation updated"
            ],
            estimated_effort=8,  # Default 8 hours
            priority="medium",
            test_requirements=template.get('tests', [])
        )
    
    def _set_task_dependencies(self, tasks: List[TaskSpec]):
        """작업 간 의존성 설정"""
        # 간단한 규칙: data_pipeline -> ml_engine -> trading_core -> dashboard
        dependency_order = ['data_pipeline', 'ml_engine', 'trading_core', 'risk_manager', 'dashboard']
        
        for i, current_comp in enumerate(dependency_order[1:], 1):
            for task in tasks:
                if task.component == current_comp:
                    # 이전 컴포넌트의 작업들을 의존성으로 추가
                    for prev_comp in dependency_order[:i]:
                        for dep_task in tasks:
                            if dep_task.component == prev_comp:
                                task.dependencies.append(dep_task.task_id)
    
    def _prioritize_tasks(self, tasks: List[TaskSpec]):
        """작업 우선순위 설정"""
        # 의존성이 없는 작업이 높은 우선순위
        for task in tasks:
            if not task.dependencies:
                task.priority = "high"
            elif len(task.dependencies) > 2:
                task.priority = "low"
            else:
                task.priority = "medium"
    
    def create_code_block(self, task: TaskSpec) -> CodeBlock:
        """
        작업을 교체 가능한 코드 블록으로 변환
        
        Args:
            task: 작업 명세
            
        Returns:
            CodeBlock: 교체 가능한 코드 블록
        """
        block = CodeBlock(
            block_id=f"BLK-{task.task_id}",
            component=task.component,
            function_name=self._generate_function_name(task.title),
            description=task.description,
            inputs=self._extract_inputs(task),
            outputs=self._extract_outputs(task),
            business_logic=self._summarize_logic(task),
            can_be_replaced=True,
            version="1.0.0"
        )
        
        self.code_blocks[block.block_id] = block
        return block
    
    def _generate_function_name(self, title: str) -> str:
        """타이틀에서 함수명 생성"""
        # 간단한 변환: 공백을 언더스코어로
        clean_title = re.sub(r'[^a-zA-Z0-9\s]', '', title)
        words = clean_title.lower().split()[:3]  # 처음 3단어만
        return '_'.join(words)
    
    def _extract_inputs(self, task: TaskSpec) -> Dict[str, str]:
        """작업에서 입력 추출"""
        inputs = {}
        
        # 컴포넌트별 기본 입력
        component_inputs = {
            'trading_core': {
                'signal': 'TradingSignal',
                'market_data': 'MarketData'
            },
            'ml_engine': {
                'features': 'np.ndarray',
                'model_params': 'Dict'
            },
            'dashboard': {
                'data': 'Dict',
                'user_config': 'UserConfig'
            }
        }
        
        return component_inputs.get(task.component, {'data': 'Any'})
    
    def _extract_outputs(self, task: TaskSpec) -> Dict[str, str]:
        """작업에서 출력 추출"""
        # 컴포넌트별 기본 출력
        component_outputs = {
            'trading_core': {'execution_result': 'ExecutionResult'},
            'ml_engine': {'prediction': 'Prediction'},
            'dashboard': {'render_data': 'Dict'}
        }
        
        return component_outputs.get(task.component, {'result': 'Any'})
    
    def _summarize_logic(self, task: TaskSpec) -> str:
        """비즈니스 로직 요약"""
        return f"""
        Component: {task.component}
        Business Requirements: {', '.join(task.business_requirements[:3])}
        Technical Requirements: {', '.join(task.technical_requirements[:3])}
        """
    
    def submit_code_for_validation(self, 
                                  task_id: str, 
                                  code: str) -> Tuple[bool, str]:
        """
        작성된 코드를 Vision Guardian에 제출하여 검증
        
        Args:
            task_id: 작업 ID
            code: 작성된 코드
            
        Returns:
            (승인 여부, 피드백 메시지)
        """
        if task_id not in self.tasks:
            return False, f"Task {task_id} not found"
        
        task = self.tasks[task_id]
        task_spec_dict = {
            'task_id': task.task_id,
            'component': task.component,
            'description': task.description,
            'requirements': task.business_requirements + task.technical_requirements
        }
        
        # Vision Guardian으로 검증
        result = self.vision_guardian.validate_code(code, task_spec_dict)
        
        # 상태 업데이트
        if result.approved:
            self.task_status[task_id] = TaskStatus.APPROVED
            self._record_execution(task_id, "APPROVED", result)
        else:
            self.task_status[task_id] = TaskStatus.REJECTED
            self._record_execution(task_id, "REJECTED", result)
        
        # 피드백 생성
        feedback = self.vision_guardian.generate_feedback(result)
        
        return result.approved, feedback
    
    def _record_execution(self, task_id: str, status: str, result):
        """실행 기록 저장"""
        self.execution_history.append({
            'task_id': task_id,
            'timestamp': datetime.now().isoformat(),
            'status': status,
            'scores': {
                'business': result.business_score,
                'architecture': result.architecture_score,
                'drift': result.drift_score
            },
            'violations': result.violations
        })
    
    def get_replaceable_blocks(self) -> List[CodeBlock]:
        """교체 가능한 코드 블록 목록 반환"""
        return [
            block for block in self.code_blocks.values()
            if block.can_be_replaced
        ]
    
    def replace_block(self, 
                     block_id: str, 
                     new_code: str,
                     new_version: str) -> Tuple[bool, str]:
        """
        코드 블록 교체
        
        Args:
            block_id: 블록 ID
            new_code: 새로운 코드
            new_version: 새 버전
            
        Returns:
            (성공 여부, 메시지)
        """
        if block_id not in self.code_blocks:
            return False, f"Block {block_id} not found"
        
        block = self.code_blocks[block_id]
        
        # 새 코드 검증
        task_spec = {
            'task_id': block_id,
            'component': block.component,
            'description': block.description
        }
        
        result = self.vision_guardian.validate_code(new_code, task_spec)
        
        if result.approved:
            # 버전 업데이트
            block.version = new_version
            return True, f"Block {block_id} successfully replaced with version {new_version}"
        else:
            feedback = self.vision_guardian.generate_feedback(result)
            return False, f"Replacement rejected:\n{feedback}"
    
    def generate_task_report(self) -> Dict:
        """작업 현황 리포트 생성"""
        total_tasks = len(self.tasks)
        
        status_counts = {}
        for status in TaskStatus:
            status_counts[status.value] = sum(
                1 for s in self.task_status.values() 
                if s == status
            )
        
        # 컴포넌트별 통계
        component_stats = {}
        for task in self.tasks.values():
            if task.component not in component_stats:
                component_stats[task.component] = 0
            component_stats[task.component] += 1
        
        # 평균 점수 계산 (실행 기록에서)
        avg_scores = {'business': 0, 'architecture': 0, 'drift': 0}
        if self.execution_history:
            for record in self.execution_history:
                for key in avg_scores:
                    avg_scores[key] += record['scores'][key]
            
            num_records = len(self.execution_history)
            for key in avg_scores:
                avg_scores[key] /= num_records
        
        return {
            'total_tasks': total_tasks,
            'status_distribution': status_counts,
            'component_distribution': component_stats,
            'average_scores': avg_scores,
            'total_blocks': len(self.code_blocks),
            'replaceable_blocks': len(self.get_replaceable_blocks()),
            'execution_history_count': len(self.execution_history),
            'timestamp': datetime.now().isoformat()
        }
    
    def export_to_notion_format(self) -> List[Dict]:
        """Notion 데이터베이스 형식으로 내보내기"""
        notion_pages = []
        
        for task_id, task in self.tasks.items():
            status = self.task_status.get(task_id, TaskStatus.PENDING)
            
            # 최근 실행 기록 찾기
            recent_execution = None
            for record in reversed(self.execution_history):
                if record['task_id'] == task_id:
                    recent_execution = record
                    break
            
            page = {
                'properties': {
                    'Task ID': {'title': [{'text': {'content': task.task_id}}]},
                    'Title': {'rich_text': [{'text': {'content': task.title}}]},
                    'Component': {'select': {'name': task.component}},
                    'Status': {'select': {'name': status.value}},
                    'Priority': {'select': {'name': task.priority}},
                    'Effort (hours)': {'number': task.estimated_effort},
                    'Dependencies': {
                        'multi_select': [{'name': dep} for dep in task.dependencies]
                    }
                },
                'children': [
                    {
                        'object': 'block',
                        'type': 'heading_2',
                        'heading_2': {
                            'rich_text': [{'text': {'content': 'Description'}}]
                        }
                    },
                    {
                        'object': 'block',
                        'type': 'paragraph',
                        'paragraph': {
                            'rich_text': [{'text': {'content': task.description}}]
                        }
                    },
                    {
                        'object': 'block',
                        'type': 'heading_2',
                        'heading_2': {
                            'rich_text': [{'text': {'content': 'Requirements'}}]
                        }
                    }
                ]
            }
            
            # 요구사항 추가
            for req in task.business_requirements + task.technical_requirements:
                page['children'].append({
                    'object': 'block',
                    'type': 'bulleted_list_item',
                    'bulleted_list_item': {
                        'rich_text': [{'text': {'content': req}}]
                    }
                })
            
            # 검증 결과 추가 (있는 경우)
            if recent_execution:
                page['children'].extend([
                    {
                        'object': 'block',
                        'type': 'heading_2',
                        'heading_2': {
                            'rich_text': [{'text': {'content': 'Validation Results'}}]
                        }
                    },
                    {
                        'object': 'block',
                        'type': 'paragraph',
                        'paragraph': {
                            'rich_text': [{
                                'text': {
                                    'content': f"Business Score: {recent_execution['scores']['business']:.1%}\n"
                                              f"Architecture Score: {recent_execution['scores']['architecture']:.1%}\n"
                                              f"Drift Score: {recent_execution['scores']['drift']:.1%}"
                                }
                            }]
                        }
                    }
                ])
            
            notion_pages.append(page)
        
        return notion_pages


# 사용 예시
if __name__ == "__main__":
    from vision_guardian import VisionGuardian
    
    # Vision Guardian 초기화
    guardian = VisionGuardian(
        prd_path="./prd.md",
        architecture_path="./architecture.json"
    )
    
    # Task Orchestrator 초기화
    orchestrator = TaskOrchestrator(guardian)
    
    # 요구사항 분해
    requirement = """
    실시간 김치프리미엄 모니터링 대시보드를 구현해주세요.
    업비트와 바이낸스의 가격 차이를 실시간으로 표시하고,
    ML 모델의 예측 신호를 함께 보여줘야 합니다.
    """
    
    tasks = orchestrator.decompose_requirement(requirement)
    
    # 작업 등록
    for task in tasks:
        orchestrator.tasks[task.task_id] = task
        orchestrator.task_status[task.task_id] = TaskStatus.PENDING
        
        # 코드 블록 생성
        block = orchestrator.create_code_block(task)
        print(f"Created block: {block.block_id} for {block.component}")
    
    # 리포트 생성
    report = orchestrator.generate_task_report()
    print(json.dumps(report, indent=2))