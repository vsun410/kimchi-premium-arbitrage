"""
Notion Governance Integration - Executive Control System의 Notion 연동
Vision Guardian과 Task Orchestrator를 Notion과 연결하여 프로젝트 거버넌스 구현
"""

import os
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import json
import asyncio
from notion_client import Client
from dataclasses import asdict

from vision_guardian import VisionGuardian, ValidationResult, ProjectVision
from task_orchestrator import TaskOrchestrator, TaskSpec, TaskStatus, CodeBlock


class NotionGovernanceIntegration:
    """
    Notion을 Executive Board로 활용하는 통합 시스템
    프로젝트 비전 유지와 코드 품질 관리를 자동화
    """
    
    def __init__(self, notion_token: str, workspace_config: Dict):
        """
        Args:
            notion_token: Notion API 토큰
            workspace_config: Notion 워크스페이스 설정
                - vision_db: 비전 데이터베이스 ID
                - tasks_db: 작업 데이터베이스 ID
                - validation_db: 검증 결과 데이터베이스 ID
                - dashboard_page: 대시보드 페이지 ID
        """
        self.notion = Client(auth=notion_token)
        self.config = workspace_config
        
        # 핵심 컴포넌트 초기화
        self.vision_guardian = None
        self.task_orchestrator = None
        
        # 데이터베이스 ID
        self.vision_db = workspace_config.get('vision_db')
        self.tasks_db = workspace_config.get('tasks_db')
        self.validation_db = workspace_config.get('validation_db')
        self.dashboard_page = workspace_config.get('dashboard_page')
        
    async def initialize_governance(self):
        """거버넌스 시스템 초기화"""
        
        # 1. Notion에서 프로젝트 비전 로드
        vision_data = await self._load_vision_from_notion()
        
        # 2. Vision Guardian 초기화
        self.vision_guardian = VisionGuardian(
            prd_path=vision_data['prd_path'],
            architecture_path=vision_data['architecture_path']
        )
        
        # 3. Task Orchestrator 초기화
        self.task_orchestrator = TaskOrchestrator(self.vision_guardian)
        
        # 4. 기존 작업 로드
        await self._sync_tasks_from_notion()
        
        print("✅ Governance system initialized")
    
    async def _load_vision_from_notion(self) -> Dict:
        """Notion에서 프로젝트 비전 문서 로드"""
        
        # 비전 데이터베이스에서 최신 비전 문서 조회
        response = self.notion.databases.query(
            database_id=self.vision_db,
            filter={
                "property": "Type",
                "select": {"equals": "Core Vision"}
            },
            sorts=[{"property": "Last Edited", "direction": "descending"}]
        )
        
        if not response['results']:
            # 기본 비전 생성
            return await self._create_default_vision()
        
        # 최신 비전 페이지에서 정보 추출
        vision_page = response['results'][0]
        
        # 페이지 콘텐츠 가져오기
        blocks = self.notion.blocks.children.list(vision_page['id'])
        
        # PRD와 아키텍처 경로 추출 (실제로는 Notion에서 직접 읽을 수도 있음)
        vision_data = {
            'prd_path': './executive_control/prd.md',
            'architecture_path': './executive_control/architecture.json',
            'page_id': vision_page['id']
        }
        
        return vision_data
    
    async def _create_default_vision(self) -> Dict:
        """기본 비전 문서 생성"""
        
        # PRD 생성
        prd_content = """
# Kimchi Premium Trading System PRD

## Vision
실시간 김치프리미엄 차익거래 자동화 시스템

## Core Objectives
1. 실시간 김치프리미엄 차익거래 자동화
2. 리스크 중립적 헤지 포지션 유지
3. 24/7 무중단 운영 시스템
4. ML 기반 객관적 진입/청산 시그널

## Red Lines (절대 금지)
1. 동시 양방향 포지션 한도 초과
2. 단일 거래 자본금 10% 초과
3. 수동 개입 필요한 로직
4. 테스트되지 않은 코드 배포
"""
        
        # PRD 파일 저장
        os.makedirs('./executive_control', exist_ok=True)
        with open('./executive_control/prd.md', 'w', encoding='utf-8') as f:
            f.write(prd_content)
        
        # 아키텍처 명세 생성
        architecture = {
            "components": {
                "trading_core": {
                    "interfaces": ["WebSocket", "REST API"],
                    "constraints": ["latency < 100ms", "stateless"]
                },
                "ml_engine": {
                    "interfaces": ["gRPC", "Message Queue"],
                    "constraints": ["prediction_time < 500ms"]
                }
            }
        }
        
        with open('./executive_control/architecture.json', 'w', encoding='utf-8') as f:
            json.dump(architecture, f, indent=2)
        
        # Notion에 비전 페이지 생성
        vision_page = await self._create_vision_page_in_notion(prd_content)
        
        return {
            'prd_path': './executive_control/prd.md',
            'architecture_path': './executive_control/architecture.json',
            'page_id': vision_page['id']
        }
    
    async def _create_vision_page_in_notion(self, prd_content: str) -> Dict:
        """Notion에 비전 페이지 생성"""
        
        page = self.notion.pages.create(
            parent={"database_id": self.vision_db},
            properties={
                "Title": {"title": [{"text": {"content": "Project Vision - Kimchi Premium Trading"}}]},
                "Type": {"select": {"name": "Core Vision"}},
                "Status": {"select": {"name": "Active"}}
            },
            children=[
                {
                    "object": "block",
                    "type": "heading_1",
                    "heading_1": {"rich_text": [{"text": {"content": "🎯 Project Vision"}}]}
                },
                {
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {"rich_text": [{"text": {"content": prd_content[:2000]}}]}
                }
            ]
        )
        
        return page
    
    async def _sync_tasks_from_notion(self):
        """Notion에서 기존 작업 동기화"""
        
        if not self.tasks_db:
            return
        
        # 작업 데이터베이스 조회
        response = self.notion.databases.query(
            database_id=self.tasks_db,
            filter={
                "property": "Status",
                "select": {"does_not_equal": "Completed"}
            }
        )
        
        for page in response['results']:
            props = page['properties']
            
            # TaskSpec 생성
            task = TaskSpec(
                task_id=self._get_property_value(props.get('Task ID')),
                title=self._get_property_value(props.get('Title')),
                description=self._get_property_value(props.get('Description', '')),
                component=self._get_property_value(props.get('Component', 'general')),
                business_requirements=[],  # 페이지 콘텐츠에서 로드
                technical_requirements=[],
                acceptance_criteria=[],
                estimated_effort=self._get_property_value(props.get('Effort', 8)),
                priority=self._get_property_value(props.get('Priority', 'medium'))
            )
            
            # Task Orchestrator에 등록
            self.task_orchestrator.tasks[task.task_id] = task
            
            # 상태 매핑
            status_str = self._get_property_value(props.get('Status', 'pending'))
            status_map = {
                'pending': TaskStatus.PENDING,
                'in_progress': TaskStatus.IN_PROGRESS,
                'in_review': TaskStatus.IN_REVIEW,
                'approved': TaskStatus.APPROVED,
                'completed': TaskStatus.COMPLETED
            }
            self.task_orchestrator.task_status[task.task_id] = status_map.get(
                status_str.lower(), TaskStatus.PENDING
            )
    
    def _get_property_value(self, prop: Any) -> Any:
        """Notion 속성 값 추출"""
        if not prop:
            return None
            
        prop_type = prop.get('type')
        
        if prop_type == 'title':
            texts = prop.get('title', [])
            return ' '.join([t['plain_text'] for t in texts]) if texts else ''
        elif prop_type == 'rich_text':
            texts = prop.get('rich_text', [])
            return ' '.join([t['plain_text'] for t in texts]) if texts else ''
        elif prop_type == 'select':
            return prop['select']['name'] if prop.get('select') else ''
        elif prop_type == 'number':
            return prop.get('number', 0)
        
        return ''
    
    async def submit_requirement(self, requirement: str) -> List[str]:
        """
        새로운 요구사항 제출 및 작업 분해
        
        Args:
            requirement: 자연어 요구사항
            
        Returns:
            생성된 작업 ID 리스트
        """
        
        # 1. 요구사항을 작업으로 분해
        tasks = self.task_orchestrator.decompose_requirement(requirement)
        
        # 2. 각 작업을 Notion에 생성
        task_ids = []
        for task in tasks:
            # Task Orchestrator에 등록
            self.task_orchestrator.tasks[task.task_id] = task
            self.task_orchestrator.task_status[task.task_id] = TaskStatus.PENDING
            
            # Notion에 페이지 생성
            await self._create_task_in_notion(task)
            task_ids.append(task.task_id)
        
        # 3. 대시보드 업데이트
        await self._update_dashboard()
        
        return task_ids
    
    async def _create_task_in_notion(self, task: TaskSpec) -> Dict:
        """Notion에 작업 페이지 생성"""
        
        page = self.notion.pages.create(
            parent={"database_id": self.tasks_db},
            properties={
                "Task ID": {"title": [{"text": {"content": task.task_id}}]},
                "Title": {"rich_text": [{"text": {"content": task.title}}]},
                "Component": {"select": {"name": task.component}},
                "Priority": {"select": {"name": task.priority}},
                "Status": {"select": {"name": "Pending"}},
                "Effort": {"number": task.estimated_effort}
            },
            children=self._create_task_blocks(task)
        )
        
        return page
    
    def _create_task_blocks(self, task: TaskSpec) -> List[Dict]:
        """작업 페이지 블록 생성"""
        blocks = []
        
        # 설명
        blocks.extend([
            {
                "object": "block",
                "type": "heading_2",
                "heading_2": {"rich_text": [{"text": {"content": "📝 Description"}}]}
            },
            {
                "object": "block",
                "type": "paragraph",
                "paragraph": {"rich_text": [{"text": {"content": task.description}}]}
            }
        ])
        
        # 비즈니스 요구사항
        if task.business_requirements:
            blocks.append({
                "object": "block",
                "type": "heading_2",
                "heading_2": {"rich_text": [{"text": {"content": "💼 Business Requirements"}}]}
            })
            for req in task.business_requirements:
                blocks.append({
                    "object": "block",
                    "type": "bulleted_list_item",
                    "bulleted_list_item": {"rich_text": [{"text": {"content": req}}]}
                })
        
        # 기술 요구사항
        if task.technical_requirements:
            blocks.append({
                "object": "block",
                "type": "heading_2",
                "heading_2": {"rich_text": [{"text": {"content": "🔧 Technical Requirements"}}]}
            })
            for req in task.technical_requirements:
                blocks.append({
                    "object": "block",
                    "type": "bulleted_list_item",
                    "bulleted_list_item": {"rich_text": [{"text": {"content": req}}]}
                })
        
        # 수락 기준
        if task.acceptance_criteria:
            blocks.append({
                "object": "block",
                "type": "heading_2",
                "heading_2": {"rich_text": [{"text": {"content": "✅ Acceptance Criteria"}}]}
            })
            for criteria in task.acceptance_criteria:
                blocks.append({
                    "object": "block",
                    "type": "to_do",
                    "to_do": {
                        "rich_text": [{"text": {"content": criteria}}],
                        "checked": False
                    }
                })
        
        return blocks
    
    async def validate_code(self, 
                           task_id: str, 
                           code: str) -> Tuple[bool, str]:
        """
        Claude Code가 작성한 코드 검증
        
        Args:
            task_id: 작업 ID
            code: 검증할 코드
            
        Returns:
            (승인 여부, 피드백)
        """
        
        # 1. Vision Guardian으로 검증
        approved, feedback = self.task_orchestrator.submit_code_for_validation(
            task_id, code
        )
        
        # 2. 검증 결과를 Notion에 기록
        await self._record_validation_result(task_id, code, approved, feedback)
        
        # 3. 작업 상태 업데이트
        if approved:
            await self._update_task_status(task_id, "Approved")
        else:
            await self._update_task_status(task_id, "Rejected")
        
        # 4. 대시보드 업데이트
        await self._update_dashboard()
        
        return approved, feedback
    
    async def _record_validation_result(self, 
                                       task_id: str, 
                                       code: str,
                                       approved: bool,
                                       feedback: str):
        """검증 결과를 Notion에 기록"""
        
        # 검증 데이터베이스에 새 페이지 생성
        page = self.notion.pages.create(
            parent={"database_id": self.validation_db},
            properties={
                "Task ID": {"title": [{"text": {"content": task_id}}]},
                "Status": {"select": {"name": "Approved" if approved else "Rejected"}},
                "Timestamp": {"date": {"start": datetime.now().isoformat()}},
                "Validator": {"select": {"name": "Vision Guardian"}}
            },
            children=[
                {
                    "object": "block",
                    "type": "heading_2",
                    "heading_2": {"rich_text": [{"text": {"content": "📊 Validation Result"}}]}
                },
                {
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {"rich_text": [{"text": {"content": feedback[:2000]}}]}
                },
                {
                    "object": "block",
                    "type": "heading_2",
                    "heading_2": {"rich_text": [{"text": {"content": "💻 Submitted Code"}}]}
                },
                {
                    "object": "block",
                    "type": "code",
                    "code": {
                        "rich_text": [{"text": {"content": code[:2000]}}],
                        "language": "python"
                    }
                }
            ]
        )
    
    async def _update_task_status(self, task_id: str, status: str):
        """Notion에서 작업 상태 업데이트"""
        
        # 작업 페이지 찾기
        response = self.notion.databases.query(
            database_id=self.tasks_db,
            filter={
                "property": "Task ID",
                "title": {"equals": task_id}
            }
        )
        
        if response['results']:
            page_id = response['results'][0]['id']
            
            # 상태 업데이트
            self.notion.pages.update(
                page_id=page_id,
                properties={
                    "Status": {"select": {"name": status}}
                }
            )
    
    async def _update_dashboard(self):
        """Executive Dashboard 업데이트"""
        
        if not self.dashboard_page:
            return
        
        # 현재 상태 집계
        report = self.task_orchestrator.generate_task_report()
        
        # 대시보드 콘텐츠 생성
        dashboard_blocks = self._create_dashboard_blocks(report)
        
        # 기존 블록 삭제 (간단하게 구현)
        existing_blocks = self.notion.blocks.children.list(self.dashboard_page)
        for block in existing_blocks['results']:
            self.notion.blocks.delete(block['id'])
        
        # 새 블록 추가
        for block in dashboard_blocks:
            self.notion.blocks.children.append(
                self.dashboard_page,
                children=[block]
            )
    
    def _create_dashboard_blocks(self, report: Dict) -> List[Dict]:
        """대시보드 블록 생성"""
        
        blocks = []
        
        # 헤더
        blocks.append({
            "object": "block",
            "type": "heading_1",
            "heading_1": {
                "rich_text": [{"text": {"content": "🎯 Executive Control Dashboard"}}]
            }
        })
        
        # 업데이트 시간
        blocks.append({
            "object": "block",
            "type": "paragraph",
            "paragraph": {
                "rich_text": [{
                    "text": {
                        "content": f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    }
                }]
            }
        })
        
        # 통계 섹션
        blocks.append({
            "object": "block",
            "type": "heading_2",
            "heading_2": {"rich_text": [{"text": {"content": "📊 Project Statistics"}}]}
        })
        
        # 작업 상태 분포
        status_text = "Task Status:\n"
        for status, count in report['status_distribution'].items():
            status_text += f"• {status}: {count}\n"
        
        blocks.append({
            "object": "block",
            "type": "paragraph",
            "paragraph": {"rich_text": [{"text": {"content": status_text}}]}
        })
        
        # 평균 점수
        if report['average_scores']['business'] > 0:
            score_text = f"""
Vision Alignment Scores:
• Business Logic: {report['average_scores']['business']:.1%}
• Architecture: {report['average_scores']['architecture']:.1%}  
• Code Drift: {report['average_scores']['drift']:.1%}
"""
            blocks.append({
                "object": "block",
                "type": "callout",
                "callout": {
                    "rich_text": [{"text": {"content": score_text}}],
                    "icon": {"emoji": "📈"}
                }
            })
        
        # 컴포넌트별 분포
        blocks.append({
            "object": "block",
            "type": "heading_2",
            "heading_2": {"rich_text": [{"text": {"content": "🔧 Component Distribution"}}]}
        })
        
        comp_text = ""
        for component, count in report['component_distribution'].items():
            comp_text += f"• {component}: {count} tasks\n"
        
        blocks.append({
            "object": "block",
            "type": "paragraph",
            "paragraph": {"rich_text": [{"text": {"content": comp_text}}]}
        })
        
        # 코드 블록 정보
        blocks.append({
            "object": "block",
            "type": "heading_2",
            "heading_2": {"rich_text": [{"text": {"content": "🧩 Code Blocks"}}]}
        })
        
        blocks.append({
            "object": "block",
            "type": "paragraph",
            "paragraph": {
                "rich_text": [{
                    "text": {
                        "content": f"Total Blocks: {report['total_blocks']}\n"
                                  f"Replaceable: {report['replaceable_blocks']}"
                    }
                }]
            }
        })
        
        return blocks
    
    async def get_pending_tasks(self) -> List[TaskSpec]:
        """대기 중인 작업 목록 반환"""
        
        pending_tasks = []
        for task_id, status in self.task_orchestrator.task_status.items():
            if status == TaskStatus.PENDING:
                task = self.task_orchestrator.tasks.get(task_id)
                if task:
                    pending_tasks.append(task)
        
        return pending_tasks
    
    async def get_replaceable_blocks(self) -> List[CodeBlock]:
        """교체 가능한 코드 블록 목록 반환"""
        return self.task_orchestrator.get_replaceable_blocks()
    
    async def replace_code_block(self,
                                block_id: str,
                                new_code: str,
                                new_version: str) -> Tuple[bool, str]:
        """
        코드 블록 교체
        
        Args:
            block_id: 블록 ID
            new_code: 새 코드
            new_version: 새 버전
            
        Returns:
            (성공 여부, 메시지)
        """
        
        # 1. Vision Guardian으로 새 코드 검증
        success, message = self.task_orchestrator.replace_block(
            block_id, new_code, new_version
        )
        
        # 2. 성공 시 Notion에 기록
        if success:
            await self._record_block_replacement(block_id, new_version)
        
        # 3. 대시보드 업데이트
        await self._update_dashboard()
        
        return success, message
    
    async def _record_block_replacement(self, block_id: str, new_version: str):
        """블록 교체 기록"""
        
        # 검증 데이터베이스에 교체 기록 추가
        page = self.notion.pages.create(
            parent={"database_id": self.validation_db},
            properties={
                "Task ID": {"title": [{"text": {"content": f"Block Replace: {block_id}"}}]},
                "Status": {"select": {"name": "Block Replaced"}},
                "Timestamp": {"date": {"start": datetime.now().isoformat()}},
                "Validator": {"select": {"name": "Vision Guardian"}}
            },
            children=[
                {
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [{
                            "text": {
                                "content": f"Block {block_id} replaced with version {new_version}"
                            }
                        }]
                    }
                }
            ]
        )


# 사용 예시
async def main():
    # Notion 토큰과 설정
    notion_token = os.getenv("NOTION_TOKEN")
    workspace_config = {
        'vision_db': os.getenv("NOTION_VISION_DB"),
        'tasks_db': os.getenv("NOTION_TASKS_DB"),
        'validation_db': os.getenv("NOTION_VALIDATION_DB"),
        'dashboard_page': os.getenv("NOTION_DASHBOARD_PAGE")
    }
    
    # 통합 시스템 초기화
    governance = NotionGovernanceIntegration(notion_token, workspace_config)
    await governance.initialize_governance()
    
    # 1. 새로운 요구사항 제출
    requirement = """
    실시간 김치프리미엄 모니터링 대시보드를 구현해주세요.
    업비트와 바이낸스의 가격 차이를 실시간으로 표시하고,
    ML 모델의 예측 신호를 함께 보여줘야 합니다.
    """
    
    task_ids = await governance.submit_requirement(requirement)
    print(f"Created {len(task_ids)} tasks")
    
    # 2. 코드 검증
    sample_code = """
    async def display_kimchi_premium(self):
        upbit_price = await self.get_upbit_price()
        binance_price = await self.get_binance_price()
        premium = (upbit_price - binance_price) / binance_price * 100
        return {'premium': premium}
    """
    
    if task_ids:
        approved, feedback = await governance.validate_code(task_ids[0], sample_code)
        print(f"Validation result: {'Approved' if approved else 'Rejected'}")
        print(feedback)
    
    # 3. 대기 중인 작업 확인
    pending = await governance.get_pending_tasks()
    print(f"Pending tasks: {len(pending)}")
    
    # 4. 교체 가능한 블록 확인
    blocks = await governance.get_replaceable_blocks()
    print(f"Replaceable blocks: {len(blocks)}")


if __name__ == "__main__":
    asyncio.run(main())