"""
Notion 프로젝트를 한국어로 업데이트하는 스크립트
모든 프로젝트 이름, 태스크, 문서를 한국어로 변경
"""

import os
import asyncio
from notion_client import Client
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()


class NotionKoreanUpdater:
    """Notion 콘텐츠를 한국어로 업데이트하는 클래스"""
    
    def __init__(self):
        self.notion = Client(auth=os.getenv("NOTION_TOKEN"))
        
        # 프로젝트 한글 이름 매핑
        self.project_names_kr = {
            "trading_core": {
                "name": "트레이딩 코어 - 초저지연 실행 엔진",
                "emoji": "⚡",
                "description": "마이크로초 단위 주문 실행을 위한 고성능 트레이딩 인프라",
                "components": ["주문 실행", "시장 데이터 처리", "지연 모니터", "연결 풀"]
            },
            "ml_engine": {
                "name": "ML 엔진 - 퀀트 리서치 및 실행 플랫폼",
                "emoji": "🤖",
                "description": "기관급 머신러닝 인프라와 알고리즘 트레이딩 시스템",
                "components": ["피처 엔지니어링", "모델 학습", "추론 엔진", "백테스팅"]
            },
            "dashboard": {
                "name": "대시보드 - 전문 트레이딩 터미널",
                "emoji": "📊",
                "description": "블룸버그 터미널급 실시간 트레이딩 대시보드",
                "components": ["실시간 차트", "포트폴리오 뷰", "리스크 메트릭", "알림 시스템"]
            },
            "risk_management": {
                "name": "리스크 관리 - 자동화된 위험 통제",
                "emoji": "🛡️",
                "description": "실시간 포지션 모니터링과 자동화된 위험 관리 시스템",
                "components": ["포지션 사이징", "리스크 한도", "드로다운 제어", "긴급 정지"]
            }
        }
        
        # 태스크 한글 템플릿
        self.task_templates_kr = {
            "주문 실행": "주문 실행 모듈 구현",
            "시장 데이터 처리": "시장 데이터 핸들러 구현",
            "지연 모니터": "지연시간 모니터링 시스템 구현",
            "연결 풀": "WebSocket 연결 풀 구현",
            "피처 엔지니어링": "특징 추출 파이프라인 구현",
            "모델 학습": "ML 모델 학습 시스템 구현",
            "추론 엔진": "실시간 추론 엔진 구현",
            "백테스팅": "백테스팅 엔진 구현",
            "실시간 차트": "실시간 차트 컴포넌트 구현",
            "포트폴리오 뷰": "포트폴리오 대시보드 구현",
            "리스크 메트릭": "리스크 지표 계산 시스템 구현",
            "알림 시스템": "실시간 알림 시스템 구현",
            "포지션 사이징": "포지션 크기 계산 로직 구현",
            "리스크 한도": "리스크 한도 관리 시스템 구현",
            "드로다운 제어": "최대 손실 제어 시스템 구현",
            "긴급 정지": "긴급 정지 메커니즘 구현"
        }
        
        # 상태 한글 매핑
        self.status_kr = {
            "Backlog": "백로그",
            "Todo": "할 일",
            "In Progress": "진행 중",
            "In Review": "검토 중",
            "Done": "완료",
            "Blocked": "차단됨"
        }
        
        # 우선순위 한글 매핑
        self.priority_kr = {
            "Critical": "긴급",
            "High": "높음",
            "Medium": "보통",
            "Low": "낮음"
        }
        
    async def update_all_projects(self):
        """모든 프로젝트를 한국어로 업데이트"""
        
        print("="*60)
        print("      Notion 프로젝트 한국어 변환 시작")
        print("="*60)
        print()
        
        # Load configuration
        try:
            with open("multi_project_config.json", "r", encoding="utf-8") as f:
                config = json.load(f)
        except FileNotFoundError:
            print("[ERROR] multi_project_config.json 파일을 찾을 수 없습니다")
            print("[INFO] 먼저 프로젝트 설정을 완료해주세요")
            return False
        
        # Update each project
        notion_resources = config.get("notion_resources", {}).get("projects", {})
        
        for project_key in self.project_names_kr.keys():
            if project_key not in notion_resources:
                continue
                
            kr_config = self.project_names_kr[project_key]
            project_resources = notion_resources[project_key]
            
            print(f"[INFO] {kr_config['name']} 업데이트 중...")
            
            # Update project page title
            try:
                page_id = project_resources.get("page")
                if page_id:
                    self.notion.pages.update(
                        page_id=page_id,
                        properties={},
                        icon={"emoji": kr_config["emoji"]},
                        # Page title update requires different approach
                    )
                    print(f"  [OK] 프로젝트 페이지 아이콘 업데이트")
            except Exception as e:
                print(f"  [WARNING] 페이지 업데이트 실패: {e}")
            
            # Update tasks database
            try:
                tasks_db_id = project_resources.get("tasks_db")
                if tasks_db_id:
                    await self.update_tasks_to_korean(tasks_db_id, kr_config)
                    print(f"  [OK] 태스크 데이터베이스 업데이트")
            except Exception as e:
                print(f"  [WARNING] 태스크 DB 업데이트 실패: {e}")
            
            # Update documentation database
            try:
                docs_db_id = project_resources.get("docs_db")
                if docs_db_id:
                    await self.update_docs_to_korean(docs_db_id, kr_config)
                    print(f"  [OK] 문서 데이터베이스 업데이트")
            except Exception as e:
                print(f"  [WARNING] 문서 DB 업데이트 실패: {e}")
        
        # Update shared databases
        print()
        print("[INFO] 공유 데이터베이스 업데이트 중...")
        
        shared_dbs = config.get("notion_resources", {}).get("shared_databases", {})
        
        # Update ADR database
        if shared_dbs.get("adr"):
            try:
                self.notion.databases.update(
                    database_id=shared_dbs["adr"],
                    title=[{"text": {"content": "📐 아키텍처 결정 기록 (ADR)"}}]
                )
                print("  [OK] ADR 데이터베이스 제목 업데이트")
            except Exception as e:
                print(f"  [WARNING] ADR DB 업데이트 실패: {e}")
        
        # Update Dependencies database
        if shared_dbs.get("dependencies"):
            try:
                self.notion.databases.update(
                    database_id=shared_dbs["dependencies"],
                    title=[{"text": {"content": "🔗 프로젝트 간 의존성"}}]
                )
                print("  [OK] 의존성 데이터베이스 제목 업데이트")
            except Exception as e:
                print(f"  [WARNING] 의존성 DB 업데이트 실패: {e}")
        
        # Update Research database
        if shared_dbs.get("research"):
            try:
                self.notion.databases.update(
                    database_id=shared_dbs["research"],
                    title=[{"text": {"content": "📚 공유 연구 자료"}}]
                )
                print("  [OK] 연구 데이터베이스 제목 업데이트")
            except Exception as e:
                print(f"  [WARNING] 연구 DB 업데이트 실패: {e}")
        
        print()
        print("="*60)
        print("✅ 한국어 변환 완료!")
        print("="*60)
        
        # Save Korean configuration
        kr_config_file = "multi_project_config_kr.json"
        config["language"] = "ko"
        config["project_names"] = self.project_names_kr
        
        with open(kr_config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        print(f"\n[INFO] 한국어 설정이 {kr_config_file}에 저장되었습니다")
        
        return True
    
    async def update_tasks_to_korean(self, db_id: str, kr_config: dict):
        """태스크 데이터베이스를 한국어로 업데이트"""
        
        # Update database title
        try:
            self.notion.databases.update(
                database_id=db_id,
                title=[{"text": {"content": f"📋 {kr_config['name']} - 태스크"}}]
            )
        except Exception as e:
            print(f"    [WARNING] DB 제목 업데이트 실패: {e}")
        
        # Get all tasks
        try:
            tasks = self.notion.databases.query(database_id=db_id)
            
            for task in tasks.get("results", []):
                task_title = task["properties"].get("Task", {}).get("title", [])
                if task_title and len(task_title) > 0:
                    current_title = task_title[0].get("text", {}).get("content", "")
                    
                    # Translate common patterns
                    kr_title = current_title
                    kr_title = kr_title.replace("Implement", "구현:")
                    kr_title = kr_title.replace("Write tests for", "테스트 작성:")
                    kr_title = kr_title.replace("module", "모듈")
                    kr_title = kr_title.replace("Order Execution", "주문 실행")
                    kr_title = kr_title.replace("Market Data Handler", "시장 데이터 처리")
                    kr_title = kr_title.replace("Latency Monitor", "지연 모니터")
                    kr_title = kr_title.replace("Connection Pool", "연결 풀")
                    kr_title = kr_title.replace("Feature Engineering", "피처 엔지니어링")
                    kr_title = kr_title.replace("Model Training", "모델 학습")
                    kr_title = kr_title.replace("Inference Engine", "추론 엔진")
                    kr_title = kr_title.replace("Backtesting", "백테스팅")
                    kr_title = kr_title.replace("Real-time Charts", "실시간 차트")
                    kr_title = kr_title.replace("Portfolio View", "포트폴리오 뷰")
                    kr_title = kr_title.replace("Risk Metrics", "리스크 메트릭")
                    kr_title = kr_title.replace("Alert System", "알림 시스템")
                    kr_title = kr_title.replace("Position Sizing", "포지션 사이징")
                    kr_title = kr_title.replace("Risk Limits", "리스크 한도")
                    kr_title = kr_title.replace("Drawdown Control", "드로다운 제어")
                    kr_title = kr_title.replace("Emergency Stop", "긴급 정지")
                    
                    # Update task
                    try:
                        self.notion.pages.update(
                            page_id=task["id"],
                            properties={
                                "Task": {"title": [{"text": {"content": kr_title}}]}
                            }
                        )
                    except Exception as e:
                        pass  # Skip individual task errors
                        
        except Exception as e:
            print(f"    [WARNING] 태스크 업데이트 실패: {e}")
    
    async def update_docs_to_korean(self, db_id: str, kr_config: dict):
        """문서 데이터베이스를 한국어로 업데이트"""
        
        # Update database title
        try:
            self.notion.databases.update(
                database_id=db_id,
                title=[{"text": {"content": f"📚 {kr_config['name']} - 문서"}}]
            )
        except Exception as e:
            print(f"    [WARNING] DB 제목 업데이트 실패: {e}")
        
        # Get all documents
        try:
            docs = self.notion.databases.query(database_id=db_id)
            
            for doc in docs.get("results", []):
                doc_title = doc["properties"].get("Title", {}).get("title", [])
                if doc_title and len(doc_title) > 0:
                    current_title = doc_title[0].get("text", {}).get("content", "")
                    
                    # Translate document titles
                    kr_title = current_title
                    kr_title = kr_title.replace("Architecture", "아키텍처")
                    kr_title = kr_title.replace("API Documentation", "API 문서")
                    kr_title = kr_title.replace("User Guide", "사용자 가이드")
                    kr_title = kr_title.replace("Trading Core - Ultra-Low Latency Execution Engine", "트레이딩 코어")
                    kr_title = kr_title.replace("ML Engine - Quantitative Research & Execution Platform", "ML 엔진")
                    kr_title = kr_title.replace("Dashboard - Professional Trading Terminal", "대시보드")
                    kr_title = kr_title.replace("Risk Management - Automated Risk Controls", "리스크 관리")
                    
                    # Update document
                    try:
                        self.notion.pages.update(
                            page_id=doc["id"],
                            properties={
                                "Title": {"title": [{"text": {"content": kr_title}}]}
                            }
                        )
                    except Exception as e:
                        pass  # Skip individual doc errors
                        
        except Exception as e:
            print(f"    [WARNING] 문서 업데이트 실패: {e}")


async def main():
    """메인 실행 함수"""
    updater = NotionKoreanUpdater()
    await updater.update_all_projects()


if __name__ == "__main__":
    asyncio.run(main())