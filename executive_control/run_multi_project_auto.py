"""
자동 실행 Multi-Project Setup
기존 대시보드 페이지를 사용하여 4개 프로젝트 워크스페이스 자동 생성
"""

import os
import sys
import asyncio
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from executive_control.setup_multi_project_dashboard import MultiProjectDashboardSetup
from executive_control.notion_project_manager import NotionProjectManager, initialize_project_with_sample_data


async def run_auto_setup():
    """자동으로 4개 프로젝트 워크스페이스 생성"""
    
    print("="*70)
    print("      4개 독립 프로젝트 관리 시스템 구축 (자동 모드)")
    print("="*70)
    print()
    
    # Check Notion token
    notion_token = os.getenv("NOTION_TOKEN")
    if not notion_token:
        print("[ERROR] NOTION_TOKEN을 찾을 수 없습니다")
        print("[INFO] .env 파일에 NOTION_TOKEN이 있는지 확인하세요")
        return False
    
    print("[OK] Notion API 토큰 확인됨")
    
    # Use existing dashboard page
    parent_page = os.getenv("NOTION_DASHBOARD_PAGE")
    if not parent_page:
        print("[ERROR] NOTION_DASHBOARD_PAGE를 찾을 수 없습니다")
        print("[INFO] .env 파일에 NOTION_DASHBOARD_PAGE가 있는지 확인하세요")
        return False
    
    print(f"[OK] 기존 대시보드 페이지 사용: {parent_page}")
    print()
    
    print("="*50)
    print("STEP 1: 프로젝트 워크스페이스 생성")
    print("="*50)
    
    try:
        # Create workspace structure
        setup = MultiProjectDashboardSetup(notion_token)
        resources = await setup.setup_complete_workspace(parent_page)
        
        print("\n[SUCCESS] 4개 프로젝트 워크스페이스 생성 완료!")
        print()
        print("생성된 프로젝트:")
        print("  1. Trading Core - Ultra-Low Latency Execution Engine")
        print("  2. ML Engine - Quantitative Research & Execution Platform")
        print("  3. Dashboard - Professional Trading Terminal")
        print("  4. Risk Management - Automated Risk Controls")
        print()
        
        # Add sample data
        print("="*50)
        print("STEP 2: 샘플 태스크 및 문서 추가")
        print("="*50)
        
        manager = NotionProjectManager(notion_token, "multi_project_config.json")
        
        projects = [
            ('trading_core', 'Trading Core'),
            ('ml_engine', 'ML Engine'),
            ('dashboard', 'Dashboard'),
            ('risk_management', 'Risk Management')
        ]
        
        for project_key, project_name in projects:
            print(f"[INFO] {project_name} 초기화 중...")
            await initialize_project_with_sample_data(manager, project_key)
            print(f"[OK] {project_name} 샘플 데이터 추가 완료")
        
        print()
        print("="*50)
        print("STEP 3: 프로젝트 간 의존성 설정")
        print("="*50)
        
        # Create cross-project dependencies
        dependencies = [
            {
                "from": "Dashboard",
                "to": "Trading Core", 
                "type": "API",
                "desc": "실시간 거래 데이터 및 주문 상태"
            },
            {
                "from": "Trading Core",
                "to": "ML Engine",
                "type": "API", 
                "desc": "거래 결정을 위한 ML 신호"
            },
            {
                "from": "Trading Core",
                "to": "Risk Management",
                "type": "Service",
                "desc": "포지션 검증 및 리스크 체크"
            },
            {
                "from": "Dashboard",
                "to": "Risk Management", 
                "type": "API",
                "desc": "리스크 메트릭 및 알림"
            },
            {
                "from": "ML Engine",
                "to": "Risk Management",
                "type": "Data",
                "desc": "모델 예측 및 신뢰도 점수"
            }
        ]
        
        print("[INFO] 프로젝트 간 의존성 추가 중...")
        
        # Add to dependencies database (if it exists in resources)
        if 'dependencies_db' in resources:
            from notion_client import Client
            notion = Client(auth=notion_token)
            
            for dep in dependencies:
                print(f"  - {dep['from']} → {dep['to']} ({dep['type']})")
                
                try:
                    notion.pages.create(
                        parent={"database_id": resources['dependencies_db']},
                        properties={
                            "Name": {"title": [{"text": {"content": f"{dep['from']} → {dep['to']}"}}]},
                            "From Project": {"select": {"name": dep['from']}},
                            "To Project": {"select": {"name": dep['to']}},
                            "Type": {"select": {"name": dep['type']}},
                            "Description": {"rich_text": [{"text": {"content": dep['desc']}}]}
                        }
                    )
                except Exception as e:
                    print(f"  [WARNING] 의존성 추가 실패: {e}")
        
        print()
        print("="*70)
        print("✅ 모든 설정 완료!")
        print("="*70)
        print()
        print("📋 생성된 리소스:")
        print(f"  - 마스터 대시보드: {resources.get('master_page', 'N/A')}")
        print(f"  - Trading Core 페이지: {resources.get('trading_core_page', 'N/A')}")
        print(f"  - ML Engine 페이지: {resources.get('ml_engine_page', 'N/A')}")
        print(f"  - Dashboard 페이지: {resources.get('dashboard_page', 'N/A')}")
        print(f"  - Risk Management 페이지: {resources.get('risk_management_page', 'N/A')}")
        print()
        print("🎯 다음 단계:")
        print("  1. Notion에서 생성된 페이지들 확인")
        print("  2. 각 프로젝트별 태스크 추가 및 관리")
        print("  3. 아키텍처 결정 사항 기록")
        print("  4. 프로젝트 진행 상황 모니터링")
        
        # Save configuration
        config = {
            "master_page": resources.get('master_page'),
            "projects": {
                "trading_core": resources.get('trading_core_page'),
                "ml_engine": resources.get('ml_engine_page'),
                "dashboard": resources.get('dashboard_page'),
                "risk_management": resources.get('risk_management_page')
            },
            "databases": {
                "adr": resources.get('adr_db'),
                "dependencies": resources.get('dependencies_db'),
                "research": resources.get('research_db')
            }
        }
        
        import json
        with open("multi_project_workspace.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        
        print()
        print("[INFO] 설정이 multi_project_workspace.json에 저장되었습니다")
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] 설정 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """메인 진입점"""
    try:
        success = asyncio.run(run_auto_setup())
        if not success:
            sys.exit(1)
        print("\n[SUCCESS] 4개 독립 프로젝트 관리 시스템 구축 완료!")
    except KeyboardInterrupt:
        print("\n[INFO] 사용자에 의해 취소됨")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] 실행 실패: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()