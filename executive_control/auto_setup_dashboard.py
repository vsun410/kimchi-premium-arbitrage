"""
Automatic Notion Dashboard Setup
자동으로 Notion Executive Dashboard 생성
"""

import os
import sys
import asyncio
from dotenv import load_dotenv
from setup_notion_executive_dashboard import NotionExecutiveDashboardSetup

async def auto_setup():
    """자동 설정"""
    
    # .env 파일 로드
    load_dotenv()
    
    notion_token = os.getenv("NOTION_TOKEN")
    if not notion_token:
        print("[ERROR] NOTION_TOKEN not found")
        return
    
    print("\n[INFO] Starting Automatic Notion Dashboard Setup")
    print("="*50)
    
    # 접근 가능한 빈 페이지 사용
    # URL: https://www.notion.so/25bd547955ef80fab225da25da498940
    parent_page_id = "25bd5479-55ef-80fa-b225-da25da498940"
    
    print(f"[INFO] Using parent page ID: {parent_page_id}")
    print("[INFO] Creating Executive Control Dashboard...")
    
    try:
        # Dashboard 설정
        setup = NotionExecutiveDashboardSetup(notion_token)
        resources = await setup.setup_complete_dashboard(parent_page_id)
        
        print("\n" + "="*50)
        print("[SUCCESS] Dashboard Setup Complete!")
        print("="*50)
        
        # 생성된 리소스 출력
        print("\n[CREATED RESOURCES]:")
        for key, value in resources.items():
            print(f"  - {key}: {value}")
        
        # .env 파일 업데이트
        env_content = f"""# Notion API Token
NOTION_TOKEN={notion_token}

# Database IDs (auto-generated)
NOTION_VISION_DB={resources.get('vision_db', '')}
NOTION_TASKS_DB={resources.get('tasks_db', '')}
NOTION_VALIDATION_DB={resources.get('validation_db', '')}
NOTION_BLOCKS_DB={resources.get('blocks_db', '')}
NOTION_DASHBOARD_PAGE={resources.get('dashboard_page', '')}

# Additional Settings
VALIDATION_THRESHOLD_BUSINESS=0.8
VALIDATION_THRESHOLD_ARCHITECTURE=0.8
VALIDATION_THRESHOLD_DRIFT=0.2
"""
        
        with open('.env', 'w') as f:
            f.write(env_content)
        
        print("\n[INFO] Environment variables updated in .env")
        
        # Dashboard URL
        dashboard_url = f"https://notion.so/{resources['dashboard_page'].replace('-', '')}"
        print(f"\n[DASHBOARD URL]: {dashboard_url}")
        
        # 다음 단계 안내
        print("\n" + "="*50)
        print("[NEXT STEPS]:")
        print("1. Open your Notion dashboard at the URL above")
        print("2. Run: python executive_control/initialize.py")
        print("3. Start using the validation system!")
        print("="*50)
        
        return resources
        
    except Exception as e:
        print(f"\n[ERROR] Failed to create dashboard: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    result = asyncio.run(auto_setup())
    if result:
        print("\n[SUCCESS] Ready to use Executive Control System!")
    else:
        print("\n[FAILED] Please check the error messages above")