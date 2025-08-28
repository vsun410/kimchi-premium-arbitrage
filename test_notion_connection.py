#!/usr/bin/env python3
"""
Quick Notion API Connection Test
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from notion_client import Client

# Set UTF-8 encoding for Windows
if sys.platform == 'win32':
    import locale
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

def test_connection():
    """Test Notion API connection"""
    
    # Load environment variables from .env.notion
    env_notion_path = Path('.env.notion')
    if env_notion_path.exists():
        load_dotenv(env_notion_path)
        print("[OK] .env.notion 파일을 로드했습니다.")
    else:
        print("[ERROR] .env.notion 파일을 찾을 수 없습니다.")
        return False
    
    # Get API token
    token = os.getenv('NOTION_TOKEN')
    if not token:
        print("[ERROR] NOTION_TOKEN을 찾을 수 없습니다.")
        return False
    
    print(f"[TOKEN] 토큰 발견: {token[:20]}...{token[-10:]}")
    
    try:
        # Initialize Notion client
        client = Client(auth=token)
        
        # Test connection by getting user info
        print("\n[TEST] 연결 테스트 중...")
        user_info = client.users.me()
        
        print("\n[SUCCESS] Notion API 연결 성공!")
        print(f"   Bot 타입: {user_info.get('type', 'Unknown')}")
        print(f"   Bot ID: {user_info.get('id', 'Unknown')}")
        print(f"   Bot 이름: {user_info.get('name', 'Unknown')}")
        
        # Try to search for databases
        print("\n[SEARCH] 데이터베이스 검색 중...")
        search_response = client.search(
            filter={"value": "database", "property": "object"},
            page_size=5
        )
        
        if search_response['results']:
            print(f"\n[FOUND] {len(search_response['results'])}개의 데이터베이스 발견:")
            for idx, db in enumerate(search_response['results'], 1):
                db_title = "Untitled"
                if db.get('title'):
                    if len(db['title']) > 0:
                        db_title = db['title'][0]['plain_text']
                print(f"   {idx}. {db_title}")
                print(f"      ID: {db['id']}")
        else:
            print("\n   공유된 데이터베이스가 없습니다.")
            print("   Notion에서 Integration을 데이터베이스에 초대해주세요.")
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] 연결 실패: {str(e)}")
        if "Invalid token" in str(e):
            print("   토큰이 유효하지 않습니다. 올바른 토큰인지 확인해주세요.")
        elif "Unauthorized" in str(e):
            print("   권한이 없습니다. Integration 설정을 확인해주세요.")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("Notion API 연결 테스트")
    print("=" * 50)
    
    success = test_connection()
    
    print("\n" + "=" * 50)
    if success:
        print("[SUCCESS] 테스트 성공! Notion API를 사용할 수 있습니다.")
    else:
        print("[WARNING] 테스트 실패. 위의 오류를 확인해주세요.")
    print("=" * 50)