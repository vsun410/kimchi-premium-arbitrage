"""
Notion API Setup Helper
Notion API 설정 도우미
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from notion_client import Client
import json
from typing import Dict, Optional

class NotionSetup:
    """Notion 설정 및 검증 도우미"""
    
    def __init__(self):
        self.env_path = Path('.env')
        self.env_notion_path = Path('.env.notion')
        self.load_environment()
        
    def load_environment(self):
        """환경 변수 로드"""
        if self.env_path.exists():
            load_dotenv(self.env_path)
            print("✅ .env 파일을 찾았습니다.")
        elif self.env_notion_path.exists():
            print("📝 .env.notion 파일을 찾았습니다.")
            print("   .env로 복사하고 실제 값을 입력해주세요.")
            self.create_env_from_template()
        else:
            print("❌ .env 파일이 없습니다.")
            self.create_env_template()
    
    def create_env_from_template(self):
        """템플릿에서 .env 파일 생성"""
        response = input("\n.env.notion을 .env로 복사하시겠습니까? (y/n): ")
        if response.lower() == 'y':
            import shutil
            shutil.copy(self.env_notion_path, self.env_path)
            print("✅ .env 파일이 생성되었습니다.")
            print("\n다음 단계:")
            print("1. .env 파일을 열어주세요")
            print("2. NOTION_TOKEN에 실제 API 키를 입력하세요")
            print("3. 각 데이터베이스 ID를 입력하세요")
            print("4. 이 스크립트를 다시 실행하세요")
    
    def create_env_template(self):
        """새 .env 템플릿 생성"""
        template = """# Notion API Configuration
NOTION_TOKEN=secret_YOUR_TOKEN_HERE

# Database IDs (32자리 ID)
NOTION_TASKS_DB=
NOTION_ARCHITECTURE_DB=
NOTION_TEST_REPORTS_DB=
NOTION_METRICS_DB=
NOTION_RISKS_DB=

# Settings
NOTION_SYNC_INTERVAL=60
NOTION_AUTO_SYNC=true
"""
        with open(self.env_path, 'w') as f:
            f.write(template)
        print(f"✅ {self.env_path} 템플릿이 생성되었습니다.")
    
    def validate_token(self) -> bool:
        """Notion API 토큰 검증"""
        token = os.getenv('NOTION_TOKEN')
        
        if not token or token == 'secret_YOUR_TOKEN_HERE':
            print("\n❌ NOTION_TOKEN이 설정되지 않았습니다.")
            print("\n설정 방법:")
            print("1. https://www.notion.so/my-integrations 접속")
            print("2. 'New integration' 클릭")
            print("3. 이름 입력 (예: Kimchi Trading Bot)")
            print("4. 워크스페이스 선택")
            print("5. 'Submit' 클릭")
            print("6. 'Internal Integration Token' 복사")
            print("7. .env 파일의 NOTION_TOKEN에 붙여넣기")
            return False
        
        try:
            client = Client(auth=token)
            # 토큰 테스트 - 사용자 정보 가져오기
            response = client.users.me()
            print(f"✅ Notion 연결 성공!")
            print(f"   봇 이름: {response.get('name', 'Unknown')}")
            print(f"   봇 ID: {response.get('id', 'Unknown')}")
            return True
        except Exception as e:
            print(f"❌ Notion 연결 실패: {str(e)}")
            print("\n가능한 원인:")
            print("1. 잘못된 토큰")
            print("2. 네트워크 문제")
            print("3. Notion API 서버 문제")
            return False
    
    def find_databases(self) -> Dict[str, str]:
        """워크스페이스의 데이터베이스 찾기"""
        token = os.getenv('NOTION_TOKEN')
        if not token or not self.validate_token():
            return {}
        
        client = Client(auth=token)
        databases = {}
        
        try:
            print("\n🔍 데이터베이스 검색 중...")
            # 모든 데이터베이스 검색
            response = client.search(
                filter={"value": "database", "property": "object"}
            )
            
            for result in response['results']:
                db_title = "Untitled"
                if result.get('title'):
                    db_title = result['title'][0]['plain_text'] if result['title'] else "Untitled"
                elif result.get('child_database'):
                    db_title = result['child_database'].get('title', 'Untitled')
                
                db_id = result['id']
                databases[db_title] = db_id
                print(f"   📊 {db_title}: {db_id}")
            
            if not databases:
                print("   데이터베이스를 찾을 수 없습니다.")
                print("\n데이터베이스 공유 방법:")
                print("1. Notion에서 데이터베이스 페이지 열기")
                print("2. 우측 상단 'Share' 클릭")
                print("3. 'Invite' 섹션에서 Integration 선택")
                print("4. 생성한 Integration 선택 후 'Invite'")
            
            return databases
            
        except Exception as e:
            print(f"❌ 데이터베이스 검색 실패: {str(e)}")
            return {}
    
    def setup_databases(self, databases: Dict[str, str]):
        """찾은 데이터베이스를 .env에 매핑"""
        if not databases:
            return
        
        print("\n📝 데이터베이스 매핑:")
        
        # 자동 매핑 시도
        mapping = {
            'Tasks': 'NOTION_TASKS_DB',
            'Architecture': 'NOTION_ARCHITECTURE_DB',
            'Test Reports': 'NOTION_TEST_REPORTS_DB',
            'Metrics': 'NOTION_METRICS_DB',
            'Risks': 'NOTION_RISKS_DB',
            'Decisions': 'NOTION_DECISIONS_DB'
        }
        
        env_updates = []
        
        for db_name, db_id in databases.items():
            for keyword, env_var in mapping.items():
                if keyword.lower() in db_name.lower():
                    env_updates.append(f"{env_var}={db_id}")
                    print(f"   {env_var} → {db_name}")
                    break
        
        if env_updates:
            response = input("\n.env 파일에 이 설정을 추가하시겠습니까? (y/n): ")
            if response.lower() == 'y':
                self.update_env_file(env_updates)
                print("✅ .env 파일이 업데이트되었습니다.")
    
    def update_env_file(self, updates: list):
        """env 파일 업데이트"""
        # 기존 내용 읽기
        lines = []
        if self.env_path.exists():
            with open(self.env_path, 'r') as f:
                lines = f.readlines()
        
        # 업데이트 적용
        for update in updates:
            key, value = update.split('=')
            found = False
            for i, line in enumerate(lines):
                if line.startswith(key + '='):
                    lines[i] = f"{update}\n"
                    found = True
                    break
            if not found:
                lines.append(f"{update}\n")
        
        # 파일 쓰기
        with open(self.env_path, 'w') as f:
            f.writelines(lines)
    
    def test_connection(self):
        """전체 연결 테스트"""
        print("\n" + "="*50)
        print("🧪 Notion API 연결 테스트")
        print("="*50)
        
        # 1. 토큰 검증
        if not self.validate_token():
            return
        
        # 2. 데이터베이스 찾기
        databases = self.find_databases()
        
        # 3. 데이터베이스 매핑
        if databases:
            self.setup_databases(databases)
        
        # 4. 최종 상태 출력
        self.print_status()
    
    def print_status(self):
        """현재 설정 상태 출력"""
        print("\n" + "="*50)
        print("📊 현재 설정 상태")
        print("="*50)
        
        # 필수 환경 변수 체크
        required_vars = [
            'NOTION_TOKEN',
            'NOTION_TASKS_DB',
            'NOTION_METRICS_DB'
        ]
        
        all_set = True
        for var in required_vars:
            value = os.getenv(var, '')
            if value and value != 'secret_YOUR_TOKEN_HERE':
                print(f"✅ {var}: 설정됨")
            else:
                print(f"❌ {var}: 미설정")
                all_set = False
        
        if all_set:
            print("\n🎉 모든 필수 설정이 완료되었습니다!")
            print("   이제 Notion 통합을 사용할 수 있습니다.")
        else:
            print("\n⚠️ 일부 설정이 필요합니다.")
            print("   위의 미설정 항목을 .env 파일에 추가해주세요.")
    
    def create_sample_task(self):
        """테스트용 Task 생성"""
        token = os.getenv('NOTION_TOKEN')
        tasks_db = os.getenv('NOTION_TASKS_DB')
        
        if not token or not tasks_db:
            print("❌ NOTION_TOKEN과 NOTION_TASKS_DB가 필요합니다.")
            return
        
        client = Client(auth=token)
        
        try:
            response = client.pages.create(
                parent={"database_id": tasks_db},
                properties={
                    "Title": {"title": [{"text": {"content": "Test Task from Python"}}]},
                    "Status": {"status": {"name": "Not Started"}},
                    "Priority": {"select": {"name": "P2"}}
                }
            )
            print(f"✅ 테스트 Task 생성 성공!")
            print(f"   URL: https://notion.so/{response['id'].replace('-', '')}")
            
        except Exception as e:
            print(f"❌ Task 생성 실패: {str(e)}")


def main():
    """메인 실행 함수"""
    print("""
╔════════════════════════════════════════════════════════════╗
║             Notion API Setup Helper                       ║
║                                                            ║
║  1. Notion Integration 생성                              ║
║  2. API Token 설정                                       ║  
║  3. Database 연결                                        ║
║  4. 연결 테스트                                          ║
╚════════════════════════════════════════════════════════════╝
    """)
    
    setup = NotionSetup()
    
    while True:
        print("\n메뉴를 선택하세요:")
        print("1. 연결 테스트")
        print("2. 데이터베이스 찾기")
        print("3. 샘플 Task 생성")
        print("4. 현재 상태 확인")
        print("5. 종료")
        
        choice = input("\n선택 (1-5): ")
        
        if choice == '1':
            setup.test_connection()
        elif choice == '2':
            setup.find_databases()
        elif choice == '3':
            setup.create_sample_task()
        elif choice == '4':
            setup.print_status()
        elif choice == '5':
            print("👋 종료합니다.")
            break
        else:
            print("❌ 잘못된 선택입니다.")


if __name__ == "__main__":
    main()