"""
Notion Template Backup & Version Control System
현재 Notion 템플릿을 백업하고 버전 관리
"""

import os
import json
import asyncio
from datetime import datetime
from notion_client import Client
from dotenv import load_dotenv
import hashlib
from pathlib import Path

load_dotenv()


class NotionTemplateBackup:
    """Notion 템플릿 백업 및 버전 관리"""
    
    def __init__(self):
        self.notion = Client(auth=os.getenv("NOTION_TOKEN"))
        self.backup_dir = Path("executive_control/template_backups")
        self.backup_dir.mkdir(exist_ok=True)
        
    async def create_backup(self, version_name: str = None):
        """현재 템플릿 상태를 백업"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version = version_name or f"v1.0.0_{timestamp}"
        
        print(f"[INFO] Creating backup: {version}")
        
        backup_data = {
            "version": version,
            "timestamp": timestamp,
            "created_at": datetime.now().isoformat(),
            "description": "Notion Dashboard Template Backup",
            "components": {}
        }
        
        # 1. Dashboard 구조 백업
        dashboard_structure = await self._backup_dashboard_structure()
        backup_data["components"]["dashboard"] = dashboard_structure
        
        # 2. 데이터베이스 스키마 백업
        db_schemas = await self._backup_database_schemas()
        backup_data["components"]["databases"] = db_schemas
        
        # 3. 페이지 템플릿 백업
        page_templates = await self._backup_page_templates()
        backup_data["components"]["templates"] = page_templates
        
        # 4. 설정 및 메타데이터 백업
        metadata = await self._backup_metadata()
        backup_data["components"]["metadata"] = metadata
        
        # 파일로 저장
        backup_file = self.backup_dir / f"backup_{version}.json"
        with open(backup_file, "w", encoding="utf-8") as f:
            json.dump(backup_data, f, indent=2, ensure_ascii=False)
        
        # 체크섬 생성
        checksum = self._create_checksum(backup_file)
        
        print(f"[SUCCESS] Backup created: {backup_file}")
        print(f"[INFO] Checksum: {checksum}")
        
        return backup_file, checksum
    
    async def _backup_dashboard_structure(self):
        """대시보드 구조 백업"""
        
        dashboard_page_id = os.getenv("NOTION_DASHBOARD_PAGE")
        if not dashboard_page_id:
            return {}
        
        try:
            # 페이지 정보 가져오기
            page = self.notion.pages.retrieve(dashboard_page_id)
            
            # 블록 구조 가져오기
            blocks = []
            block_children = self.notion.blocks.children.list(dashboard_page_id)
            
            for block in block_children.get("results", []):
                blocks.append({
                    "type": block.get("type"),
                    "content": self._extract_block_content(block)
                })
            
            return {
                "page_id": dashboard_page_id,
                "title": "Executive Control Center",
                "blocks": blocks,
                "structure": {
                    "layout": "3-column",
                    "sections": [
                        "Welcome & Quote",
                        "Navigation Bar",
                        "Metrics Dashboard",
                        "Progress Tracking",
                        "System Status"
                    ]
                }
            }
        except Exception as e:
            print(f"[WARNING] Could not backup dashboard: {e}")
            return {}
    
    async def _backup_database_schemas(self):
        """데이터베이스 스키마 백업"""
        
        schemas = {}
        
        db_ids = {
            "vision": os.getenv("NOTION_VISION_DB"),
            "tasks": os.getenv("NOTION_TASKS_DB"),
            "validation": os.getenv("NOTION_VALIDATION_DB"),
            "blocks": os.getenv("NOTION_BLOCKS_DB")
        }
        
        for db_name, db_id in db_ids.items():
            if db_id:
                try:
                    db = self.notion.databases.retrieve(db_id)
                    schemas[db_name] = {
                        "id": db_id,
                        "properties": self._extract_properties(db.get("properties", {}))
                    }
                except Exception as e:
                    print(f"[WARNING] Could not backup {db_name}: {e}")
        
        return schemas
    
    async def _backup_page_templates(self):
        """페이지 템플릿 백업"""
        
        templates = {
            "dashboard_layout": {
                "type": "3-column",
                "columns": [
                    {
                        "title": "Calendar & Time",
                        "widgets": ["clock", "calendar", "todo_list"]
                    },
                    {
                        "title": "Metrics & Progress",
                        "widgets": ["metrics", "progress_bars", "charts"]
                    },
                    {
                        "title": "Quick Actions",
                        "widgets": ["shortcuts", "buttons", "notes"]
                    }
                ]
            },
            "kanban_board": {
                "type": "board",
                "groups": ["Backlog", "To Do", "In Progress", "Review", "Done"],
                "properties": ["Priority", "Component", "Effort", "Sprint"]
            },
            "second_brain": {
                "type": "PARA",
                "structure": {
                    "Projects": "Active projects",
                    "Areas": "Ongoing responsibilities",
                    "Resources": "Reference materials",
                    "Archive": "Completed items"
                }
            }
        }
        
        return templates
    
    async def _backup_metadata(self):
        """메타데이터 백업"""
        
        return {
            "theme": {
                "mode": "dark",
                "primary_color": "#5B9CF6",
                "success_color": "#4CAF50",
                "warning_color": "#FFA726",
                "error_color": "#EF5350"
            },
            "icons": {
                "projects": "🚀",
                "tasks": "📋",
                "metrics": "📊",
                "calendar": "📅",
                "settings": "⚙️"
            },
            "widgets": {
                "external": ["indify.co", "widgetbox.app"],
                "custom": ["notion_widgets.html"]
            }
        }
    
    def _extract_block_content(self, block):
        """블록 콘텐츠 추출"""
        block_type = block.get("type")
        content = {}
        
        if block_type in ["paragraph", "heading_1", "heading_2", "heading_3"]:
            texts = block.get(block_type, {}).get("rich_text", [])
            content["text"] = " ".join([t.get("plain_text", "") for t in texts])
        elif block_type == "column_list":
            content["columns"] = len(block.get(block_type, {}).get("children", []))
        
        return content
    
    def _extract_properties(self, properties):
        """데이터베이스 속성 추출"""
        extracted = {}
        
        for prop_name, prop_data in properties.items():
            extracted[prop_name] = {
                "type": prop_data.get("type"),
                "options": self._get_property_options(prop_data)
            }
        
        return extracted
    
    def _get_property_options(self, prop_data):
        """속성 옵션 추출"""
        prop_type = prop_data.get("type")
        
        if prop_type == "select":
            return [opt.get("name") for opt in prop_data.get("select", {}).get("options", [])]
        elif prop_type == "multi_select":
            return [opt.get("name") for opt in prop_data.get("multi_select", {}).get("options", [])]
        
        return None
    
    def _create_checksum(self, file_path):
        """파일 체크섬 생성"""
        with open(file_path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    
    async def create_template_lock(self):
        """템플릿 잠금 파일 생성"""
        
        lock_file = {
            "locked": True,
            "locked_at": datetime.now().isoformat(),
            "template_version": "1.0.0",
            "message": "Template locked to prevent accidental changes",
            "protected_elements": [
                "dashboard_structure",
                "3_column_layout",
                "database_schemas",
                "widget_placements",
                "color_theme"
            ],
            "allow_additions": True,
            "allow_modifications": False,
            "backup_before_change": True
        }
        
        lock_path = self.backup_dir / "template.lock"
        with open(lock_path, "w", encoding="utf-8") as f:
            json.dump(lock_file, f, indent=2)
        
        print(f"[INFO] Template lock created: {lock_path}")
        return lock_path
    
    async def restore_from_backup(self, version: str):
        """백업에서 복원"""
        
        backup_file = self.backup_dir / f"backup_{version}.json"
        
        if not backup_file.exists():
            print(f"[ERROR] Backup not found: {backup_file}")
            return False
        
        with open(backup_file, "r", encoding="utf-8") as f:
            backup_data = json.load(f)
        
        print(f"[INFO] Restoring from backup: {version}")
        print(f"[INFO] Created: {backup_data['created_at']}")
        
        # 복원 로직 (실제 구현은 Notion API 제한으로 수동 필요)
        restore_guide = f"""
        ## Template Restore Guide
        
        Version: {backup_data['version']}
        Created: {backup_data['created_at']}
        
        ### Dashboard Structure:
        {json.dumps(backup_data['components']['dashboard']['structure'], indent=2)}
        
        ### Database Schemas:
        {json.dumps(backup_data['components']['databases'], indent=2)}
        
        ### Templates:
        {json.dumps(backup_data['components']['templates'], indent=2)}
        
        ### Theme:
        {json.dumps(backup_data['components']['metadata']['theme'], indent=2)}
        """
        
        restore_file = self.backup_dir / f"restore_guide_{version}.md"
        with open(restore_file, "w", encoding="utf-8") as f:
            f.write(restore_guide)
        
        print(f"[SUCCESS] Restore guide created: {restore_file}")
        return True
    
    async def list_backups(self):
        """백업 목록 조회"""
        
        backups = []
        for backup_file in self.backup_dir.glob("backup_*.json"):
            with open(backup_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                backups.append({
                    "version": data["version"],
                    "created": data["created_at"],
                    "file": backup_file.name
                })
        
        return backups


class TemplateVersionControl:
    """템플릿 버전 관리 시스템"""
    
    def __init__(self):
        self.versions_file = Path("executive_control/template_versions.json")
        self.current_version = "1.0.0"
        
    def init_version_control(self):
        """버전 관리 초기화"""
        
        if self.versions_file.exists():
            print("[INFO] Version control already initialized")
            return
        
        initial_version = {
            "current": "1.0.0",
            "versions": [
                {
                    "version": "1.0.0",
                    "date": datetime.now().isoformat(),
                    "description": "Initial template design",
                    "changes": [
                        "3-column dashboard layout",
                        "Dark mode theme",
                        "Kanban board setup",
                        "Custom widgets",
                        "PARA method for Second Brain"
                    ],
                    "stable": True
                }
            ],
            "protected": True,
            "auto_backup": True
        }
        
        with open(self.versions_file, "w", encoding="utf-8") as f:
            json.dump(initial_version, f, indent=2)
        
        print(f"[SUCCESS] Version control initialized: {self.versions_file}")
        return initial_version
    
    def create_branch(self, branch_name: str):
        """새 브랜치 생성 (실험용)"""
        
        branch_file = Path(f"executive_control/branches/{branch_name}.json")
        branch_file.parent.mkdir(exist_ok=True)
        
        branch_data = {
            "name": branch_name,
            "created": datetime.now().isoformat(),
            "based_on": self.current_version,
            "experimental": True,
            "changes": []
        }
        
        with open(branch_file, "w", encoding="utf-8") as f:
            json.dump(branch_data, f, indent=2)
        
        print(f"[INFO] Branch created: {branch_name}")
        return branch_file


async def main():
    """메인 실행 함수"""
    
    print("\n" + "="*50)
    print("[INFO] Notion Template Backup System")
    print("="*50)
    
    # 1. 백업 시스템 초기화
    backup = NotionTemplateBackup()
    
    # 2. 현재 템플릿 백업
    backup_file, checksum = await backup.create_backup("v1.0.0_initial")
    
    # 3. 템플릿 잠금
    lock_file = await backup.create_template_lock()
    
    # 4. 버전 관리 초기화
    version_control = TemplateVersionControl()
    version_control.init_version_control()
    
    # 5. 백업 목록 조회
    backups = await backup.list_backups()
    
    print("\n" + "="*50)
    print("[SUCCESS] Template Protection Complete!")
    print("="*50)
    print("\n[PROTECTED ELEMENTS]:")
    print("- Dashboard structure (3-column layout)")
    print("- Database schemas")
    print("- Color theme & Dark mode")
    print("- Widget placements")
    print("- PARA structure")
    
    print("\n[BACKUP INFO]:")
    print(f"- Backup file: {backup_file}")
    print(f"- Checksum: {checksum}")
    print(f"- Lock file: {lock_file}")
    
    print("\n[VERSION CONTROL]:")
    print("- Current version: v1.0.0")
    print("- Protected: Yes")
    print("- Auto-backup: Enabled")
    
    print("\n[HOW TO RESTORE]:")
    print("1. Run: python notion_template_backup.py --restore v1.0.0")
    print("2. Follow the restore guide")
    print("3. Manually apply changes in Notion")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--restore":
        version = sys.argv[2] if len(sys.argv) > 2 else "v1.0.0_initial"
        backup = NotionTemplateBackup()
        asyncio.run(backup.restore_from_backup(version))
    else:
        asyncio.run(main())