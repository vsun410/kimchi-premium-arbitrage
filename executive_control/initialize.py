"""
Executive Control System Initialization
시스템 초기화 및 상태 확인
"""

import os
import sys
import asyncio
from pathlib import Path
from typing import Optional
import json
from dotenv import load_dotenv

from vision_guardian import VisionGuardian
from task_orchestrator import TaskOrchestrator
from notion_governance_integration import NotionGovernanceIntegration
from claude_code_interceptor import ClaudeCodeInterceptor


class ExecutiveControlInitializer:
    """Executive Control System 초기화 관리자"""
    
    def __init__(self):
        self.config_path = Path("executive_control/config.json")
        self.env_path = Path(".env.executive")
        self.is_initialized = False
        
    async def initialize_system(self):
        """전체 시스템 초기화"""
        
        print("\n" + "="*60)
        print("[INFO] Executive Control System Initialization")
        print("="*60)
        
        # 1. 환경 변수 로드
        if not await self._load_environment():
            return False
        
        # 2. Notion 연결 테스트
        if not await self._test_notion_connection():
            return False
        
        # 3. Vision Guardian 초기화
        if not await self._initialize_vision_guardian():
            return False
        
        # 4. Git hooks 설치
        if not await self._setup_git_hooks():
            return False
        
        # 5. VS Code 설정
        if not await self._setup_vscode():
            return False
        
        # 6. 검증 스크립트 생성
        if not await self._create_validation_scripts():
            return False
        
        # 7. 설정 저장
        await self._save_configuration()
        
        print("\n" + "="*60)
        print("[SUCCESS] Executive Control System Initialized Successfully!")
        print("="*60)
        
        await self._show_usage_guide()
        
        return True
    
    async def _load_environment(self) -> bool:
        """환경 변수 로드"""
        print("\n[INFO] Loading environment variables...")
        
        # .env.executive 파일 확인
        if self.env_path.exists():
            load_dotenv(self.env_path)
            print("  [OK] Loaded .env.executive")
        
        # 기본 .env 파일도 로드
        if Path(".env").exists():
            load_dotenv()
            print("  [OK] Loaded .env")
        
        # 필수 환경 변수 확인
        required_vars = [
            "NOTION_TOKEN",
            "NOTION_VISION_DB",
            "NOTION_TASKS_DB",
            "NOTION_VALIDATION_DB",
            "NOTION_DASHBOARD_PAGE"
        ]
        
        missing_vars = []
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            print(f"\n[ERROR] Missing required environment variables:")
            for var in missing_vars:
                print(f"  - {var}")
            
            print("\n[TIP] Please run the dashboard setup first:")
            print("  python -m executive_control.setup_notion_executive_dashboard")
            return False
        
        print("  [OK] All required environment variables found")
        return True
    
    async def _test_notion_connection(self) -> bool:
        """Notion 연결 테스트"""
        print("\n[INFO] Testing Notion connection...")
        
        try:
            config = {
                'vision_db': os.getenv('NOTION_VISION_DB'),
                'tasks_db': os.getenv('NOTION_TASKS_DB'),
                'validation_db': os.getenv('NOTION_VALIDATION_DB'),
                'dashboard_page': os.getenv('NOTION_DASHBOARD_PAGE')
            }
            
            governance = NotionGovernanceIntegration(
                os.getenv('NOTION_TOKEN'),
                config
            )
            
            # 간단한 조회 테스트
            governance.notion.users.me()
            print("  [OK] Notion connection successful")
            return True
            
        except Exception as e:
            print(f"  [ERROR] Notion connection failed: {e}")
            return False
    
    async def _initialize_vision_guardian(self) -> bool:
        """Vision Guardian 초기화"""
        print("\n🛡️ Initializing Vision Guardian...")
        
        try:
            # PRD와 아키텍처 파일 확인/생성
            prd_path = Path("executive_control/prd.md")
            arch_path = Path("executive_control/architecture.json")
            
            if not prd_path.exists():
                print("  [INFO] Creating default PRD...")
                await self._create_default_prd()
            
            if not arch_path.exists():
                print("  🏗️ Creating default architecture...")
                await self._create_default_architecture()
            
            # Vision Guardian 초기화
            guardian = VisionGuardian(
                str(prd_path),
                str(arch_path)
            )
            
            print("  [OK] Vision Guardian initialized")
            return True
            
        except Exception as e:
            print(f"  [ERROR] Vision Guardian initialization failed: {e}")
            return False
    
    async def _create_default_prd(self):
        """기본 PRD 생성"""
        prd_content = """# Kimchi Premium Trading System PRD

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

## Success Metrics
- Sharpe Ratio > 1.5
- Max Drawdown < 15%
- Win Rate > 60%
- System Uptime > 99%
"""
        
        prd_path = Path("executive_control/prd.md")
        prd_path.parent.mkdir(exist_ok=True)
        prd_path.write_text(prd_content, encoding='utf-8')
    
    async def _create_default_architecture(self):
        """기본 아키텍처 명세 생성"""
        architecture = {
            "components": {
                "trading_core": {
                    "interfaces": ["WebSocket", "REST API"],
                    "dependencies": ["market_data", "risk_manager"],
                    "constraints": ["latency < 100ms", "stateless"]
                },
                "ml_engine": {
                    "interfaces": ["gRPC", "Message Queue"],
                    "dependencies": ["historical_data", "feature_store"],
                    "constraints": ["prediction_time < 500ms"]
                },
                "dashboard": {
                    "interfaces": ["REST API", "WebSocket"],
                    "dependencies": ["trading_core", "ml_engine"],
                    "constraints": ["real-time updates", "responsive UI"]
                },
                "risk_manager": {
                    "interfaces": ["Internal API"],
                    "dependencies": ["trading_core"],
                    "constraints": ["position_limit < 10%", "max_leverage < 3x"]
                },
                "data_pipeline": {
                    "interfaces": ["WebSocket", "REST API"],
                    "dependencies": [],
                    "constraints": ["data_lag < 1s", "99.9% uptime"]
                }
            },
            "data_flow": {
                "market_data": "trading_core",
                "trading_core": "risk_manager",
                "ml_engine": "trading_core",
                "all": "dashboard"
            }
        }
        
        arch_path = Path("executive_control/architecture.json")
        arch_path.parent.mkdir(exist_ok=True)
        arch_path.write_text(json.dumps(architecture, indent=2), encoding='utf-8')
    
    async def _setup_git_hooks(self) -> bool:
        """Git hooks 설치"""
        print("\n[INFO] Setting up Git hooks...")
        
        try:
            config = {
                'vision_db': os.getenv('NOTION_VISION_DB'),
                'tasks_db': os.getenv('NOTION_TASKS_DB'),
                'validation_db': os.getenv('NOTION_VALIDATION_DB'),
                'dashboard_page': os.getenv('NOTION_DASHBOARD_PAGE')
            }
            
            governance = NotionGovernanceIntegration(
                os.getenv('NOTION_TOKEN'),
                config
            )
            
            interceptor = ClaudeCodeInterceptor(governance)
            interceptor.install_git_hooks()
            
            return True
            
        except Exception as e:
            print(f"  ⚠️ Git hooks setup failed: {e}")
            print("  (Non-critical, continuing...)")
            return True  # Non-critical error
    
    async def _setup_vscode(self) -> bool:
        """VS Code 설정"""
        print("\n[INFO] Setting up VS Code integration...")
        
        try:
            config = {
                'vision_db': os.getenv('NOTION_VISION_DB'),
                'tasks_db': os.getenv('NOTION_TASKS_DB'),
                'validation_db': os.getenv('NOTION_VALIDATION_DB'),
                'dashboard_page': os.getenv('NOTION_DASHBOARD_PAGE')
            }
            
            governance = NotionGovernanceIntegration(
                os.getenv('NOTION_TOKEN'),
                config
            )
            
            interceptor = ClaudeCodeInterceptor(governance)
            interceptor.create_vscode_extension()
            
            return True
            
        except Exception as e:
            print(f"  ⚠️ VS Code setup failed: {e}")
            print("  (Non-critical, continuing...)")
            return True  # Non-critical error
    
    async def _create_validation_scripts(self) -> bool:
        """검증 스크립트 생성"""
        print("\n[INFO] Creating validation scripts...")
        
        try:
            config = {
                'vision_db': os.getenv('NOTION_VISION_DB'),
                'tasks_db': os.getenv('NOTION_TASKS_DB'),
                'validation_db': os.getenv('NOTION_VALIDATION_DB'),
                'dashboard_page': os.getenv('NOTION_DASHBOARD_PAGE')
            }
            
            governance = NotionGovernanceIntegration(
                os.getenv('NOTION_TOKEN'),
                config
            )
            
            interceptor = ClaudeCodeInterceptor(governance)
            interceptor.create_validation_script()
            
            return True
            
        except Exception as e:
            print(f"  ⚠️ Validation script creation failed: {e}")
            print("  (Non-critical, continuing...)")
            return True  # Non-critical error
    
    async def _save_configuration(self):
        """설정 저장"""
        print("\n[INFO] Saving configuration...")
        
        config = {
            "version": "1.0.0",
            "initialized": True,
            "components": {
                "vision_guardian": True,
                "task_orchestrator": True,
                "notion_integration": True,
                "claude_interceptor": True
            },
            "databases": {
                "vision_db": os.getenv('NOTION_VISION_DB'),
                "tasks_db": os.getenv('NOTION_TASKS_DB'),
                "validation_db": os.getenv('NOTION_VALIDATION_DB'),
                "dashboard_page": os.getenv('NOTION_DASHBOARD_PAGE')
            },
            "settings": {
                "validation_threshold_business": 0.8,
                "validation_threshold_architecture": 0.8,
                "validation_threshold_drift": 0.2,
                "auto_validation": True,
                "git_hooks_enabled": True
            }
        }
        
        self.config_path.parent.mkdir(exist_ok=True)
        self.config_path.write_text(json.dumps(config, indent=2), encoding='utf-8')
        print("  [OK] Configuration saved")
    
    async def _show_usage_guide(self):
        """사용 가이드 표시"""
        
        dashboard_url = f"https://notion.so/{os.getenv('NOTION_DASHBOARD_PAGE', '').replace('-', '')}"
        
        guide = f"""
╔════════════════════════════════════════════════════════════╗
║            Executive Control System Usage Guide           ║
╚════════════════════════════════════════════════════════════╝

[DASHBOARD URL]:
   {dashboard_url}

[QUICK COMMANDS]:

1. Validate a single file:
   python validate.py <file_path>

2. Validate all files:
   python validate.py --all

3. Start file monitoring:
   python -m executive_control.monitor

4. Submit new requirement:
   python -m executive_control.submit "Your requirement here"

5. Check system status:
   python -m executive_control.status

[VS CODE]:
   Press Ctrl+Shift+V to validate current file

[GIT]:
   All commits are automatically validated

[DOCS]:
   See executive_control/README.md for full documentation

[TIPS]:
   - Keep your Notion dashboard open for real-time updates
   - Review validation failures in the dashboard
   - Use code blocks for replaceable components
   - Monitor drift score to prevent vision deviation
"""
        
        print(guide)
    
    async def check_status(self):
        """시스템 상태 확인"""
        
        print("\n" + "="*60)
        print("[STATUS] Executive Control System Status")
        print("="*60)
        
        # 설정 파일 확인
        if self.config_path.exists():
            config = json.loads(self.config_path.read_text())
            print(f"\n[OK] System initialized: v{config['version']}")
            
            print("\n📦 Components:")
            for component, status in config['components'].items():
                status_icon = "[OK]" if status else "[ERROR]"
                print(f"  {status_icon} {component}")
            
            print("\n⚙️ Settings:")
            for key, value in config['settings'].items():
                print(f"  - {key}: {value}")
        else:
            print("\n[ERROR] System not initialized")
            print("Run: python -m executive_control.initialize")
        
        # Notion 연결 상태
        try:
            from notion_client import Client
            notion = Client(auth=os.getenv('NOTION_TOKEN'))
            notion.users.me()
            print("\n[OK] Notion connection: Active")
        except:
            print("\n[ERROR] Notion connection: Failed")
        
        # Git hooks 상태
        git_hook_path = Path(".git/hooks/pre-commit")
        if git_hook_path.exists():
            print("[OK] Git hooks: Installed")
        else:
            print("[ERROR] Git hooks: Not installed")
        
        # VS Code 설정 상태
        vscode_settings = Path(".vscode/settings.json")
        if vscode_settings.exists():
            print("[OK] VS Code integration: Configured")
        else:
            print("[ERROR] VS Code integration: Not configured")
        
        print("\n" + "="*60)


async def main():
    """메인 함수"""
    
    initializer = ExecutiveControlInitializer()
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "status":
            await initializer.check_status()
        elif command == "init":
            await initializer.initialize_system()
        else:
            print(f"Unknown command: {command}")
            print("Usage: python -m executive_control.initialize [init|status]")
    else:
        # 기본: 초기화 실행
        await initializer.initialize_system()


if __name__ == "__main__":
    asyncio.run(main())