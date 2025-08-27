"""
Test Multi-Project Setup
Validation script to test the multi-project workspace setup
"""

import os
import sys
import asyncio
import json
from pathlib import Path

# Add executive_control to path
sys.path.append(str(Path(__file__).parent))

from notion_project_manager import NotionProjectManager, Priority, TaskStatus, DocType


async def test_configuration():
    """Test that configuration files exist and are valid"""
    
    print("[INFO] Testing configuration files...")
    
    # Check environment file
    env_file = ".env.multi_project"
    if not os.path.exists(env_file):
        print(f"[ERROR] {env_file} not found. Run setup first.")
        return False
    
    # Check configuration file
    config_file = "multi_project_config.json"
    if not os.path.exists(config_file):
        print(f"[ERROR] {config_file} not found. Run setup first.")
        return False
    
    # Validate configuration
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        required_keys = ['workspace_name', 'projects', 'notion_resources']
        for key in required_keys:
            if key not in config:
                print(f"[ERROR] Missing key '{key}' in configuration")
                return False
        
        print("[OK] Configuration files valid")
        return True
        
    except json.JSONDecodeError as e:
        print(f"[ERROR] Invalid JSON in configuration: {e}")
        return False


async def test_notion_connection():
    """Test connection to Notion API"""
    
    print("[INFO] Testing Notion API connection...")
    
    notion_token = os.getenv("NOTION_TOKEN")
    if not notion_token:
        print("[ERROR] NOTION_TOKEN not found in environment")
        return False
    
    try:
        manager = NotionProjectManager(notion_token, "multi_project_config.json")
        
        # Simple API test - just initialize the manager
        if not manager.config:
            print("[ERROR] Failed to load configuration")
            return False
        
        print("[OK] Notion API connection successful")
        return True
        
    except Exception as e:
        print(f"[ERROR] Notion API connection failed: {e}")
        return False


async def test_database_access():
    """Test access to created databases"""
    
    print("[INFO] Testing database access...")
    
    try:
        manager = NotionProjectManager(
            os.getenv("NOTION_TOKEN"), 
            "multi_project_config.json"
        )
        
        # Test master dashboard exists
        master_dashboard = manager.resources.get('master_dashboard')
        if not master_dashboard:
            print("[ERROR] Master dashboard not found in configuration")
            return False
        
        # Test shared databases exist
        shared_dbs = manager.resources.get('shared_databases', {})
        required_shared_dbs = ['adr', 'dependencies', 'research']
        
        for db_name in required_shared_dbs:
            if db_name not in shared_dbs:
                print(f"[ERROR] Shared database '{db_name}' not found")
                return False
        
        # Test project databases exist
        projects = manager.resources.get('projects', {})
        expected_projects = ['trading_core', 'ml_engine', 'dashboard', 'risk_management']
        
        for project_key in expected_projects:
            if project_key not in projects:
                print(f"[ERROR] Project '{project_key}' not found")
                return False
                
            project_resources = projects[project_key]
            required_resources = ['page', 'tasks_db', 'docs_db']
            
            for resource in required_resources:
                if resource not in project_resources:
                    print(f"[ERROR] Resource '{resource}' not found for project '{project_key}'")
                    return False
        
        print("[OK] All databases accessible")
        return True
        
    except Exception as e:
        print(f"[ERROR] Database access test failed: {e}")
        return False


async def test_basic_operations():
    """Test basic create/read operations"""
    
    print("[INFO] Testing basic operations...")
    
    try:
        manager = NotionProjectManager(
            os.getenv("NOTION_TOKEN"), 
            "multi_project_config.json"
        )
        
        # Test creating a task
        test_task_id = await manager.create_task(
            project_key="trading_core",
            title="[TEST] Connection test task",
            component="Connection Pool",
            priority=Priority.LOW,
            description="This is a test task created by the validation script",
            tags=["test", "validation"]
        )
        
        print(f"[OK] Created test task: {test_task_id}")
        
        # Test updating task progress
        success = await manager.update_task_progress(
            project_key="trading_core",
            task_id=test_task_id,
            progress=0.5,
            status=TaskStatus.IN_PROGRESS
        )
        
        if success:
            print("[OK] Updated task progress")
        else:
            print("[WARNING] Task progress update failed")
        
        # Test getting project stats
        stats = await manager.get_project_stats("trading_core")
        print(f"[OK] Retrieved stats: {stats['total_tasks']} total tasks")
        
        # Test creating documentation
        doc_id = await manager.create_documentation(
            project_key="trading_core",
            title="[TEST] API Documentation",
            doc_type=DocType.API_DOCS,
            component="Connection Pool",
            content="This is test documentation created by the validation script",
            tags=["test", "api"]
        )
        
        print(f"[OK] Created test documentation: {doc_id}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Basic operations test failed: {e}")
        return False


async def test_cross_project_features():
    """Test cross-project features"""
    
    print("[INFO] Testing cross-project features...")
    
    try:
        manager = NotionProjectManager(
            os.getenv("NOTION_TOKEN"), 
            "multi_project_config.json"
        )
        
        # Test creating architecture decision
        adr_id = await manager.create_architecture_decision(
            title="[TEST] Test Architecture Decision",
            decision_id="TEST-001",
            affects_projects=["Trading Core", "Dashboard"],
            context="This is a test ADR created by the validation script",
            decision="Use test framework for validation",
            consequences="Better test coverage and validation"
        )
        
        print(f"[OK] Created test ADR: {adr_id}")
        
        # Test creating cross-project dependency
        dep_id = await manager.add_cross_project_dependency(
            from_project="Dashboard",
            to_project="Trading Core",
            dependency_type="API",
            description="[TEST] Test dependency created by validation script"
        )
        
        print(f"[OK] Created test dependency: {dep_id}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Cross-project features test failed: {e}")
        return False


async def cleanup_test_data():
    """Clean up test data (optional)"""
    
    print("[INFO] Test data cleanup...")
    print("[NOTE] Test entries created with [TEST] prefix for easy identification")
    print("[NOTE] You can manually delete them from Notion if desired")


async def run_full_test():
    """Run complete test suite"""
    
    print("="*60)
    print("           MULTI-PROJECT WORKSPACE VALIDATION")
    print("="*60)
    print()
    
    tests = [
        ("Configuration Files", test_configuration),
        ("Notion API Connection", test_notion_connection),
        ("Database Access", test_database_access),
        ("Basic Operations", test_basic_operations),
        ("Cross-Project Features", test_cross_project_features),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"Running: {test_name}")
        print("-" * len(f"Running: {test_name}"))
        
        try:
            success = await test_func()
            if success:
                passed += 1
                print(f"[PASS] {test_name}")
            else:
                failed += 1
                print(f"[FAIL] {test_name}")
        except Exception as e:
            failed += 1
            print(f"[FAIL] {test_name} - Exception: {e}")
        
        print()
    
    # Summary
    print("="*60)
    print("                    TEST SUMMARY")
    print("="*60)
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total:  {passed + failed}")
    print()
    
    if failed == 0:
        print("🎉 All tests passed! Your multi-project workspace is ready to use.")
        print()
        print("Next steps:")
        print("1. Open your Notion workspace")
        print("2. Start adding real project tasks")
        print("3. Customize the workspace to your needs")
        await cleanup_test_data()
    else:
        print(f"⚠️  {failed} test(s) failed. Please check the setup:")
        print("1. Ensure setup_multi_project_dashboard.py ran successfully")
        print("2. Check that your Notion token is valid")
        print("3. Verify database permissions in Notion")
        print("4. Re-run setup if necessary")
    
    return failed == 0


async def run_quick_check():
    """Run quick connectivity check"""
    
    print("Quick Connectivity Check")
    print("-" * 25)
    
    # Check files exist
    config_exists = os.path.exists("multi_project_config.json")
    env_exists = os.path.exists(".env.multi_project")
    
    print(f"Config file:  {'✓' if config_exists else '✗'}")
    print(f"Env file:     {'✓' if env_exists else '✗'}")
    
    # Check token
    token_exists = bool(os.getenv("NOTION_TOKEN"))
    print(f"Notion token: {'✓' if token_exists else '✗'}")
    
    if not all([config_exists, env_exists, token_exists]):
        print("\n❌ Setup incomplete. Run setup_multi_project_dashboard.py first.")
        return False
    
    # Quick API test
    try:
        manager = NotionProjectManager(
            os.getenv("NOTION_TOKEN"), 
            "multi_project_config.json"
        )
        
        # Test one project stats
        stats = await manager.get_project_stats("trading_core")
        print(f"API access:   ✓ ({stats['total_tasks']} tasks found)")
        
        print("\n✅ Quick check passed! Workspace is operational.")
        return True
        
    except Exception as e:
        print(f"API access:   ✗ ({e})")
        print("\n❌ Quick check failed. Check your setup and permissions.")
        return False


def main():
    """Main entry point"""
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "quick":
            success = asyncio.run(run_quick_check())
        elif sys.argv[1] == "full":
            success = asyncio.run(run_full_test())
        else:
            print("Usage:")
            print("  python test_multi_project_setup.py quick  # Quick connectivity check")
            print("  python test_multi_project_setup.py full   # Full validation suite")
            success = False
    else:
        # Default to full test
        success = asyncio.run(run_full_test())
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()