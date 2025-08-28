"""
Multi-Project Setup Runner
Simple script to execute the multi-project dashboard setup and initialization
"""

import os
import sys
import asyncio
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add executive_control to path
sys.path.append(str(Path(__file__).parent))

from setup_multi_project_dashboard import MultiProjectDashboardSetup
from notion_project_manager import NotionProjectManager, initialize_project_with_sample_data


async def run_complete_setup():
    """Run complete multi-project setup including sample data"""
    
    print("="*70)
    print("           MULTI-PROJECT CRYPTO TRADING WORKSPACE SETUP")
    print("="*70)
    print()
    
    # Check Notion token
    notion_token = os.getenv("NOTION_TOKEN")
    if not notion_token:
        print("[ERROR] NOTION_TOKEN not found in environment variables")
        print()
        print("To fix this:")
        print("1. Get your Notion API token from https://www.notion.so/my-integrations")
        print("2. Set the environment variable:")
        print("   export NOTION_TOKEN=your_token_here")
        print("3. Or add it to your .env file")
        return False
    
    print("[INFO] Notion API token found")
    print()
    
    # Get parent page
    print("STEP 1: Specify Parent Page")
    print("-" * 30)
    print("Enter the Notion page where you want to create the workspace.")
    print("This can be either:")
    print("  - A page URL (e.g., https://notion.so/My-Page-abc123...)")
    print("  - A page ID (e.g., abc12345-1234-1234-1234-123456789abc)")
    print()
    
    parent_page = input("Parent page URL or ID: ").strip()
    
    if not parent_page:
        print("[ERROR] Parent page is required")
        return False
    
    # Extract ID from URL if needed
    if "notion.so" in parent_page:
        parent_page = parent_page.split("/")[-1].split("-")[-1]
    
    # Format ID with hyphens if needed
    if len(parent_page) == 32 and "-" not in parent_page:
        parent_page = f"{parent_page[:8]}-{parent_page[8:12]}-{parent_page[12:16]}-{parent_page[16:20]}-{parent_page[20:]}"
    
    print(f"[OK] Using parent page ID: {parent_page}")
    print()
    
    # Setup options
    print("STEP 2: Setup Options")
    print("-" * 20)
    print("What would you like to create?")
    print("  1. Workspace structure only")
    print("  2. Workspace + sample tasks and documentation")
    print("  3. Complete setup + cross-project dependencies")
    print()
    
    choice = input("Choose option (1-3): ").strip()
    
    include_sample_data = choice in ["2", "3"]
    include_dependencies = choice == "3"
    
    print()
    print("STEP 3: Creating Workspace")
    print("-" * 25)
    
    # Create workspace structure
    try:
        setup = MultiProjectDashboardSetup(notion_token)
        resources = await setup.setup_complete_workspace(parent_page)
        
        print("\n[SUCCESS] Workspace structure created!")
        
        # Add sample data if requested
        if include_sample_data:
            print("\nSTEP 4: Adding Sample Data")
            print("-" * 23)
            
            manager = NotionProjectManager(notion_token, "multi_project_config.json")
            
            # Initialize each project
            projects = ['trading_core', 'ml_engine', 'dashboard', 'risk_management']
            for project_key in projects:
                print(f"[INFO] Initializing {project_key}...")
                await initialize_project_with_sample_data(manager, project_key)
            
            print("\n[SUCCESS] Sample data added to all projects!")
            
            # Add cross-project dependencies if requested
            if include_dependencies:
                print("\nSTEP 5: Creating Cross-Project Dependencies")
                print("-" * 40)
                
                # Sample dependencies
                dependencies = [
                    {
                        "from": "Dashboard",
                        "to": "Trading Core", 
                        "type": "API",
                        "desc": "Real-time trading data and order status"
                    },
                    {
                        "from": "Trading Core",
                        "to": "ML Engine",
                        "type": "API", 
                        "desc": "ML signals for trading decisions"
                    },
                    {
                        "from": "Trading Core",
                        "to": "Risk Management",
                        "type": "Service",
                        "desc": "Position validation and risk checks"
                    },
                    {
                        "from": "Dashboard",
                        "to": "Risk Management", 
                        "type": "API",
                        "desc": "Risk metrics and portfolio analysis"
                    }
                ]
                
                for dep in dependencies:
                    await manager.add_cross_project_dependency(
                        from_project=dep["from"],
                        to_project=dep["to"],
                        dependency_type=dep["type"],
                        description=dep["desc"]
                    )
                
                # Sample architecture decisions
                await manager.create_architecture_decision(
                    title="Use WebSocket for real-time data streaming",
                    decision_id="ADR-001",
                    affects_projects=["Trading Core", "Dashboard"],
                    context="Need sub-millisecond latency for trading data",
                    decision="Implement WebSocket connections with connection pooling",
                    consequences="Lower latency but increased complexity in connection management"
                )
                
                await manager.create_architecture_decision(
                    title="Use Redis for shared state management",
                    decision_id="ADR-002", 
                    affects_projects=["Trading Core", "ML Engine", "Risk Management"],
                    context="Multiple services need access to current positions and market state",
                    decision="Centralized Redis instance for shared state with pub/sub for updates",
                    consequences="Single point of failure but simplified state consistency"
                )
                
                print("\n[SUCCESS] Cross-project dependencies and ADRs created!")
        
        print("\n" + "="*70)
        print("                        SETUP COMPLETE!")
        print("="*70)
        
        # Display URLs
        master_dashboard_id = resources.get('master_dashboard', '').replace('-', '')
        print(f"\n🎯 Master Dashboard: https://notion.so/{master_dashboard_id}")
        print("\n📋 Project Workspaces:")
        
        for project_key, project_resources in resources['projects'].items():
            project_name = project_key.replace('_', ' ').title()
            page_id = project_resources.get('page', '').replace('-', '')
            print(f"   {project_name}: https://notion.so/{page_id}")
        
        print("\n📚 Shared Databases:")
        for db_name, db_id in resources['shared_databases'].items():
            db_display_name = db_name.replace('_', ' ').title()
            print(f"   {db_display_name}: https://notion.so/{db_id.replace('-', '')}")
        
        print("\n📁 Files Created:")
        print("   - .env.multi_project (environment variables)")
        print("   - multi_project_config.json (workspace configuration)")
        print("   - MULTI_PROJECT_SETUP_COMPLETE.md (setup guide)")
        
        print("\n💡 Next Steps:")
        print("   1. Review the setup guide in MULTI_PROJECT_SETUP_COMPLETE.md")
        print("   2. Customize the workspace to your needs")
        print("   3. Start adding your actual project tasks")
        print("   4. Set up integrations with your development tools")
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Setup failed: {e}")
        print("\nPlease check:")
        print("  - Your Notion token is valid")
        print("  - The parent page exists and you have edit permissions")
        print("  - Your internet connection is stable")
        return False


async def run_quick_demo():
    """Run a quick demo showing project statistics"""
    
    notion_token = os.getenv("NOTION_TOKEN")
    if not notion_token:
        print("[ERROR] NOTION_TOKEN not found")
        return
    
    try:
        manager = NotionProjectManager(notion_token, "multi_project_config.json")
        
        print("="*50)
        print("        PROJECT STATISTICS DEMO")
        print("="*50)
        print()
        
        projects = ['trading_core', 'ml_engine', 'dashboard', 'risk_management']
        
        for project_key in projects:
            try:
                stats = await manager.get_project_stats(project_key)
                project_name = project_key.replace('_', ' ').title()
                
                print(f"📊 {project_name}")
                print(f"   Tasks: {stats['total_tasks']} total, {stats['completed_tasks']} done")
                print(f"   Progress: {stats['completion_rate']:.1%}")
                print(f"   Effort: {stats['completed_effort_days']:.1f}/{stats['total_effort_days']:.1f} days")
                print()
                
            except Exception as e:
                print(f"⚠️  {project_key}: Could not load stats ({e})")
                print()
        
        print("[SUCCESS] Statistics demo completed!")
        
    except Exception as e:
        print(f"[ERROR] Demo failed: {e}")


def main():
    """Main entry point"""
    
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        asyncio.run(run_quick_demo())
    else:
        success = asyncio.run(run_complete_setup())
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()