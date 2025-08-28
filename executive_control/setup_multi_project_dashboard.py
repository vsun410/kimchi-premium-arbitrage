"""
Multi-Project Notion Dashboard Setup
Create comprehensive project management system for 4 independent crypto trading projects
"""

import os
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
from notion_client import Client
import json


class MultiProjectDashboardSetup:
    """
    Setup comprehensive Notion workspace for 4 independent crypto trading projects:
    1. Trading Core - Ultra-Low Latency Execution Engine
    2. ML Engine - Quantitative Research & Execution Platform  
    3. Dashboard - Professional Trading Terminal
    4. Risk Management - Automated Risk Controls
    """
    
    def __init__(self, notion_token: str):
        self.notion = Client(auth=notion_token)
        self.created_resources = {
            'projects': {},
            'master_dashboard': None,
            'shared_databases': {}
        }
        
        # Project configurations
        self.projects = {
            'trading_core': {
                'name': 'Trading Core - Ultra-Low Latency Execution Engine',
                'emoji': '⚡',
                'description': 'High-frequency trading engine with sub-millisecond latency',
                'components': ['Order Execution', 'Market Data Handler', 'Latency Monitor', 'Connection Pool'],
                'tech_stack': ['C++', 'WebSocket', 'FIX Protocol', 'Redis'],
                'priorities': ['Latency', 'Reliability', 'Throughput']
            },
            'ml_engine': {
                'name': 'ML Engine - Quantitative Research & Execution Platform',
                'emoji': '🤖',
                'description': 'Advanced ML models for market prediction and signal generation',
                'components': ['Feature Engineering', 'Model Training', 'Inference Engine', 'Backtesting'],
                'tech_stack': ['Python', 'PyTorch', 'Pandas', 'NumPy', 'Ray'],
                'priorities': ['Accuracy', 'Speed', 'Scalability']
            },
            'dashboard': {
                'name': 'Dashboard - Professional Trading Terminal',
                'emoji': '📊',
                'description': 'Real-time trading dashboard with advanced analytics',
                'components': ['Real-time Charts', 'Portfolio View', 'Risk Metrics', 'Alert System'],
                'tech_stack': ['React', 'TypeScript', 'WebSocket', 'D3.js'],
                'priorities': ['User Experience', 'Real-time Updates', 'Performance']
            },
            'risk_management': {
                'name': 'Risk Management - Automated Risk Controls',
                'emoji': '🛡️',
                'description': 'Comprehensive risk management and position sizing system',
                'components': ['Position Sizing', 'Risk Limits', 'Drawdown Control', 'Emergency Stop'],
                'tech_stack': ['Python', 'PostgreSQL', 'Redis', 'Kafka'],
                'priorities': ['Safety', 'Compliance', 'Automation']
            }
        }
    
    async def setup_complete_workspace(self, parent_page_id: str) -> Dict:
        """
        Setup complete multi-project workspace
        
        Args:
            parent_page_id: Parent page where workspace will be created
            
        Returns:
            Dictionary with all created resource IDs
        """
        
        print("[INFO] Setting up Multi-Project Crypto Trading Workspace...")
        
        # 1. Create master dashboard page
        master_dashboard = await self._create_master_dashboard(parent_page_id)
        self.created_resources['master_dashboard'] = master_dashboard['id']
        
        # 2. Create shared databases for cross-project tracking
        shared_dbs = await self._create_shared_databases(master_dashboard['id'])
        self.created_resources['shared_databases'] = shared_dbs
        
        # 3. Create individual project workspaces
        for project_key, project_config in self.projects.items():
            print(f"[INFO] Creating workspace for {project_config['name']}...")
            
            project_workspace = await self._create_project_workspace(
                master_dashboard['id'], 
                project_key, 
                project_config
            )
            
            self.created_resources['projects'][project_key] = project_workspace
        
        # 4. Add cross-project widgets to master dashboard
        await self._add_master_dashboard_widgets(master_dashboard['id'])
        
        # 5. Create configuration files
        await self._create_configuration_files()
        
        print("\n[SUCCESS] Multi-Project Dashboard Setup Complete!")
        print(f"\n[MASTER DASHBOARD]: https://notion.so/{master_dashboard['id'].replace('-', '')}")
        
        return self.created_resources
    
    async def _create_master_dashboard(self, parent_id: str) -> Dict:
        """Create master dashboard page that links all projects"""
        
        page = self.notion.pages.create(
            parent={"page_id": parent_id},
            properties={
                "title": [{"text": {"content": "🎯 Multi-Project Crypto Trading Workspace"}}]
            },
            children=[
                {
                    "object": "block",
                    "type": "heading_1",
                    "heading_1": {
                        "rich_text": [{"text": {"content": "Multi-Project Crypto Trading Workspace"}}]
                    }
                },
                {
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [{
                            "text": {
                                "content": "Comprehensive project management system for 4 independent crypto trading system projects"
                            }
                        }]
                    }
                },
                {
                    "object": "block",
                    "type": "divider",
                    "divider": {}
                },
                {
                    "object": "block",
                    "type": "callout",
                    "callout": {
                        "rich_text": [{
                            "text": {
                                "content": "This workspace manages 4 independent projects: Trading Core, ML Engine, Dashboard, and Risk Management. Each project has its own dedicated workspace with task tracking, documentation, and progress monitoring."
                            }
                        }],
                        "icon": {"emoji": "🎯"},
                        "color": "blue_background"
                    }
                }
            ]
        )
        
        print("[OK] Created master dashboard page")
        return page
    
    async def _create_shared_databases(self, dashboard_id: str) -> Dict:
        """Create shared databases for cross-project tracking"""
        
        shared_dbs = {}
        
        # 1. Architecture Decision Records (ADR) Database
        adr_db = self.notion.databases.create(
            parent={"page_id": dashboard_id},
            title=[{"text": {"content": "🏛️ Architecture Decision Records"}}],
            properties={
                "Title": {"title": {}},
                "Decision ID": {"rich_text": {}},
                "Status": {
                    "select": {
                        "options": [
                            {"name": "Proposed", "color": "yellow"},
                            {"name": "Accepted", "color": "green"},
                            {"name": "Deprecated", "color": "red"},
                            {"name": "Superseded", "color": "gray"}
                        ]
                    }
                },
                "Affects Projects": {
                    "multi_select": {
                        "options": [
                            {"name": "Trading Core", "color": "purple"},
                            {"name": "ML Engine", "color": "pink"},
                            {"name": "Dashboard", "color": "blue"},
                            {"name": "Risk Management", "color": "red"}
                        ]
                    }
                },
                "Decision Date": {"date": {}},
                "Context": {"rich_text": {}},
                "Consequences": {"rich_text": {}},
                "Author": {"people": {}}
            }
        )
        shared_dbs['adr'] = adr_db['id']
        
        # 2. Cross-Project Dependencies Database
        deps_db = self.notion.databases.create(
            parent={"page_id": dashboard_id},
            title=[{"text": {"content": "🔗 Cross-Project Dependencies"}}],
            properties={
                "Dependency": {"title": {}},
                "From Project": {
                    "select": {
                        "options": [
                            {"name": "Trading Core", "color": "purple"},
                            {"name": "ML Engine", "color": "pink"},
                            {"name": "Dashboard", "color": "blue"},
                            {"name": "Risk Management", "color": "red"}
                        ]
                    }
                },
                "To Project": {
                    "select": {
                        "options": [
                            {"name": "Trading Core", "color": "purple"},
                            {"name": "ML Engine", "color": "pink"},
                            {"name": "Dashboard", "color": "blue"},
                            {"name": "Risk Management", "color": "red"}
                        ]
                    }
                },
                "Type": {
                    "select": {
                        "options": [
                            {"name": "API", "color": "blue"},
                            {"name": "Data", "color": "green"},
                            {"name": "Service", "color": "yellow"},
                            {"name": "Library", "color": "gray"}
                        ]
                    }
                },
                "Status": {
                    "select": {
                        "options": [
                            {"name": "Active", "color": "green"},
                            {"name": "Planned", "color": "yellow"},
                            {"name": "Deprecated", "color": "red"}
                        ]
                    }
                },
                "Description": {"rich_text": {}},
                "Owner": {"people": {}}
            }
        )
        shared_dbs['dependencies'] = deps_db['id']
        
        # 3. Shared Research Database
        research_db = self.notion.databases.create(
            parent={"page_id": dashboard_id},
            title=[{"text": {"content": "📚 Shared Research & Documentation"}}],
            properties={
                "Title": {"title": {}},
                "Type": {
                    "select": {
                        "options": [
                            {"name": "Market Research", "color": "blue"},
                            {"name": "Technical Analysis", "color": "purple"},
                            {"name": "Algorithm Study", "color": "pink"},
                            {"name": "Best Practices", "color": "green"},
                            {"name": "Tool Evaluation", "color": "yellow"}
                        ]
                    }
                },
                "Relevant Projects": {
                    "multi_select": {
                        "options": [
                            {"name": "Trading Core", "color": "purple"},
                            {"name": "ML Engine", "color": "pink"},
                            {"name": "Dashboard", "color": "blue"},
                            {"name": "Risk Management", "color": "red"}
                        ]
                    }
                },
                "Status": {
                    "select": {
                        "options": [
                            {"name": "Draft", "color": "gray"},
                            {"name": "In Review", "color": "yellow"},
                            {"name": "Published", "color": "green"},
                            {"name": "Archived", "color": "red"}
                        ]
                    }
                },
                "Author": {"people": {}},
                "Created": {"created_time": {}},
                "Last Updated": {"last_edited_time": {}},
                "Tags": {"multi_select": {}}
            }
        )
        shared_dbs['research'] = research_db['id']
        
        print("[OK] Created shared databases")
        return shared_dbs
    
    async def _create_project_workspace(self, parent_id: str, project_key: str, config: Dict) -> Dict:
        """Create dedicated workspace for a single project"""
        
        project_resources = {}
        
        # 1. Create main project page
        project_page = self.notion.pages.create(
            parent={"page_id": parent_id},
            properties={
                "title": [{"text": {"content": f"{config['emoji']} {config['name']}"}}]
            },
            children=[
                {
                    "object": "block",
                    "type": "heading_1",
                    "heading_1": {
                        "rich_text": [{"text": {"content": f"{config['emoji']} {config['name']}"}}]
                    }
                },
                {
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [{"text": {"content": config['description']}}]
                    }
                },
                {
                    "object": "block",
                    "type": "heading_2",
                    "heading_2": {
                        "rich_text": [{"text": {"content": "🎯 Project Overview"}}]
                    }
                },
                {
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [{"text": {"content": f"Tech Stack: {', '.join(config['tech_stack'])}"}}]
                    }
                },
                {
                    "object": "block",
                    "type": "paragraph", 
                    "paragraph": {
                        "rich_text": [{"text": {"content": f"Key Priorities: {', '.join(config['priorities'])}"}}]
                    }
                }
            ]
        )
        project_resources['page'] = project_page['id']
        
        # 2. Create project task database
        tasks_db = await self._create_project_tasks_database(
            project_page['id'], 
            project_key, 
            config
        )
        project_resources['tasks_db'] = tasks_db['id']
        
        # 3. Create project documentation database
        docs_db = await self._create_project_docs_database(
            project_page['id'], 
            project_key, 
            config
        )
        project_resources['docs_db'] = docs_db['id']
        
        # 4. Create progress tracking widgets
        await self._add_project_widgets(project_page['id'], project_key, config)
        
        return project_resources
    
    async def _create_project_tasks_database(self, project_page_id: str, project_key: str, config: Dict) -> Dict:
        """Create task database for specific project"""
        
        database = self.notion.databases.create(
            parent={"page_id": project_page_id},
            title=[{"text": {"content": f"📋 {config['name']} - Tasks"}}],
            properties={
                "Task": {"title": {}},
                "Status": {
                    "select": {
                        "options": [
                            {"name": "Backlog", "color": "gray"},
                            {"name": "Todo", "color": "default"},
                            {"name": "In Progress", "color": "yellow"},
                            {"name": "In Review", "color": "orange"},
                            {"name": "Done", "color": "green"},
                            {"name": "Blocked", "color": "red"}
                        ]
                    }
                },
                "Priority": {
                    "select": {
                        "options": [
                            {"name": "Critical", "color": "red"},
                            {"name": "High", "color": "orange"},
                            {"name": "Medium", "color": "yellow"},
                            {"name": "Low", "color": "gray"}
                        ]
                    }
                },
                "Component": {
                    "select": {
                        "options": [
                            {"name": component, "color": ["blue", "purple", "green", "yellow", "orange", "brown", "pink", "gray"][i % 8]}
                            for i, component in enumerate(config['components'])
                        ]
                    }
                },
                "Assignee": {"people": {}},
                "Due Date": {"date": {}},
                "Effort (Days)": {"number": {"format": "number"}},
                "Progress": {"number": {"format": "percent"}},
                "Tags": {"multi_select": {}},
                "Created": {"created_time": {}},
                "Last Updated": {"last_edited_time": {}}
            }
        )
        
        return database
    
    async def _create_project_docs_database(self, project_page_id: str, project_key: str, config: Dict) -> Dict:
        """Create documentation database for specific project"""
        
        database = self.notion.databases.create(
            parent={"page_id": project_page_id},
            title=[{"text": {"content": f"📚 {config['name']} - Documentation"}}],
            properties={
                "Document": {"title": {}},
                "Type": {
                    "select": {
                        "options": [
                            {"name": "API Documentation", "color": "blue"},
                            {"name": "Architecture", "color": "purple"},
                            {"name": "User Guide", "color": "green"},
                            {"name": "Technical Spec", "color": "orange"},
                            {"name": "Meeting Notes", "color": "yellow"},
                            {"name": "Research", "color": "pink"}
                        ]
                    }
                },
                "Status": {
                    "select": {
                        "options": [
                            {"name": "Draft", "color": "gray"},
                            {"name": "In Review", "color": "yellow"},
                            {"name": "Published", "color": "green"},
                            {"name": "Outdated", "color": "red"}
                        ]
                    }
                },
                "Component": {
                    "select": {
                        "options": [
                            {"name": component, "color": ["blue", "purple", "green", "yellow", "orange", "brown", "pink", "gray"][i % 8]}
                            for i, component in enumerate(config['components'])
                        ]
                    }
                },
                "Author": {"people": {}},
                "Created": {"created_time": {}},
                "Last Updated": {"last_edited_time": {}},
                "Version": {"rich_text": {}},
                "Tags": {"multi_select": {}}
            }
        )
        
        return database
    
    async def _add_project_widgets(self, project_page_id: str, project_key: str, config: Dict):
        """Add progress tracking widgets to project page"""
        
        widgets = [
            {
                "object": "block",
                "type": "heading_2",
                "heading_2": {
                    "rich_text": [{"text": {"content": "📊 Project Dashboard"}}]
                }
            },
            {
                "object": "block",
                "type": "column_list",
                "column_list": {
                    "children": [
                        {
                            "object": "block",
                            "type": "column",
                            "column": {
                                "children": [
                                    {
                                        "object": "block",
                                        "type": "callout",
                                        "callout": {
                                            "rich_text": [{"text": {"content": "Task Status\n\nTodo: 0\nIn Progress: 0\nDone: 0"}}],
                                            "icon": {"emoji": "📋"},
                                            "color": "gray_background"
                                        }
                                    }
                                ]
                            }
                        },
                        {
                            "object": "block",
                            "type": "column",
                            "column": {
                                "children": [
                                    {
                                        "object": "block",
                                        "type": "callout",
                                        "callout": {
                                            "rich_text": [{"text": {"content": "Progress Metrics\n\nCompletion: 0%\nVelocity: N/A\nBurndown: N/A"}}],
                                            "icon": {"emoji": "📈"},
                                            "color": "blue_background"
                                        }
                                    }
                                ]
                            }
                        },
                        {
                            "object": "block",
                            "type": "column",
                            "column": {
                                "children": [
                                    {
                                        "object": "block",
                                        "type": "callout",
                                        "callout": {
                                            "rich_text": [{"text": {"content": "Quality Metrics\n\nCode Coverage: N/A\nTest Pass Rate: N/A\nBug Count: 0"}}],
                                            "icon": {"emoji": "🔍"},
                                            "color": "green_background"
                                        }
                                    }
                                ]
                            }
                        }
                    ]
                }
            },
            {
                "object": "block",
                "type": "heading_2",
                "heading_2": {
                    "rich_text": [{"text": {"content": "🛠️ Key Components"}}]
                }
            }
        ]
        
        # Add component status
        for component in config['components']:
            widgets.append({
                "object": "block",
                "type": "toggle",
                "toggle": {
                    "rich_text": [{"text": {"content": f"{component} - Not Started"}}],
                    "children": [
                        {
                            "object": "block",
                            "type": "paragraph",
                            "paragraph": {
                                "rich_text": [{"text": {"content": f"Status: Planning\nTasks: 0\nProgress: 0%"}}]
                            }
                        }
                    ]
                }
            })
        
        for widget in widgets:
            self.notion.blocks.children.append(project_page_id, children=[widget])
        
        print(f"[OK] Added widgets for {config['name']}")
    
    async def _add_master_dashboard_widgets(self, dashboard_id: str):
        """Add cross-project tracking widgets to master dashboard"""
        
        widgets = [
            {
                "object": "block",
                "type": "heading_2",
                "heading_2": {
                    "rich_text": [{"text": {"content": "📊 Cross-Project Overview"}}]
                }
            },
            {
                "object": "block",
                "type": "table",
                "table": {
                    "table_width": 6,
                    "has_column_header": True,
                    "has_row_header": False,
                    "children": [
                        {
                            "object": "block",
                            "type": "table_row",
                            "table_row": {
                                "cells": [
                                    [{"text": {"content": "Project"}}],
                                    [{"text": {"content": "Status"}}],
                                    [{"text": {"content": "Progress"}}],
                                    [{"text": {"content": "Tasks"}}],
                                    [{"text": {"content": "Risk Level"}}],
                                    [{"text": {"content": "Next Milestone"}}]
                                ]
                            }
                        }
                    ]
                }
            }
        ]
        
        # Add row for each project
        for project_key, config in self.projects.items():
            project_row = {
                "object": "block",
                "type": "table_row",
                "table_row": {
                    "cells": [
                        [{"text": {"content": f"{config['emoji']} {config['name'][:20]}..."}}],
                        [{"text": {"content": "Planning"}}],
                        [{"text": {"content": "0%"}}],
                        [{"text": {"content": "0/0"}}],
                        [{"text": {"content": "Low"}}],
                        [{"text": {"content": "TBD"}}]
                    ]
                }
            }
            widgets.append(project_row)
        
        # Add project navigation section
        widgets.extend([
            {
                "object": "block",
                "type": "heading_2",
                "heading_2": {
                    "rich_text": [{"text": {"content": "🚀 Project Navigation"}}]
                }
            },
            {
                "object": "block",
                "type": "column_list",
                "column_list": {
                    "children": []
                }
            }
        ])
        
        # Create navigation columns
        nav_columns = []
        for project_key, config in self.projects.items():
            project_resources = self.created_resources['projects'].get(project_key, {})
            project_page_id = project_resources.get('page', '')
            
            column = {
                "object": "block",
                "type": "column",
                "column": {
                    "children": [
                        {
                            "object": "block",
                            "type": "callout",
                            "callout": {
                                "rich_text": [
                                    {"text": {"content": f"{config['name']}\n\n"}},
                                    {"text": {"content": config['description']}}
                                ],
                                "icon": {"emoji": config['emoji']},
                                "color": "default"
                            }
                        }
                    ]
                }
            }
            nav_columns.append(column)
        
        widgets[-1]["column_list"]["children"] = nav_columns
        
        # Group table and its rows together
        table_with_rows = []
        non_table_widgets = []
        current_table = None
        
        for widget in widgets:
            if widget.get("type") == "table":
                current_table = widget
                current_table["table"]["children"] = []
            elif widget.get("type") == "table_row" and current_table:
                current_table["table"]["children"].append(widget)
            else:
                if current_table:
                    table_with_rows.append(current_table)
                    current_table = None
                non_table_widgets.append(widget)
        
        # Add any remaining table
        if current_table:
            table_with_rows.append(current_table)
        
        # Append all widgets
        for widget in table_with_rows + non_table_widgets:
            try:
                self.notion.blocks.children.append(dashboard_id, children=[widget])
            except Exception as e:
                print(f"[WARNING] Could not add widget: {e}")
        
        print("[OK] Added master dashboard widgets")
    
    async def _create_configuration_files(self):
        """Create configuration files for the workspace"""
        
        # 1. Environment configuration
        env_content = f"""# Multi-Project Notion Workspace Configuration
# Generated by setup_multi_project_dashboard.py on {datetime.now().isoformat()}

# Notion API Token
NOTION_TOKEN={os.getenv('NOTION_TOKEN', 'your_notion_token_here')}

# Master Dashboard
NOTION_MASTER_DASHBOARD={self.created_resources.get('master_dashboard', '')}

# Shared Databases
NOTION_ADR_DB={self.created_resources['shared_databases'].get('adr', '')}
NOTION_DEPENDENCIES_DB={self.created_resources['shared_databases'].get('dependencies', '')}
NOTION_SHARED_RESEARCH_DB={self.created_resources['shared_databases'].get('research', '')}

# Project-Specific Resources
"""
        
        for project_key, resources in self.created_resources['projects'].items():
            env_content += f"""
# {project_key.replace('_', ' ').title()} Project
NOTION_{project_key.upper()}_PAGE={resources.get('page', '')}
NOTION_{project_key.upper()}_TASKS_DB={resources.get('tasks_db', '')}
NOTION_{project_key.upper()}_DOCS_DB={resources.get('docs_db', '')}
"""
        
        with open('.env.multi_project', 'w', encoding='utf-8') as f:
            f.write(env_content)
        
        # 2. Project configuration JSON
        config_data = {
            'workspace_name': 'Multi-Project Crypto Trading Workspace',
            'created_at': datetime.now().isoformat(),
            'projects': self.projects,
            'notion_resources': self.created_resources
        }
        
        with open('multi_project_config.json', 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        
        # 3. Setup completion guide
        guide_content = f"""# Multi-Project Workspace Setup Complete!

## [SUCCESS] Created Resources

### Master Dashboard
- Main Workspace: https://notion.so/{self.created_resources.get('master_dashboard', '').replace('-', '')}

### Shared Databases
- Architecture Decisions: https://notion.so/{self.created_resources['shared_databases'].get('adr', '').replace('-', '')}
- Cross-Project Dependencies: https://notion.so/{self.created_resources['shared_databases'].get('dependencies', '').replace('-', '')}
- Shared Research: https://notion.so/{self.created_resources['shared_databases'].get('research', '').replace('-', '')}

### Project Workspaces
"""
        
        for project_key, config in self.projects.items():
            resources = self.created_resources['projects'].get(project_key, {})
            guide_content += f"""
#### {config['emoji']} {config['name']}
- Project Page: https://notion.so/{resources.get('page', '').replace('-', '')}
- Task Database: https://notion.so/{resources.get('tasks_db', '').replace('-', '')}
- Documentation: https://notion.so/{resources.get('docs_db', '').replace('-', '')}
"""
        
        guide_content += f"""

## Next Steps

### 1. Configure Environment
- Copy API token from main .env file to .env.multi_project
- Import environment variables: `source .env.multi_project`

### 2. Initialize Each Project
For each project, create initial tasks and documentation:

```python
# Example for Trading Core project
from multi_project_manager import MultiProjectManager

manager = MultiProjectManager()
await manager.initialize_project('trading_core')
```

### 3. Set Up Cross-Project Dependencies
- Map API dependencies between projects
- Define data flow requirements
- Document integration points

### 4. Create Initial Architecture Decisions
- Add ADRs for major architectural choices
- Define project boundaries and interfaces
- Document technology stack decisions

### 5. Import Existing Work
- Migrate existing tasks from other systems
- Import documentation and research
- Set up initial project baselines

## Configuration Files Created

- `.env.multi_project` - Environment variables
- `multi_project_config.json` - Project configuration
- `MULTI_PROJECT_SETUP_COMPLETE.md` - This guide

## Usage Examples

### Creating Tasks
```python
from notion_project_manager import NotionProjectManager

# Create task in Trading Core project
manager = NotionProjectManager('trading_core')
await manager.create_task(
    title="Implement WebSocket connection pool",
    component="Connection Pool", 
    priority="High"
)
```

### Cross-Project Dependencies
```python
# Add dependency between projects
await manager.add_cross_project_dependency(
    from_project="trading_core",
    to_project="ml_engine", 
    dependency_type="API",
    description="ML signals feed"
)
```

### Architecture Decisions
```python
# Record architecture decision
await manager.create_architecture_decision(
    title="Use Redis for real-time data cache",
    affects_projects=["trading_core", "dashboard"],
    context="Need sub-millisecond data access",
    decision="Redis in-memory cache with persistence"
)
```

## Resource IDs (for reference)
{json.dumps(self.created_resources, indent=2)}
"""
        
        with open('MULTI_PROJECT_SETUP_COMPLETE.md', 'w', encoding='utf-8') as f:
            f.write(guide_content)
        
        print("[OK] Created configuration files")
        print("[INFO] Setup guide saved to: MULTI_PROJECT_SETUP_COMPLETE.md")


async def main():
    """Main setup function"""
    
    # Check Notion token
    notion_token = os.getenv("NOTION_TOKEN")
    if not notion_token:
        print("[ERROR] NOTION_TOKEN not found in environment variables")
        print("Please set: export NOTION_TOKEN=your_token_here")
        return
    
    # Get parent page ID
    print("\n[INFO] Multi-Project Crypto Trading Workspace Setup")
    print("="*60)
    parent_page = input("Enter the parent page ID or URL where workspace should be created: ").strip()
    
    # Extract ID from URL if needed
    if "notion.so" in parent_page:
        parent_page = parent_page.split("/")[-1].split("-")[-1]
    
    # Format ID with hyphens if needed
    if len(parent_page) == 32 and "-" not in parent_page:
        parent_page = f"{parent_page[:8]}-{parent_page[8:12]}-{parent_page[12:16]}-{parent_page[16:20]}-{parent_page[20:]}"
    
    print(f"\nUsing parent page ID: {parent_page}")
    print("\n[INFO] This will create:")
    print("  - 1 Master dashboard page")
    print("  - 3 Shared databases (ADR, Dependencies, Research)")
    print("  - 4 Project workspaces with task and doc databases each")
    print("  - Progress tracking widgets")
    print("  - Configuration files")
    
    confirm = input("\nProceed with setup? (y/n): ").strip().lower()
    
    if confirm != 'y':
        print("Setup cancelled.")
        return
    
    # Setup workspace
    setup = MultiProjectDashboardSetup(notion_token)
    resources = await setup.setup_complete_workspace(parent_page)
    
    print("\n" + "="*60)
    print("[SUCCESS] Multi-Project Workspace Setup Complete!")
    print("="*60)
    print(f"\n[INFO] Created {len(resources['projects'])} project workspaces")
    print(f"[INFO] Created {len(resources['shared_databases'])} shared databases")
    print("\n[INFO] Check MULTI_PROJECT_SETUP_COMPLETE.md for next steps and URLs!")


if __name__ == "__main__":
    asyncio.run(main())