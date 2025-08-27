"""
Notion Project Manager - Helper class for managing multi-project workspace
Provides convenient methods for working with the created project databases
"""

import os
import json
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from notion_client import Client
from dataclasses import dataclass
from enum import Enum


class TaskStatus(Enum):
    BACKLOG = "Backlog"
    TODO = "Todo"  
    IN_PROGRESS = "In Progress"
    IN_REVIEW = "In Review"
    DONE = "Done"
    BLOCKED = "Blocked"


class Priority(Enum):
    CRITICAL = "Critical"
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class DocType(Enum):
    API_DOCS = "API Documentation"
    ARCHITECTURE = "Architecture"
    USER_GUIDE = "User Guide"
    TECH_SPEC = "Technical Spec"
    MEETING_NOTES = "Meeting Notes"
    RESEARCH = "Research"


@dataclass
class ProjectTask:
    """Represents a task in a project"""
    id: str
    title: str
    status: TaskStatus
    priority: Priority
    component: str
    assignee: Optional[str] = None
    due_date: Optional[str] = None
    effort_days: Optional[float] = None
    progress: float = 0.0
    description: str = ""
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


@dataclass
class ArchitectureDecision:
    """Represents an Architecture Decision Record"""
    id: str
    title: str
    decision_id: str
    status: str
    affects_projects: List[str]
    context: str
    decision: str
    consequences: str
    author: Optional[str] = None
    decision_date: Optional[str] = None


class NotionProjectManager:
    """
    Manages individual projects and cross-project operations
    """
    
    def __init__(self, notion_token: str, config_file: str = "multi_project_config.json"):
        """
        Initialize project manager
        
        Args:
            notion_token: Notion API token
            config_file: Path to project configuration file
        """
        self.notion = Client(auth=notion_token)
        self.config = self._load_config(config_file)
        self.resources = self.config.get('notion_resources', {})
        
        # Cache for database schemas
        self._db_schemas = {}
    
    def _load_config(self, config_file: str) -> Dict:
        """Load project configuration"""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"[ERROR] Configuration file {config_file} not found")
            print("Please run setup_multi_project_dashboard.py first")
            return {}
    
    async def create_task(self, 
                         project_key: str,
                         title: str,
                         component: str,
                         priority: Priority = Priority.MEDIUM,
                         status: TaskStatus = TaskStatus.TODO,
                         description: str = "",
                         assignee: Optional[str] = None,
                         due_date: Optional[str] = None,
                         effort_days: Optional[float] = None,
                         tags: List[str] = None) -> str:
        """
        Create a new task in project
        
        Args:
            project_key: Project identifier (e.g., 'trading_core')
            title: Task title
            component: Component name
            priority: Task priority
            status: Task status  
            description: Detailed description
            assignee: Assigned person (Notion user ID or email)
            due_date: Due date in ISO format
            effort_days: Estimated effort in days
            tags: List of tags
            
        Returns:
            Created task page ID
        """
        
        project_resources = self.resources['projects'].get(project_key)
        if not project_resources:
            raise ValueError(f"Project {project_key} not found in configuration")
        
        tasks_db_id = project_resources['tasks_db']
        
        # Prepare properties
        properties = {
            "Task": {"title": [{"text": {"content": title}}]},
            "Status": {"select": {"name": status.value}},
            "Priority": {"select": {"name": priority.value}},
            "Component": {"select": {"name": component}},
        }
        
        if assignee:
            # If assignee is email, try to find user
            if "@" in assignee:
                properties["Assignee"] = {"people": [{"object": "user", "person": {"email": assignee}}]}
            else:
                properties["Assignee"] = {"people": [{"object": "user", "id": assignee}]}
        
        if due_date:
            properties["Due Date"] = {"date": {"start": due_date}}
            
        if effort_days is not None:
            properties["Effort (Days)"] = {"number": effort_days}
            
        if tags:
            properties["Tags"] = {
                "multi_select": [{"name": tag} for tag in tags]
            }
        
        # Create task blocks
        children = []
        if description:
            children.extend([
                {
                    "object": "block",
                    "type": "heading_2",
                    "heading_2": {"rich_text": [{"text": {"content": "Description"}}]}
                },
                {
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {"rich_text": [{"text": {"content": description}}]}
                }
            ])
        
        # Create acceptance criteria section
        children.extend([
            {
                "object": "block",
                "type": "heading_2",
                "heading_2": {"rich_text": [{"text": {"content": "Acceptance Criteria"}}]}
            },
            {
                "object": "block",
                "type": "to_do",
                "to_do": {
                    "rich_text": [{"text": {"content": "Define specific success criteria"}}],
                    "checked": False
                }
            }
        ])
        
        # Create the task page
        page = self.notion.pages.create(
            parent={"database_id": tasks_db_id},
            properties=properties,
            children=children
        )
        
        print(f"[OK] Created task '{title}' in {project_key}")
        return page['id']
    
    async def update_task_progress(self, 
                                  project_key: str, 
                                  task_id: str, 
                                  progress: float,
                                  status: Optional[TaskStatus] = None) -> bool:
        """
        Update task progress and optionally status
        
        Args:
            project_key: Project identifier
            task_id: Task page ID
            progress: Progress percentage (0.0 to 1.0)
            status: New status if changing
            
        Returns:
            Success boolean
        """
        
        try:
            properties = {
                "Progress": {"number": progress}
            }
            
            if status:
                properties["Status"] = {"select": {"name": status.value}}
            
            self.notion.pages.update(
                page_id=task_id,
                properties=properties
            )
            
            print(f"[OK] Updated task progress: {progress:.1%}")
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to update task: {e}")
            return False
    
    async def create_documentation(self,
                                  project_key: str,
                                  title: str,
                                  doc_type: DocType,
                                  component: str,
                                  content: str = "",
                                  author: Optional[str] = None,
                                  version: str = "1.0.0",
                                  tags: List[str] = None) -> str:
        """
        Create documentation page
        
        Args:
            project_key: Project identifier
            title: Document title
            doc_type: Type of documentation
            component: Related component
            content: Document content
            author: Author (Notion user ID or email)
            version: Document version
            tags: List of tags
            
        Returns:
            Created document page ID
        """
        
        project_resources = self.resources['projects'].get(project_key)
        if not project_resources:
            raise ValueError(f"Project {project_key} not found")
            
        docs_db_id = project_resources['docs_db']
        
        # Prepare properties
        properties = {
            "Document": {"title": [{"text": {"content": title}}]},
            "Type": {"select": {"name": doc_type.value}},
            "Component": {"select": {"name": component}},
            "Status": {"select": {"name": "Draft"}},
            "Version": {"rich_text": [{"text": {"content": version}}]}
        }
        
        if author:
            if "@" in author:
                properties["Author"] = {"people": [{"object": "user", "person": {"email": author}}]}
            else:
                properties["Author"] = {"people": [{"object": "user", "id": author}]}
        
        if tags:
            properties["Tags"] = {
                "multi_select": [{"name": tag} for tag in tags]
            }
        
        # Create content blocks
        children = []
        if content:
            # Split content into paragraphs
            paragraphs = content.split('\n\n')
            for para in paragraphs:
                if para.strip():
                    children.append({
                        "object": "block",
                        "type": "paragraph",
                        "paragraph": {"rich_text": [{"text": {"content": para.strip()}}]}
                    })
        else:
            children.append({
                "object": "block",
                "type": "paragraph",
                "paragraph": {"rich_text": [{"text": {"content": "Document content to be added..."}}]}
            })
        
        # Create the document page
        page = self.notion.pages.create(
            parent={"database_id": docs_db_id},
            properties=properties,
            children=children
        )
        
        print(f"[OK] Created documentation '{title}' in {project_key}")
        return page['id']
    
    async def create_architecture_decision(self,
                                         title: str,
                                         decision_id: str,
                                         affects_projects: List[str],
                                         context: str,
                                         decision: str,
                                         consequences: str,
                                         author: Optional[str] = None) -> str:
        """
        Create Architecture Decision Record
        
        Args:
            title: ADR title
            decision_id: Unique decision identifier
            affects_projects: List of affected projects
            context: Decision context
            decision: The decision made
            consequences: Decision consequences
            author: Decision author
            
        Returns:
            Created ADR page ID
        """
        
        adr_db_id = self.resources['shared_databases']['adr']
        
        properties = {
            "Title": {"title": [{"text": {"content": title}}]},
            "Decision ID": {"rich_text": [{"text": {"content": decision_id}}]},
            "Status": {"select": {"name": "Proposed"}},
            "Affects Projects": {
                "multi_select": [{"name": project} for project in affects_projects]
            },
            "Context": {"rich_text": [{"text": {"content": context[:2000]}}]},
            "Consequences": {"rich_text": [{"text": {"content": consequences[:2000]}}]},
            "Decision Date": {"date": {"start": datetime.now().isoformat()}}
        }
        
        if author:
            if "@" in author:
                properties["Author"] = {"people": [{"object": "user", "person": {"email": author}}]}
            else:
                properties["Author"] = {"people": [{"object": "user", "id": author}]}
        
        # Create ADR content
        children = [
            {
                "object": "block",
                "type": "heading_2",
                "heading_2": {"rich_text": [{"text": {"content": "Context"}}]}
            },
            {
                "object": "block",
                "type": "paragraph",
                "paragraph": {"rich_text": [{"text": {"content": context}}]}
            },
            {
                "object": "block",
                "type": "heading_2",
                "heading_2": {"rich_text": [{"text": {"content": "Decision"}}]}
            },
            {
                "object": "block",
                "type": "paragraph",
                "paragraph": {"rich_text": [{"text": {"content": decision}}]}
            },
            {
                "object": "block",
                "type": "heading_2",
                "heading_2": {"rich_text": [{"text": {"content": "Consequences"}}]}
            },
            {
                "object": "block",
                "type": "paragraph",
                "paragraph": {"rich_text": [{"text": {"content": consequences}}]}
            }
        ]
        
        page = self.notion.pages.create(
            parent={"database_id": adr_db_id},
            properties=properties,
            children=children
        )
        
        print(f"[OK] Created Architecture Decision Record: {title}")
        return page['id']
    
    async def add_cross_project_dependency(self,
                                         from_project: str,
                                         to_project: str,
                                         dependency_type: str,
                                         description: str,
                                         owner: Optional[str] = None) -> str:
        """
        Add cross-project dependency
        
        Args:
            from_project: Project that depends on another
            to_project: Project being depended upon
            dependency_type: Type of dependency (API, Data, Service, Library)
            description: Dependency description
            owner: Dependency owner
            
        Returns:
            Created dependency page ID
        """
        
        deps_db_id = self.resources['shared_databases']['dependencies']
        
        properties = {
            "Dependency": {"title": [{"text": {"content": f"{from_project} -> {to_project}"}}]},
            "From Project": {"select": {"name": from_project}},
            "To Project": {"select": {"name": to_project}},
            "Type": {"select": {"name": dependency_type}},
            "Status": {"select": {"name": "Active"}},
            "Description": {"rich_text": [{"text": {"content": description}}]}
        }
        
        if owner:
            if "@" in owner:
                properties["Owner"] = {"people": [{"object": "user", "person": {"email": owner}}]}
            else:
                properties["Owner"] = {"people": [{"object": "user", "id": owner}]}
        
        children = [
            {
                "object": "block",
                "type": "paragraph",
                "paragraph": {"rich_text": [{"text": {"content": description}}]}
            },
            {
                "object": "block",
                "type": "heading_2",
                "heading_2": {"rich_text": [{"text": {"content": "Implementation Details"}}]}
            },
            {
                "object": "block",
                "type": "bulleted_list_item",
                "bulleted_list_item": {"rich_text": [{"text": {"content": "API endpoints to be defined"}}]}
            },
            {
                "object": "block",
                "type": "bulleted_list_item",
                "bulleted_list_item": {"rich_text": [{"text": {"content": "Data schemas to be documented"}}]}
            },
            {
                "object": "block",
                "type": "bulleted_list_item",
                "bulleted_list_item": {"rich_text": [{"text": {"content": "Error handling to be specified"}}]}
            }
        ]
        
        page = self.notion.pages.create(
            parent={"database_id": deps_db_id},
            properties=properties,
            children=children
        )
        
        print(f"[OK] Created dependency: {from_project} -> {to_project}")
        return page['id']
    
    async def get_project_tasks(self, project_key: str, status_filter: Optional[TaskStatus] = None) -> List[ProjectTask]:
        """
        Get tasks for a project
        
        Args:
            project_key: Project identifier
            status_filter: Optional status filter
            
        Returns:
            List of project tasks
        """
        
        project_resources = self.resources['projects'].get(project_key)
        if not project_resources:
            raise ValueError(f"Project {project_key} not found")
        
        tasks_db_id = project_resources['tasks_db']
        
        # Build query filter
        query_filter = None
        if status_filter:
            query_filter = {
                "property": "Status",
                "select": {"equals": status_filter.value}
            }
        
        # Query database
        response = self.notion.databases.query(
            database_id=tasks_db_id,
            filter=query_filter,
            sorts=[{"property": "Created", "direction": "descending"}]
        )
        
        tasks = []
        for page in response['results']:
            props = page['properties']
            
            task = ProjectTask(
                id=page['id'],
                title=self._get_property_text(props.get('Task')),
                status=TaskStatus(self._get_property_select(props.get('Status', 'Todo'))),
                priority=Priority(self._get_property_select(props.get('Priority', 'Medium'))),
                component=self._get_property_select(props.get('Component', '')),
                assignee=self._get_property_people(props.get('Assignee')),
                due_date=self._get_property_date(props.get('Due Date')),
                effort_days=self._get_property_number(props.get('Effort (Days)')),
                progress=self._get_property_number(props.get('Progress', 0)),
                tags=self._get_property_multi_select(props.get('Tags'))
            )
            tasks.append(task)
        
        return tasks
    
    async def get_project_stats(self, project_key: str) -> Dict[str, Any]:
        """
        Get project statistics
        
        Args:
            project_key: Project identifier
            
        Returns:
            Dictionary with project statistics
        """
        
        tasks = await self.get_project_tasks(project_key)
        
        # Status distribution
        status_dist = {}
        for status in TaskStatus:
            status_dist[status.value] = sum(1 for t in tasks if t.status == status)
        
        # Priority distribution
        priority_dist = {}
        for priority in Priority:
            priority_dist[priority.value] = sum(1 for t in tasks if t.priority == priority)
        
        # Progress metrics
        total_tasks = len(tasks)
        completed_tasks = sum(1 for t in tasks if t.status == TaskStatus.DONE)
        avg_progress = sum(t.progress for t in tasks) / total_tasks if total_tasks > 0 else 0
        
        # Effort metrics
        total_effort = sum(t.effort_days or 0 for t in tasks)
        completed_effort = sum(t.effort_days or 0 for t in tasks if t.status == TaskStatus.DONE)
        
        return {
            'total_tasks': total_tasks,
            'completed_tasks': completed_tasks,
            'completion_rate': completed_tasks / total_tasks if total_tasks > 0 else 0,
            'average_progress': avg_progress,
            'status_distribution': status_dist,
            'priority_distribution': priority_dist,
            'total_effort_days': total_effort,
            'completed_effort_days': completed_effort,
            'effort_completion_rate': completed_effort / total_effort if total_effort > 0 else 0
        }
    
    # Helper methods for extracting Notion property values
    def _get_property_text(self, prop: Any) -> str:
        """Extract text from title or rich_text property"""
        if not prop:
            return ""
        
        prop_type = prop.get('type')
        if prop_type == 'title':
            texts = prop.get('title', [])
        elif prop_type == 'rich_text':
            texts = prop.get('rich_text', [])
        else:
            return ""
        
        return ' '.join([t['plain_text'] for t in texts]) if texts else ''
    
    def _get_property_select(self, prop: Any, default: str = "") -> str:
        """Extract select property value"""
        if not prop or not prop.get('select'):
            return default
        return prop['select']['name']
    
    def _get_property_multi_select(self, prop: Any) -> List[str]:
        """Extract multi-select property values"""
        if not prop or not prop.get('multi_select'):
            return []
        return [item['name'] for item in prop['multi_select']]
    
    def _get_property_number(self, prop: Any, default: float = 0.0) -> float:
        """Extract number property value"""
        if not prop:
            return default
        return prop.get('number', default)
    
    def _get_property_date(self, prop: Any) -> Optional[str]:
        """Extract date property value"""
        if not prop or not prop.get('date'):
            return None
        return prop['date'].get('start')
    
    def _get_property_people(self, prop: Any) -> Optional[str]:
        """Extract first person from people property"""
        if not prop or not prop.get('people'):
            return None
        people = prop['people']
        if people:
            return people[0].get('name') or people[0].get('id')
        return None


# Usage examples and initialization helpers
async def initialize_project_with_sample_data(manager: NotionProjectManager, project_key: str):
    """Initialize a project with sample tasks and documentation"""
    
    project_config = manager.config['projects'].get(project_key)
    if not project_config:
        print(f"[ERROR] Project {project_key} not found in configuration")
        return
    
    print(f"[INFO] Initializing {project_key} with sample data...")
    
    # Create sample tasks for each component
    for i, component in enumerate(project_config['components']):
        await manager.create_task(
            project_key=project_key,
            title=f"Implement {component} module",
            component=component,
            priority=Priority.HIGH if i == 0 else Priority.MEDIUM,
            description=f"Design and implement the {component} module for the {project_config['name']}",
            tags=["implementation", "core"]
        )
        
        await manager.create_task(
            project_key=project_key,
            title=f"Write tests for {component}",
            component=component,
            priority=Priority.MEDIUM,
            description=f"Create comprehensive unit and integration tests for {component}",
            tags=["testing", "quality"]
        )
    
    # Create sample documentation
    for doc_type in [DocType.ARCHITECTURE, DocType.API_DOCS, DocType.USER_GUIDE]:
        await manager.create_documentation(
            project_key=project_key,
            title=f"{project_config['name']} - {doc_type.value}",
            doc_type=doc_type,
            component=project_config['components'][0],  # Use first component
            content=f"This document covers the {doc_type.value.lower()} for the {project_config['name']}.",
            tags=["documentation", "reference"]
        )
    
    print(f"[OK] Initialized {project_key} with sample data")


# Example usage script
async def example_usage():
    """Example usage of the NotionProjectManager"""
    
    notion_token = os.getenv("NOTION_TOKEN")
    if not notion_token:
        print("[ERROR] NOTION_TOKEN not found")
        return
    
    manager = NotionProjectManager(notion_token)
    
    # Initialize all projects with sample data
    for project_key in ['trading_core', 'ml_engine', 'dashboard', 'risk_management']:
        await initialize_project_with_sample_data(manager, project_key)
    
    # Create some cross-project dependencies
    await manager.add_cross_project_dependency(
        from_project="Dashboard",
        to_project="Trading Core",
        dependency_type="API",
        description="Dashboard needs real-time trading data from Trading Core"
    )
    
    await manager.add_cross_project_dependency(
        from_project="Trading Core", 
        to_project="ML Engine",
        dependency_type="API",
        description="Trading Core needs ML signals for decision making"
    )
    
    # Create architecture decision
    await manager.create_architecture_decision(
        title="Use WebSocket for real-time data",
        decision_id="ADR-001",
        affects_projects=["Trading Core", "Dashboard"],
        context="Need real-time data streaming with low latency",
        decision="Implement WebSocket connections for all real-time data feeds",
        consequences="Lower latency but more complex connection management"
    )
    
    # Get project statistics
    for project_key in ['trading_core', 'ml_engine']:
        stats = await manager.get_project_stats(project_key)
        print(f"\n{project_key.replace('_', ' ').title()} Statistics:")
        print(f"  Total tasks: {stats['total_tasks']}")
        print(f"  Completion rate: {stats['completion_rate']:.1%}")
        print(f"  Status distribution: {stats['status_distribution']}")
    
    print("\n[SUCCESS] Example usage completed!")


if __name__ == "__main__":
    asyncio.run(example_usage())