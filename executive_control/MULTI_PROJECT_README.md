# Multi-Project Crypto Trading Workspace

A comprehensive Notion-based project management system for 4 independent crypto trading system projects.

## 🎯 Overview

This system creates and manages dedicated Notion workspaces for:

1. **⚡ Trading Core** - Ultra-Low Latency Execution Engine
2. **🤖 ML Engine** - Quantitative Research & Execution Platform  
3. **📊 Dashboard** - Professional Trading Terminal
4. **🛡️ Risk Management** - Automated Risk Controls

Each project gets its own dedicated workspace with task tracking, documentation databases, progress monitoring, and cross-project dependency management.

## 🚀 Quick Start

### Prerequisites

1. **Notion Account** with API access
2. **Python 3.8+** with required packages
3. **Notion Integration** set up

### Installation

1. **Get Notion API Token**:
   - Go to [https://www.notion.so/my-integrations](https://www.notion.so/my-integrations)
   - Create new integration
   - Copy the API token

2. **Set Environment Variable**:
   ```bash
   export NOTION_TOKEN=your_token_here
   # Or add to .env file
   echo "NOTION_TOKEN=your_token_here" >> .env
   ```

3. **Install Dependencies**:
   ```bash
   pip install notion-client asyncio
   ```

### Quick Setup

Run the interactive setup script:

```bash
cd executive_control
python run_multi_project_setup.py
```

This will:
- Create master dashboard page
- Set up 4 project workspaces 
- Create shared databases (ADR, Dependencies, Research)
- Add progress tracking widgets
- Optionally populate with sample data

## 📁 Files Created

| File | Purpose |
|------|---------|
| `setup_multi_project_dashboard.py` | Main setup script - creates Notion workspace structure |
| `notion_project_manager.py` | Helper class for managing projects and tasks |
| `run_multi_project_setup.py` | Interactive setup runner with options |
| `.env.multi_project` | Environment variables (auto-generated) |
| `multi_project_config.json` | Project configuration (auto-generated) |
| `MULTI_PROJECT_SETUP_COMPLETE.md` | Setup guide with URLs (auto-generated) |

## 🏗️ Workspace Structure

### Master Dashboard
- Cross-project overview
- Project navigation
- Shared metrics

### Each Project Workspace Contains:
- **Project Overview Page** - Description, tech stack, priorities
- **Task Database** - Kanban-style task tracking
- **Documentation Database** - API docs, architecture, guides
- **Progress Widgets** - Real-time metrics and status

### Shared Databases:
- **Architecture Decision Records (ADR)** - Cross-project decisions
- **Dependencies** - Inter-project dependencies
- **Shared Research** - Common documentation and studies

## 💻 Usage Examples

### Creating Tasks

```python
from notion_project_manager import NotionProjectManager, Priority, TaskStatus

# Initialize manager
manager = NotionProjectManager(notion_token)

# Create task in Trading Core project
task_id = await manager.create_task(
    project_key="trading_core",
    title="Implement WebSocket connection pool",
    component="Connection Pool",
    priority=Priority.HIGH,
    description="Create connection pool for managing WebSocket connections",
    tags=["networking", "performance"]
)
```

### Managing Documentation

```python
from notion_project_manager import DocType

# Create API documentation
doc_id = await manager.create_documentation(
    project_key="ml_engine",
    title="ML Prediction API",
    doc_type=DocType.API_DOCS,
    component="Inference Engine",
    content="API endpoints for getting ML predictions...",
    version="1.0.0"
)
```

### Cross-Project Dependencies

```python
# Add dependency between projects
await manager.add_cross_project_dependency(
    from_project="Dashboard",
    to_project="Trading Core",
    dependency_type="API",
    description="Dashboard needs real-time trading data"
)
```

### Architecture Decisions

```python
# Record architecture decision
await manager.create_architecture_decision(
    title="Use Redis for real-time data cache",
    decision_id="ADR-001",
    affects_projects=["Trading Core", "Dashboard"],
    context="Need sub-millisecond data access for trading",
    decision="Redis in-memory cache with persistence",
    consequences="Fast access but memory overhead"
)
```

### Project Statistics

```python
# Get project statistics
stats = await manager.get_project_stats("trading_core")

print(f"Total tasks: {stats['total_tasks']}")
print(f"Completion rate: {stats['completion_rate']:.1%}")
print(f"Status distribution: {stats['status_distribution']}")
```

## 🎛️ Project Configurations

### Trading Core - Ultra-Low Latency Execution Engine
- **Tech Stack**: C++, WebSocket, FIX Protocol, Redis
- **Components**: Order Execution, Market Data Handler, Latency Monitor, Connection Pool
- **Focus**: Sub-millisecond latency, high throughput

### ML Engine - Quantitative Research & Execution Platform
- **Tech Stack**: Python, PyTorch, Pandas, NumPy, Ray
- **Components**: Feature Engineering, Model Training, Inference Engine, Backtesting
- **Focus**: Prediction accuracy, model performance

### Dashboard - Professional Trading Terminal
- **Tech Stack**: React, TypeScript, WebSocket, D3.js
- **Components**: Real-time Charts, Portfolio View, Risk Metrics, Alert System
- **Focus**: User experience, real-time updates

### Risk Management - Automated Risk Controls
- **Tech Stack**: Python, PostgreSQL, Redis, Kafka
- **Components**: Position Sizing, Risk Limits, Drawdown Control, Emergency Stop
- **Focus**: Safety, compliance, automation

## 🔧 Advanced Usage

### Batch Operations

```python
# Initialize all projects with sample data
for project_key in ['trading_core', 'ml_engine', 'dashboard', 'risk_management']:
    await initialize_project_with_sample_data(manager, project_key)
```

### Progress Tracking

```python
# Update task progress
await manager.update_task_progress(
    project_key="trading_core",
    task_id="task_page_id",
    progress=0.75,  # 75% complete
    status=TaskStatus.IN_PROGRESS
)
```

### Getting Project Tasks

```python
# Get all tasks for a project
tasks = await manager.get_project_tasks("ml_engine")

# Get only pending tasks
pending_tasks = await manager.get_project_tasks(
    "ml_engine", 
    status_filter=TaskStatus.TODO
)
```

## 📊 Database Properties

### Task Database Properties
- **Task** (Title) - Task name
- **Status** - Backlog, Todo, In Progress, In Review, Done, Blocked
- **Priority** - Critical, High, Medium, Low
- **Component** - Which component this task belongs to
- **Assignee** - Person responsible
- **Due Date** - Target completion date
- **Effort (Days)** - Estimated effort
- **Progress** - Completion percentage
- **Tags** - Categorization tags

### Documentation Database Properties
- **Document** (Title) - Document name
- **Type** - API Docs, Architecture, User Guide, etc.
- **Status** - Draft, In Review, Published, Outdated
- **Component** - Related component
- **Author** - Document creator
- **Version** - Document version
- **Tags** - Categorization tags

### Architecture Decision Records (ADR)
- **Title** - Decision name
- **Decision ID** - Unique identifier
- **Status** - Proposed, Accepted, Deprecated, Superseded
- **Affects Projects** - Which projects are impacted
- **Context** - Background and reasoning
- **Decision** - What was decided
- **Consequences** - Expected outcomes

## 🔄 Workflow Integration

### Development Workflow
1. **Planning** - Create tasks in appropriate project workspace
2. **Implementation** - Update progress as work is completed
3. **Documentation** - Create/update docs in project doc database
4. **Review** - Use In Review status for code reviews
5. **Completion** - Mark tasks as Done

### Cross-Project Coordination
1. **Dependencies** - Track what each project needs from others
2. **Architecture Decisions** - Record decisions affecting multiple projects
3. **Shared Research** - Maintain common knowledge base

## 🛠️ Customization

### Adding New Projects
1. Update project configuration in `multi_project_config.json`
2. Run setup script again with new configuration
3. Initialize new project with sample data

### Custom Task Properties
Modify database schemas in `setup_multi_project_dashboard.py`:

```python
# Add custom property
"Estimated Value": {"number": {"format": "dollar"}},
"Risk Level": {
    "select": {
        "options": [
            {"name": "Low", "color": "green"},
            {"name": "Medium", "color": "yellow"}, 
            {"name": "High", "color": "red"}
        ]
    }
}
```

### Custom Components
Update project configurations to add new components:

```python
'components': ['Order Execution', 'Market Data Handler', 'Your New Component']
```

## 🚨 Troubleshooting

### Common Issues

1. **"Database not found" error**
   - Check that setup completed successfully
   - Verify database IDs in configuration files
   - Ensure Notion integration has access to workspace

2. **"Permission denied" error**
   - Make sure Notion integration is shared with the workspace
   - Check that API token has correct permissions

3. **"Property not found" error**
   - Database schema might have changed
   - Re-run setup to recreate databases with correct schema

### Debug Mode

Run with debug information:

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Run operations
await manager.create_task(...)
```

### Validation

Check workspace integrity:

```bash
# Run demo to test all connections
python run_multi_project_setup.py demo
```

## 📈 Best Practices

### Task Management
- Use specific, actionable task titles
- Set realistic effort estimates
- Update progress regularly
- Use tags for categorization
- Set due dates for accountability

### Documentation
- Keep docs up-to-date with code changes
- Use consistent formatting
- Version important documents
- Cross-reference between projects

### Dependencies
- Document all inter-project dependencies
- Update when APIs change
- Consider impact of changes across projects
- Plan dependency updates carefully

### Architecture Decisions
- Record all significant technical decisions
- Include context and alternatives considered
- Update status when decisions change
- Review regularly for relevance

## 🔮 Future Enhancements

Planned improvements:
- **API Integration** - Connect with GitHub, JIRA, etc.
- **Automation** - Auto-update from code repositories
- **Analytics** - Advanced project metrics and reporting
- **Templates** - Pre-built project templates
- **Notifications** - Slack/email integration
- **Time Tracking** - Built-in time logging

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review generated setup guide
3. Examine configuration files
4. Test with demo mode

## 📄 License

This project management system is part of the Kimchi Premium Arbitrage trading system. Use according to project guidelines.