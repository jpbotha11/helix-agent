from typing import TypedDict, Dict, List, Optional


class AgentState(TypedDict):
    # User intent
    user_goal: str

    # Dependency tracking
    installed_dependencies: List[str]

    # Planner outputs
    platform: str                   # "python" | "node"
    entry_point: str                # e.g. "main.py"
    project_structure: Dict[str, str]
    dependencies: List[str]         # e.g. ["flask", "sqlalchemy"]
    workdir: Optional[str]          # absolute host path

    plan: List[str]
    current_step: int

    # Workspace
    workspace: Dict[str, str]       # filename -> content
    file_summaries: Dict[str, str]

    # Execution feedback
    execution_output: Optional[str]
    execution_error: Optional[str]

    # Modification mode fields
    mode: str                           # "generate" | "modify"
    source_folder: Optional[str]
    modification_request: Optional[str]
    original_files: Optional[Dict[str, str]]
    modified_files: Optional[Dict[str, str]]
    impact_analysis: Optional[dict]
    diff: Optional[str]
    approval_status: Optional[str]      # "pending" | "approved" | "rejected"

    # Final result
    final_output: Optional[str]