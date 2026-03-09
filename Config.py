import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# =========================================================
# ENVIRONMENT
# =========================================================
DOCKER_ENABLED = os.getenv("DOCKER_ENABLED", "true").lower() == "true"
DOCKER_TIMEOUT = int(os.getenv("DOCKER_TIMEOUT", "1000"))
AGENT_MODE = os.getenv("AGENT_MODE", "generate")  # "generate" | "modify"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "azure")

# =========================================================
# PATHS
# =========================================================
WORKSPACE_ROOT = Path("./agent_workspaces")
WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)

# =========================================================
# SUPPORTED FILE EXTENSIONS FOR CODEBASE READER
# =========================================================
SUPPORTED_EXTENSIONS = {".py", ".js", ".java", ".ts", ".json", ".txt", ".md"}