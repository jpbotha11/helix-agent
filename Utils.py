import shutil
from pathlib import Path
from LLM import invoke_llm
from State import AgentState
from Config import WORKSPACE_ROOT


# =========================================================
# TEXT HELPERS
# =========================================================
def strip_markdown_fences(text: str) -> str:
    text = text.strip()

    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1:]
        else:
            text = text[3:]

        if "```" in text:
            text = text.rsplit("```", 1)[0]

    return text.strip()


def normalize_reviewer_output(text: str) -> str:
    text = text.strip()

    for prefix in ("FIXED FILES:", "FIXED:", "FILES:"):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()

    if text.startswith("```"):
        text = text.split("```", 1)[1]
        if "```" in text:
            text = text.rsplit("```", 1)[0]

    return text.strip()


# =========================================================
# WORKSPACE HELPERS
# =========================================================
def ensure_workdir(state: AgentState) -> Path:
    if state["workdir"]:
        return Path(state["workdir"])

    run_id = f"run_{state['current_step']}"
    workdir = (WORKSPACE_ROOT / run_id).resolve()

    shutil.rmtree(workdir, ignore_errors=True)
    workdir.mkdir(parents=True, exist_ok=True)

    state["workdir"] = str(workdir)
    return workdir


def materialize_workspace(state: AgentState) -> Path:
    workdir = ensure_workdir(state)

    for name, content in state["workspace"].items():
        path = workdir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    return workdir

def get_entry_point(state: AgentState) -> str:
    if state.get("mode") == "modify":
        # LLM determines entry point from existing files
        file_list = "\n".join(state["workspace"].keys())
        prompt = f"""
Given these project files, which is the main entry point to run the project?
Return ONLY the filename, nothing else.

Files:
{file_list}
"""
        return invoke_llm(prompt, "entry_point_detector").strip()

    # Generate mode - use what the planner set
    entry = state["entry_point"]
    if entry not in state["workspace"]:
        raise FileNotFoundError(
            f"Entry point '{entry}' not found in workspace files: "
            f"{list(state['workspace'].keys())}"
        )
    return entry