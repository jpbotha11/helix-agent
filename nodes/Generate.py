import json
import subprocess
from State import AgentState
from LLM import invoke_llm, llm
from Utils import get_entry_point, strip_markdown_fences, normalize_reviewer_output, ensure_workdir, materialize_workspace
from Config import DOCKER_ENABLED, DOCKER_TIMEOUT


# =========================================================
# DOCKER HELPER
# =========================================================
def run_in_docker(workspace: dict, entry_point: str, dire) -> dict:
    print(f"[DOCKER] Running '{entry_point}' with files: {list(workspace.keys())}")

    cmd = [
        "docker", "run", "--rm",
        "-v", f"{dire}:/sandbox",
        "langgraph-python-sandbox",
        "sh", "-c",
        "pip install -r requirements.txt && python main.py"
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=DOCKER_TIMEOUT
    )

    stdout = result.stdout.decode("utf-8", errors="replace") if result.stdout else ""
    stderr = result.stderr.decode("utf-8", errors="replace") if result.stderr else ""

    print(f"[DOCKER] Exit code: {result.returncode}")
    print(f"[DOCKER] stdout: {stdout[:500]}")
    print(f"[DOCKER] stderr: {stderr[:500]}")

    return {"stdout": stdout, "stderr": stderr, "exit_code": result.returncode}


# =========================================================
# PLANNER
# =========================================================
def planner_node(state: AgentState) -> AgentState:
    print(f"[PLANNER] Planning project for goal: {state['user_goal']}")

    prompt = f"""
You are a senior software architect.

Goal:
{state['user_goal']}

Decide:
- Platform (python)
- Project structure
- Entry point
- Required Python dependencies (pip packages)
- Step-by-step plan

DEPENDENCY RULES:
- For FastAPI projects ALWAYS include: ["fastapi", "httpx", "uvicorn"]
- For Flask projects ALWAYS include: ["flask"]
- For SQLAlchemy projects ALWAYS include: ["sqlalchemy"]

IMPORTANT PLAN RULES:
- The plan should have NO MORE than 5 steps
- Step 1 MUST always be: "Create all project files including the entry point"
- The entry point file MUST be created in step 1
- Do not split file creation across multiple steps

Return a SINGLE JSON object with this exact shape:
{{
  "platform": "python",
  "entry_point": "main.py",
  "structure": {{
    "filename": "responsibility"
  }},
  "dependencies": ["package1", "package2"],
  "plan": [
    "Create all project files: models.py, routes.py, main.py",
    "Verify and test all endpoints work correctly"
  ]
}}
"""

    resp = invoke_llm(prompt, "planner")
    data = json.loads(strip_markdown_fences(resp))

    return {
        **state,
        "platform": data["platform"],
        "entry_point": data["entry_point"],
        "project_structure": data["structure"],
        "plan": data["plan"],
        "current_step": 0,
        "workspace": {},
        "dependencies": data.get("dependencies", []),
        "file_summaries": {}
    }


# =========================================================
# CODE WRITER
# =========================================================
def generate_code_writer_prompt(state: AgentState, step: str, error_context: str) -> str:
    print("[CODE WRITER] Generating platform-specific prompt...")

    meta_prompt = f"""
You are a prompt engineer. Generate a precise coding prompt for an LLM that will write code.

Platform: {state['platform']}
Goal: {state['user_goal']}
Current step: {step}
Project structure: {state['project_structure']}
Existing files: {list(state['workspace'].keys())}
Previous error: {state.get('execution_error') or 'none'}

Generate a prompt that:
1. Is specific to {state['platform']} best practices
2. Includes platform-specific syntax rules that are commonly gotten wrong
3. Addresses the previous error if there is one
4. Is clear and unambiguous

Return ONLY the prompt text, no explanation.
"""
    return llm.invoke(meta_prompt).content.strip()


def code_writer_node(state: AgentState) -> AgentState:
    print(f"[CODE WRITER] Writing code for step: {state['plan'][state['current_step']]}")
    step = state["plan"][state["current_step"]]

    if state.get("execution_error"):
        step = "Fix the execution error described above. Do NOT add new features or change any logic."
        error_context = f"""
PREVIOUS EXECUTION FAILED -- you MUST fix these errors:
{state['execution_error']}

Output before failure:
{state.get('execution_output') or '(none)'}

Existing file contents that need fixing:
{chr(10).join(f"### {k} ###{chr(10)}{v}" for k, v in state['workspace'].items())}
"""
    else:
        error_context = ""

    generated_prompt = generate_code_writer_prompt(state, step, error_context)

    prompt = f"""
{generated_prompt}

{error_context}

Return a SINGLE valid JSON object.
DO NOT use markdown.
DO NOT wrap the output in ```json or ``` blocks.
DO NOT add explanations or extra text.

The JSON must have exactly this shape:
{{
  "files": {{
    "filename": "file contents"
  }},
  "explanations": {{
    "filename": "what this file does"
  }}
}}

RULES
- You MUST create ALL files in the project structure in EVERY step
- You MUST ALWAYS include the entry point file: {state['entry_point']}
- NEVER leave out any file from the project structure
- If a file has not been written yet, write a complete working version of it now
"""

    raw = invoke_llm(prompt, "code_writer")
    clean = strip_markdown_fences(raw)

    try:
        data = json.loads(clean)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON from code writer:\n{clean}") from e

    state["workspace"].update(data["files"])
    state["file_summaries"].update(data["explanations"])
    print(f"[CODE WRITER] Files in workspace: {list(state['workspace'].keys())}")

    state["execution_error"] = None
    state["execution_output"] = None

    return state


# =========================================================
# DEPENDENCY INSTALLER
# =========================================================
def dependency_installer_node(state: AgentState) -> AgentState:
    print(f"[DEPENDENCY INSTALLER] Installing dependencies: {state['dependencies']}")

    if not state["dependencies"]:
        return state

    current_deps = sorted(state["dependencies"])
    installed_deps = sorted(state.get("installed_dependencies") or [])

    if current_deps == installed_deps:
        print("[DEPENDENCY INSTALLER] Dependencies unchanged -- skipping install")
        return state

    if not DOCKER_ENABLED:
        print("[DEPENDENCY INSTALLER] Docker disabled -- skipping install")
        state["workspace"]["requirements.txt"] = "\n".join(state["dependencies"])
        return state

    workDir = ensure_workdir(state)
    state["workdir"] = str(workDir)
    state["workspace"]["requirements.txt"] = "\n".join(state["dependencies"])

    # ✅ Only write requirements.txt -- docker_runner materializes all files
    req_path = workDir / "requirements.txt"
    req_path.write_text(state["workspace"]["requirements.txt"], encoding="utf-8")

    cmd = [
        "docker", "run", "--rm",
        "-v", f"{workDir}:/sandbox",
        "langgraph-python-sandbox",
        "pip", "install", "-r", "/sandbox/requirements.txt"
    ]

    print(f"[DOCKER] Running command: {' '.join(cmd)}")
    print(f"[DOCKER] Workspace dir: {workDir}")

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=DOCKER_TIMEOUT
    )

    stdout = result.stdout.decode("utf-8", errors="replace") if result.stdout else ""
    stderr = result.stderr.decode("utf-8", errors="replace") if result.stderr else ""

    print(f"[DOCKER] Exit code: {result.returncode}")
    print(f"[DOCKER] stdout: {stdout[:500]}")
    print(f"[DOCKER] stderr: {stderr[:500]}")

    return {
        **state,
        "execution_output": stdout,
        "execution_error": stderr if result.returncode != 0 else state.get("execution_error"),
        "installed_dependencies": state["dependencies"]
    }


# =========================================================
# DOCKER RUNNER
# =========================================================
def docker_runner_node(state: AgentState) -> AgentState:
    print(f"[DOCKER RUNNER] Running entry point '{state['entry_point']}' in Docker...")

    if not DOCKER_ENABLED:
        print("[DOCKER RUNNER] Docker disabled -- skipping execution")
        workdir = materialize_workspace(state)
        print(f"[DOCKER RUNNER] Files written to: {workdir}")
        return {
            **state,
            "execution_output": "(docker disabled -- code not executed)",
            "execution_error": None
        }

    #get the entry point
    #entry = state["entry_point"]
    entry = get_entry_point(state)
    

    if entry not in state["workspace"]:
        raise FileNotFoundError(
            f"Entry point '{entry}' not found in workspace files: "
            f"{list(state['workspace'].keys())}"
        )

    workdir = materialize_workspace(state)
    workdir = workdir.resolve()

    result = run_in_docker(state["workspace"], state["entry_point"], workdir)

    return {
        **state,
        "execution_output": result["stdout"] or "(no output captured)",
        "execution_error": result["stderr"] if result["stderr"].strip() else None
    }


# =========================================================
# REVIEWER
# =========================================================
def reviewer_node(state: AgentState) -> AgentState:
    print(f"[REVIEWER] Reviewing execution results for step: {state['plan'][state['current_step']]}")

    prompt = f"""
You are reviewing a Python project.
Goal:
{state['user_goal']}
Files:
{state['file_summaries']}
Execution output:
{state['execution_output'] or '(no output captured)'}
Execution error:
{state['execution_error'] or '(no errors)'}

Respond in ONE of the following ways ONLY:
1) APPROVED
OR
2) A SINGLE valid JSON object (no markdown, no text before or after)
   with EXACTLY this shape:
   {{
     "files": {{
       "filename": "corrected file content"
     }},
     "explanations": {{
       "filename": "what this file does"
     }},
     "dependencies": ["package1", "package2"]
   }}
The "dependencies" field must ALWAYS be the full list of required packages.
DO NOT prefix with words like 'FIXED FILES'.
DO NOT use markdown.

IMPORTANT RULES:
- If execution_error is "(no errors)", respond APPROVED.
- If execution_output is "(no output captured)" BUT execution_error is "(no errors)", respond APPROVED.
- If execution_output is "(docker disabled -- code not executed)", respond APPROVED.
- Only return JSON fixes if there is an actual error message.
- If there is a SyntaxError, identify the EXACT line and what is wrong.
- Return corrected files with the EXACT fix applied, return ALL files not just changed ones.
"""

    #reviewer can either approve or provide fixes in the same step, but if they approve we want to skip ALL remaining steps, not just move to the next one
    response = invoke_llm(prompt, "reviewer")

    #if complete the step successfully, allow reviewer to skip all remaining steps with "APPROVED"
    if response.strip() == "APPROVED":
        # ✅ Skip ALL remaining steps, not just increment by 1
        state["current_step"] = len(state["plan"])
        print(f"[REVIEWER] Approved -- skipping all remaining steps")
        return state

    clean = normalize_reviewer_output(response)

    try:
        data = json.loads(clean)
    except json.JSONDecodeError as e:
        raise ValueError(f"Reviewer returned invalid JSON:\n{clean}") from e

    state["workspace"].update(data["files"])
    state["file_summaries"].update(data["explanations"])

    if "dependencies" in data:
        state["dependencies"] = data["dependencies"]
        state["workspace"]["requirements.txt"] = "\n".join(data["dependencies"])

    return state


# =========================================================
# FINALIZER
# =========================================================
def finalizer_node(state: AgentState) -> AgentState:
    return {
        **state,
        "final_output": state["execution_output"]
    }