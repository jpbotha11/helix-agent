"""
Modify mode nodes.
Uses vector search + relationship graph to intelligently
select relevant files from large codebases.
"""

import json
import difflib
from pathlib import Path
from typing import Dict

from State import AgentState
from LLM import invoke_llm
from Utils import strip_markdown_fences
from Config import SUPPORTED_EXTENSIONS
from vector.Indexer import CodebaseIndexer
from vector.Retriever import CodebaseRetriever


# =========================================================
# CODEBASE READER + INDEXER
# =========================================================
def codebase_reader_node(state: AgentState) -> AgentState:
    folder = Path(state["source_folder"])
    files = {}

    for path in folder.rglob("*"):
        if (
            path.is_file()
            and path.suffix.lower() in SUPPORTED_EXTENSIONS
            and not any(p in str(path) for p in [
                ".git", "node_modules", "__pycache__", ".venv", "venv"
            ])
        ):
            relative = str(path.relative_to(folder))
            try:
                files[relative] = path.read_text(encoding="utf-8", errors="replace")
                print(f"[READER] Loaded: {relative}")
            except Exception as e:
                print(f"[READER] Skipping {relative}: {e}")

    print(f"[READER] Total files loaded: {len(files)}")

    # Index into Qdrant and build relationship graph
    print("[READER] Indexing codebase into vector store...")
    indexer = CodebaseIndexer()
    graph = indexer.index(state["source_folder"])

    return {
        **state,
        "original_files": files,
        "workspace": files.copy(),
        "vector_graph": graph,
        "vector_indexed": True
    }


# =========================================================
# IMPACT ANALYZER
# =========================================================
def impact_analyzer_node(state: AgentState) -> AgentState:
    print("[IMPACT ANALYZER] Analyzing codebase...")

    graph = state.get("vector_graph")
    retriever = CodebaseRetriever(graph, state["source_folder"])

    relevant_files, excluded_summaries = retriever.retrieve(
        state["modification_request"]
    )

    print(f"[IMPACT] Retrieved {len(relevant_files)} relevant files")
    if excluded_summaries:
        print(f"[IMPACT] {len(excluded_summaries)} files excluded (summarized)")

    file_context = chr(10).join(
        f"### {k} ###\n{v}" for k, v in relevant_files.items()
    )

    excluded_context = ""
    if excluded_summaries:
        excluded_context = f"""
The following files exist but were excluded from context due to size limits.
Be aware of them when making changes to avoid breaking imports:
{chr(10).join(f"- {s}" for s in excluded_summaries)}
"""

    prompt = f"""
You are a senior software engineer analyzing a codebase.

Modification request:
{state['modification_request']}

Relevant file contents:
{file_context}

{excluded_context}

Analyze and return a SINGLE JSON object:
{{
  "impacted_files": {{
    "filename": "reason why this file needs to change"
  }},
  "new_files": {{
    "filename": "reason why this new file is needed"
  }},
  "deleted_files": [],
  "plan": ["step 1", "step 2"],
  "summary": "brief explanation of all changes needed",
  "risk_files": ["files that could break if not updated"]
}}
"""

    response = invoke_llm(prompt, "impact_analyzer")
    data = json.loads(strip_markdown_fences(response))

    print(f"[IMPACT] Files to modify: {list(data['impacted_files'].keys())}")
    print(f"[IMPACT] New files: {list(data['new_files'].keys())}")
    print(f"[IMPACT] Risk files: {data.get('risk_files', [])}")
    print(f"[IMPACT] Summary: {data['summary']}")

    return {
        **state,
        "impact_analysis": data,
        "relevant_files": relevant_files,
        "excluded_summaries": excluded_summaries,
        "plan": data["plan"],
        "current_step": 0
    }


# =========================================================
# CODE MODIFIER
# =========================================================
def code_modifier_node(state: AgentState) -> AgentState:
    print("[CODE MODIFIER] Applying modifications...")

    impact = state["impact_analysis"]
    relevant_files = state.get("relevant_files", state["original_files"])
    excluded_summaries = state.get("excluded_summaries", [])

    excluded_context = ""
    if excluded_summaries:
        excluded_context = f"""
The following files were NOT included in context due to size limits.
Do NOT break their imports or interfaces when making changes:
{chr(10).join(f"- {s}" for s in excluded_summaries)}
"""

    prompt = f"""
You are implementing code modifications.

Modification request:
{state['modification_request']}

Files that need changing:
{impact['impacted_files']}

New files needed:
{impact['new_files']}

Files to delete:
{impact['deleted_files']}

Risk files (must not break):
{impact.get('risk_files', [])}

Current file contents:
{chr(10).join(f"### {k} ###{chr(10)}{v}" for k, v in relevant_files.items())}

{excluded_context}

Return a SINGLE JSON object with ALL modified files in their final state:
{{
  "files": {{
    "filename": "complete updated file content"
  }},
  "explanations": {{
    "filename": "what changed and why"
  }}
}}

RULES:
- Return COMPLETE file content for every file, not just changed parts
- For unchanged files you have context for, return them exactly as they are
- For deleted files, do NOT include them in the output
- Only modify what is necessary for the request
- Do NOT change interfaces or function signatures of files you cannot see
- DO NOT use markdown
"""

    response = invoke_llm(prompt, "code_modifier")
    data = json.loads(strip_markdown_fences(response))

    # Merge modified files back into full original_files
    merged = {**state["original_files"], **data["files"]}

    return {
        **state,
        "modified_files": data["files"],
        "workspace": merged,
        "file_summaries": data["explanations"]
    }


# =========================================================
# DIFF GENERATOR
# =========================================================
def diff_generator_node(state: AgentState) -> AgentState:
    print("[DIFF GENERATOR] Generating diff...")

    diffs = []
    all_files = set(
        list(state["original_files"].keys()) +
        list(state["modified_files"].keys())
    )

    for filename in sorted(all_files):
        original = state["original_files"].get(filename, "")
        modified = state["modified_files"].get(filename, "")

        if original == modified:
            continue

        if not original:
            label = f"[NEW FILE] {filename}"
        elif not modified:
            label = f"[DELETED] {filename}"
        else:
            label = f"[MODIFIED] {filename}"

        diff = list(difflib.unified_diff(
            original.splitlines(keepends=True),
            modified.splitlines(keepends=True),
            fromfile=f"original/{filename}",
            tofile=f"modified/{filename}",
            lineterm=""
        ))

        diffs.append(f"\n{'='*60}\n{label}\n{'='*60}")
        diffs.extend(diff)

    full_diff = "\n".join(diffs)

    print("\n[DIFF PREVIEW]")
    print(full_diff if full_diff.strip() else "(no changes detected)")

    return {**state, "diff": full_diff}


# =========================================================
# HUMAN APPROVAL
# =========================================================
def human_approval_node(state: AgentState) -> AgentState:
    # Skip if no changes
    if not state.get("diff") or not state["diff"].strip():
        print("[APPROVAL] No changes detected -- auto approving")
        return {
            **state,
            "approval_status": "approved",
            "workspace": state["modified_files"]
        }

    print("\n" + "="*60)
    print("APPROVAL REQUIRED")
    print("="*60)
    print(state["diff"])
    print("\nDo you want to apply these changes?")
    print("  [y] Yes   -- apply changes")
    print("  [n] No    -- reject and stop")
    print("  [e] Edit  -- provide new instructions and retry")

    choice = input("\nYour choice: ").strip().lower()

    if choice == "y":
        return {
            **state,
            "approval_status": "approved",
            "workspace": state["modified_files"]
        }
    elif choice == "e":
        new_instruction = input("New instruction: ").strip()
        return {
            **state,
            "approval_status": "rejected",
            "modification_request": (
                f"{state['modification_request']}\n\nAdditional instruction: {new_instruction}"
            )
        }
    else:
        return {**state, "approval_status": "rejected"}

