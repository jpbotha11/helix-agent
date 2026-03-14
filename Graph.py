from langgraph.graph import StateGraph, END
from State import AgentState
from nodes.Generate import (
    planner_node,
    code_writer_node,
    dependency_installer_node,
    docker_runner_node,
    reviewer_node,
    finalizer_node,
)
from nodes.Modify import (
    codebase_reader_node,
    impact_analyzer_node,
    code_modifier_node,
    diff_generator_node,
    human_approval_node,
)


# =========================================================
# LOOP CONTROL
# =========================================================
def should_continue(state: AgentState):
    if state["current_step"] >= len(state["plan"]):
        print("[LOOP CONTROL] All steps completed. Finishing.")
        return "finish"
    print(f"[LOOP CONTROL] Moving to next step: {state['plan'][state['current_step']]}")
    return "continue"


def should_continue_modification(state: AgentState):
    if state["approval_status"] == "approved":
        print("[APPROVAL] Changes approved -- running code")
        return "run"
    elif state["approval_status"] == "rejected" and state.get("modification_request"):
        print("[APPROVAL] Changes rejected -- retrying with new instructions")
        return "retry"
    else:
        print("[APPROVAL] Changes rejected -- stopping")
        return "finish"


def should_require_approval(state: AgentState):
    # ✅ Only ask for approval if there are actual changes
    if state.get("diff") and state["diff"].strip():
        print("[APPROVAL] Changes detected -- requesting human approval")
        return "approve"
    else:
        print("[APPROVAL] No changes detected -- auto approving")
        return "run"


# =========================================================
# GRAPH BUILDER
# =========================================================
def build_graph(mode: str):
    graph = StateGraph(AgentState)

    if mode == "modify":
        graph.add_node("codebase_reader", codebase_reader_node)
        graph.add_node("impact_analyzer", impact_analyzer_node)
        graph.add_node("code_modifier", code_modifier_node)
        graph.add_node("diff_generator", diff_generator_node)
        graph.add_node("human_approval", human_approval_node)
        graph.add_node("docker_runner", docker_runner_node)
        graph.add_node("reviewer", reviewer_node)
        graph.add_node("finalizer", finalizer_node)

        graph.set_entry_point("codebase_reader")
        graph.add_edge("codebase_reader", "impact_analyzer")
        graph.add_edge("impact_analyzer", "code_modifier")
        graph.add_edge("code_modifier", "diff_generator")
        
          # ✅ Replace with conditional
        graph.add_conditional_edges(
            "diff_generator",
            should_require_approval,
            {
                "approve": "human_approval",  # changes exist -- ask user
                "run": "docker_runner"         # no changes -- skip straight to run
            }
        )

        graph.add_conditional_edges(
            "human_approval",
            should_continue_modification,
            {
                "run": "docker_runner",
                "retry": "code_modifier",
                "finish": "finalizer"
            }
        )

        graph.add_edge("docker_runner", "reviewer")
        graph.add_conditional_edges(
            "reviewer",
            should_continue,
            {
                "continue": "code_modifier",
                "finish": "finalizer"
            }
        )

    else:  # generate mode
        graph.add_node("planner", planner_node)
        graph.add_node("code_writer", code_writer_node)
        graph.add_node("dependency_installer", dependency_installer_node)
        graph.add_node("docker_runner", docker_runner_node)
        graph.add_node("reviewer", reviewer_node)
        graph.add_node("finalizer", finalizer_node)

        graph.set_entry_point("planner")
        graph.add_edge("planner", "dependency_installer")
        graph.add_edge("dependency_installer", "code_writer")
        graph.add_edge("code_writer", "docker_runner")
        graph.add_edge("docker_runner", "reviewer")

        graph.add_conditional_edges(
            "reviewer",
            should_continue,
            {
                "continue": "dependency_installer",
                "finish": "finalizer"
            }
        )

    graph.add_edge("finalizer", END)
    return graph.compile()