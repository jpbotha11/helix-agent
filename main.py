import os

os.environ["PYTHONIOENCODING"] = "utf-8"

import LLM as llm_module
from LLM import langfuse
from Graph import build_graph
from Config import AGENT_MODE

# =========================================================
# INITIAL STATE TEMPLATES
# =========================================================
def generate_initial_state(user_goal: str) -> dict:
    return {
        "mode": "generate",
        "source_folder": None,
        "modification_request": None,
        "original_files": {},
        "modified_files": {},
        "impact_analysis": None,
        "diff": None,
        "approval_status": None,
        "user_goal": user_goal,
        "platform": "",
        "entry_point": "",
        "project_structure": {},
        "plan": [],
        "current_step": 0,
        "workspace": {},
        "workdir": None,
        "file_summaries": {},
        "execution_output": None,
        "execution_error": None,
        "final_output": None,
        "installed_dependencies": [],
        "dependencies": []
    }


def modify_initial_state(user_goal: str, source_folder: str) -> dict:
    return {
        "mode": "modify",
        "source_folder": source_folder,
        "modification_request": user_goal,
        "user_goal": user_goal,
        "original_files": {},
        "modified_files": {},
        "impact_analysis": None,
        "diff": None,
        "approval_status": "pending",
        "platform": "python",
        "entry_point": "main.py",
        "project_structure": {},
        "plan": [],
        "current_step": 0,
        "workspace": {},
        "workdir": None,
        "file_summaries": {},
        "execution_output": None,
        "execution_error": None,
        "final_output": None,
        "installed_dependencies": [],
        "dependencies": []
    }


# =========================================================
# RUN
# =========================================================
if __name__ == "__main__":
    mode = AGENT_MODE

    if mode == "modify":
        user_goal = """
        Add a feature importance analysis to the existing ML pipeline:

1. For the Random Forest model, extract and print feature importances.
2. Save feature importances to a new CSV file called feature_importance.csv 
   with columns: feature, importance.
3. Print the top feature with highest importance to console.

        """
        source_folder = "./agent_workspaces/run_0"
        app = build_graph("modify")
        initial_state = modify_initial_state(user_goal, source_folder)

    else:
        user_goal = """
Create a multi-file Python app that builds a machine learning pipeline:

1. Generate a synthetic dataset with 200 samples and 4 features using 
   scikit-learn make_classification with 2 classes.

2. Split data into 80% train and 20% test sets.

3. Train three models and compare them:
   - Logistic Regression
   - Random Forest
   - K-Nearest Neighbors

4. For each model evaluate and print:
   - Accuracy
   - Precision
   - Recall
   - F1 Score

5. Save results to a CSV file called model_comparison.csv.

6. Print a formatted summary table to console showing all models and metrics.

Use separate files for: dataset generation, model training, evaluation, and main entry point.
Dependencies: scikit-learn, pandas
"""
        app = build_graph("generate")
        initial_state = generate_initial_state(user_goal)

    # ✅ Start Langfuse trace
    llm_module.current_trace = langfuse.trace(
        name="langgraph-coder",
        input={"user_goal": user_goal, "mode": mode}
    )

    result = app.invoke(initial_state)

    # ✅ End trace
    llm_module.current_trace.update(output={"final_output": result["final_output"]})
    langfuse.flush()

    print("\n✅ FINAL OUTPUT:\n")
    print(result["final_output"])
    print("----------------------------------------------")
    print("--------------------DONE----------------------")
    print("----------------------------------------------")