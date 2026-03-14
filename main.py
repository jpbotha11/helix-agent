import os

os.environ["PYTHONIOENCODING"] = "utf-8"

import LLM as llm_module
from LLM import langfuse
from Graph import build_graph
from Config import AGENT_MODE, WORKING_DIRECTORY

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
        "workdir": WORKING_DIRECTORY,
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
        "workdir": WORKING_DIRECTORY,
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
        Change the crawler to read the seed url from a text file called "seed_url.txt" instead of having it hardcoded in the script.
        The crawler should read the URL from the file and then proceed with the crawling process as before.
        Crawl 3 layers deep.
        Save the html of each file in a text file.
        Do not accept any command line arguments when running the script.

        """
        source_folder = WORKING_DIRECTORY
        app = build_graph("modify")
        initial_state = modify_initial_state(user_goal, source_folder)

    else:
        user_goal =  """
        Create me a basic web crawler that will start with a seed url, extract all the links from that page, and then visit each of those links and repeat the process for a total of 3 iterations. The crawler should save the results in a JSON file with the following format:
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