# Helix Agent

An autonomous AI coding agent built with [LangGraph](https://github.com/langchain-ai/langgraph) that generates, executes, and self-heals multi-file Python projects — and can intelligently modify existing codebases.

---

## Features

- **Generate mode** — describe a project in natural language, the agent plans, writes, executes, and validates it automatically
- **Modify mode** — point the agent at an existing codebase, describe what to change, review a diff, and approve before anything is applied
- **Self-healing loop** — if execution fails, the agent automatically fixes errors and retries
- **Multi-file projects** — generates proper project structure with separate files per responsibility
- **Docker sandboxing** — all code runs in an isolated Docker container
- **Langfuse tracing** — full observability of every LLM call, token usage, and loop iteration
- **Multi-provider LLM** — supports Azure OpenAI, Ollama, and LM Studio
- **Human approval** — diff preview before any changes are applied to existing code

---

## Architecture

```
agent/
├── main.py           # Entry point
├── config.py         # Environment variables and constants
├── state.py          # AgentState TypedDict
├── llm.py            # LLM init, Langfuse tracing, invoke_llm
├── utils.py          # Text helpers, workspace file management
├── graph.py          # LangGraph graph builder, loop control
└── nodes/
    ├── generate.py   # planner, code_writer, dependency_installer, docker_runner, reviewer, finalizer
    └── modify.py     # codebase_reader, impact_analyzer, code_modifier, diff_generator, human_approval
```

### Generate Mode Flow

```
planner → dependency_installer → code_writer → docker_runner → reviewer
                ▲                                                   │
                └───────────── continue (if error) ─────────────────┘
                                                                    │
                                             finish → finalizer → END
```

### Modify Mode Flow

```
codebase_reader → impact_analyzer → code_modifier → diff_generator
                                          ▲                │
                                          │         ┌──────┴──────┐
                                          │      approve        no changes
                                          │         │                │
                                          │   human_approval    docker_runner
                                          │         │                │
                                          └─────────┘           reviewer
                                          (if rejected)              │
                                                               finalizer → END
```

---

## Prerequisites

- Python 3.11+
- Docker Desktop
- An LLM provider (Azure OpenAI, Ollama, or LM Studio)
- Langfuse instance (local or cloud)

---

## Installation

```bash
git clone https://github.com/your-username/langgraph-coder.git
cd langgraph-coder
pip install -r requirements.txt
```

### Python dependencies

```
langgraph
langchain-openai
langchain-ollama
langfuse
python-dotenv
```

### Docker sandbox image

Build the Python sandbox image used to execute generated code:

```bash
docker build -f Dockerfile.sandbox -t langgraph-python-sandbox .
```

Example `Dockerfile.sandbox`:

```dockerfile
FROM python:3.11-slim
WORKDIR /sandbox
RUN pip install --upgrade pip
```

---

## Configuration

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

```env
# LLM Provider: "azure" | "ollama" | "lmstudio"
LLM_PROVIDER=azure

# Azure OpenAI
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o

# Ollama (if using ollama)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1

# LM Studio (if using lmstudio)
LMSTUDIO_BASE_URL=http://localhost:1234/v1
LMSTUDIO_MODEL=local-model

# Langfuse
LANGFUSE_PUBLIC_KEY=your-public-key
LANGFUSE_SECRET_KEY=your-secret-key
LANGFUSE_BASE_URL=http://localhost:3000

# Agent settings
AGENT_MODE=generate         # "generate" | "modify"
DOCKER_ENABLED=true         # set to false to skip Docker execution
DOCKER_TIMEOUT=1000         # seconds
```

---

## Usage

### Generate Mode

Set `AGENT_MODE=generate` in your `.env`, then edit the `user_goal` in `main.py`:

```python
user_goal = """
Create a multi-file Python app that builds a machine learning pipeline:
1. Generate a synthetic dataset with 200 samples using scikit-learn
2. Train and compare Logistic Regression, Random Forest, and KNN models
3. Save results to model_comparison.csv
4. Print a formatted summary table to console
Dependencies: scikit-learn, pandas
"""
```

Run:

```bash
python main.py
```

Generated files are saved to `./agent_workspaces/run_0/`.

### Modify Mode

Set `AGENT_MODE=modify` in your `.env`, then edit `main.py`:

```python
user_goal = "Add a feature importance analysis for the Random Forest model and save results to feature_importance.csv"
source_folder = "./agent_workspaces/run_0"  # path to existing project
```

Run:

```bash
python main.py
```

The agent will:
1. Read all files in the source folder
2. Analyze which files need to change
3. Apply modifications
4. Show a unified diff
5. Ask for your approval before executing

---

## Docker Feature Flag

To generate and inspect code without running it in Docker:

```env
DOCKER_ENABLED=false
```

Files are still written to disk at `./agent_workspaces/run_0/` so you can inspect and run them manually.

---

## Langfuse Tracing

Every agent run is traced in Langfuse with full visibility into:

- Each LLM call per node (planner, code_writer, reviewer, etc.)
- Input prompts and output responses
- Token usage per call
- Number of retry loop iterations

Set up a local Langfuse instance with Docker Compose or use [Langfuse Cloud](https://langfuse.com).

---

## Example Projects Tested

| Project | Dependencies | Mode |
|---|---|---|
| SQLAlchemy ORM with student/teacher tables | sqlalchemy | generate |
| Pandas data processing with CSV output | pandas, faker | generate |
| Flask REST API with TestClient | flask | generate |
| FastAPI REST API with TestClient | fastapi, httpx | generate |
| scikit-learn ML pipeline comparison | scikit-learn, pandas | generate |
| Add DELETE endpoint to FastAPI project | — | modify |
| Add feature importance to ML pipeline | — | modify |

---

## Roadmap

- [ ] Node.js platform support
- [ ] Java platform support
- [ ] Large codebase support — file indexing + embeddings for 100s of files
- [ ] Max retry limit to prevent infinite loops
- [ ] Web UI for diff approval
- [ ] GitHub PR integration for modify mode

---

## License

MIT
