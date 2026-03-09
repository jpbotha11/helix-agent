import os
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_ollama import ChatOllama
from langfuse import Langfuse
from Config import LLM_PROVIDER

# =========================================================
# LANGFUSE
# =========================================================
langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_BASE_URL")
)

current_trace = None  # Set at runtime in main.py


# =========================================================
# LLM INIT
# =========================================================
def get_llm(provider: str = None):
    """
    Initialize LLM based on provider.

    Args:
        provider: "azure", "ollama", or "lmstudio".
                  If None, uses LLM_PROVIDER from config.

    Returns:
        Configured LLM instance
    """
    if provider is None:
        provider = LLM_PROVIDER.lower()

    print(f"[LLM] Initializing {provider.upper()} provider...")

    if provider == "azure":
        return AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2024-02-01",
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4")
        )

    elif provider == "ollama":
        return ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "llama3.1"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=0
        )

    elif provider == "lmstudio":
        return ChatOpenAI(
            base_url=os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1"),
            api_key="lm-studio",
            model=os.getenv("LMSTUDIO_MODEL", "local-model"),
            temperature=0
        )

    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'azure', 'ollama', or 'lmstudio'")


llm = get_llm()


# =========================================================
# LLM INVOCATION WITH LANGFUSE TRACING
# =========================================================
def invoke_llm(prompt: str, node_name: str) -> str:
    global current_trace

    if current_trace:
        generation = current_trace.generation(
            name=f"{node_name}-llm-call",
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            input=[{"role": "user", "content": prompt}],
        )

        raw = llm.invoke(prompt)
        response = raw.content

        generation.end(
            output=response,
            usage={
                "input": raw.response_metadata.get("token_usage", {}).get("prompt_tokens", 0),
                "output": raw.response_metadata.get("token_usage", {}).get("completion_tokens", 0),
                "total": raw.response_metadata.get("token_usage", {}).get("total_tokens", 0),
            },
            metadata={
                "node": node_name,
                "step": current_trace.id,
            }
        )
    else:
        response = llm.invoke(prompt).content

    return response