"""
Codebase retriever.
Semantic search over indexed chunks, expands via relationship graph,
and manages context budget to avoid hitting LLM token limits.
"""

import os
import tiktoken
from pathlib import Path
from typing import Dict, List, Set, Tuple

from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient

from vector.VectorGraph import FileRelationshipGraph
from vector.Indexer import QDRANT_COLLECTION, get_embedding_model

# =========================================================
# CONFIG
# =========================================================
QDRANT_URL = os.getenv("QDRANT_HOST", "http://localhost:6333")
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "60000"))  # leave room for prompt + response
RETRIEVAL_LIMIT = int(os.getenv("RETRIEVAL_LIMIT", "15"))            # top N chunks from vector search
RELATIONSHIP_DEPTH = int(os.getenv("RELATIONSHIP_DEPTH", "2"))       # hops in relationship graph


# =========================================================
# RETRIEVER
# =========================================================
class CodebaseRetriever:

    def __init__(self, graph: FileRelationshipGraph, source_folder: str):
        self.graph = graph
        self.source_folder = source_folder
        self.embedder = get_embedding_model()
        self.client = QdrantClient(url=QDRANT_URL)

        # Use cl100k tokenizer (GPT-4 compatible) for token counting
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self.tokenizer = None

    def retrieve(self, modification_request: str) -> Tuple[Dict[str, str], List[str]]:
        """
        Main retrieval pipeline:
        1. Semantic search for relevant chunks
        2. Expand via relationship graph
        3. Trim to context budget
        4. Return file contents + summary of excluded files

        Returns:
            - files: Dict[filepath, content] -- files within context budget
            - excluded_summaries: List[str] -- summaries of files that were cut
        """
        print(f"[RETRIEVER] Retrieving context for: {modification_request[:80]}...")

        # Step 1 -- semantic search
        candidate_files = self._semantic_search(modification_request)
        print(f"[RETRIEVER] Semantic search returned {len(candidate_files)} candidate files")

        # Step 2 -- expand via relationship graph
        expanded_files = self.graph.expand_impact(
            list(candidate_files),
            depth=RELATIONSHIP_DEPTH
        )
        print(f"[RETRIEVER] After relationship expansion: {len(expanded_files)} files")

        # Step 3 -- load file contents
        file_contents = self._load_files(expanded_files)

        # Step 4 -- trim to context budget
        within_budget, excluded = self._trim_to_budget(
            file_contents,
            candidate_files  # prioritize semantically relevant files
        )

        print(f"[RETRIEVER] Within budget: {len(within_budget)} files")
        print(f"[RETRIEVER] Excluded (summarized): {len(excluded)} files")

        # Step 5 -- summarize excluded files
        excluded_summaries = self._summarize_excluded(excluded)

        return within_budget, excluded_summaries

    # =========================================================
    # PRIVATE
    # =========================================================
    def _semantic_search(self, query: str) -> Set[str]:
        vector = self.embedder.embed_query(query)

        results = self.client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=vector,
            limit=RETRIEVAL_LIMIT,
        )

        files = set()
        for r in results.points:
            path = r.payload.get("file_path")
            if path:
                files.add(path)

        return files

    def _load_files(self, file_paths: Set[str]) -> Dict[str, str]:
        contents = {}
        base = Path(self.source_folder)

        for rel_path in file_paths:
            full_path = base / rel_path
            try:
                contents[rel_path] = full_path.read_text(encoding="utf-8", errors="replace")
            except Exception as e:
                print(f"[RETRIEVER] Could not load {rel_path}: {e}")

        return contents

    def _trim_to_budget(
        self,
        files: Dict[str, str],
        priority_files: Set[str]
    ) -> Tuple[Dict[str, str], Dict[str, str]]:
        """
        Fit files into MAX_CONTEXT_TOKENS budget.
        Priority files (semantically relevant) go first.
        Relationship-expanded files fill remaining budget.
        """
        within_budget = {}
        excluded = {}
        used_tokens = 0

        # Sort: priority files first, then relationship-expanded
        sorted_files = (
            [(f, c) for f, c in files.items() if f in priority_files] +
            [(f, c) for f, c in files.items() if f not in priority_files]
        )

        for file_path, content in sorted_files:
            tokens = self._count_tokens(content)

            if used_tokens + tokens <= MAX_CONTEXT_TOKENS:
                within_budget[file_path] = content
                used_tokens += tokens
            else:
                excluded[file_path] = content

        return within_budget, excluded

    def _summarize_excluded(self, excluded: Dict[str, str]) -> List[str]:
        """
        For files that didn't fit in context, generate a one-line summary
        so the LLM still knows they exist and what they do.
        """
        summaries = []
        for file_path, content in excluded.items():
            lines = content.splitlines()
            preview = " | ".join(
                line.strip() for line in lines[:5]
                if line.strip() and not line.strip().startswith("#")
            )[:200]
            summaries.append(
                f"{file_path} ({len(lines)} lines) -- {preview}"
            )
        return summaries

    def _count_tokens(self, text: str) -> int:
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        # Fallback: rough estimate
        return len(text) // 4