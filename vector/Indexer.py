"""
Codebase indexer.
Scans all files, chunks them, embeds with LM Studio/Azure,
and stores in Qdrant with metadata for retrieval.
"""

import hashlib
import os
from pathlib import Path
from typing import Dict, List

from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
)

from vector.Chunker import FileChunker, FileChunk, LANGUAGE_MAP
from vector.VectorGraph import FileRelationshipGraph
from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings

# =========================================================
# CONFIG
# =========================================================
QDRANT_URL = os.getenv("QDRANT_HOST", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "helix_codebase")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "768"))  # nomic-embed = 768
SUPPORTED_EXTENSIONS = {
    ".py", ".js", ".ts", ".java", ".cs", ".go",
    ".rb", ".php", ".cpp", ".c", ".md", ".json",
    ".yaml", ".yml", ".toml", ".txt"
}
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "50"))

#ensure the collection exists on startup



# =========================================================
# EMBEDDING MODEL
# =========================================================
def get_embedding_model():
    provider = os.getenv("LLM_PROVIDER", "azure").lower()

    if provider == "lmstudio":
        return OpenAIEmbeddings(
            model=os.getenv("LMSTUDIO_EMBEDDING_MODEL", "text-embedding-nomic-embed-text-v1.5"),
            base_url=os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1"),
            openai_api_key="lm-studio",
            check_embedding_ctx_length=False
        )
    elif provider == "azure":
        return AzureOpenAIEmbeddings(
            model=os.getenv("AZURE_EMBEDDING_MODEL", "text-embedding-3-small"),
            azure_endpoint=os.getenv("AZURE_EMBEDDING_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_deployment=os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-3-small"),
        )
    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")


# =========================================================
# INDEXER
# =========================================================
class CodebaseIndexer:

    def __init__(self):
        self.chunker = FileChunker()
        self.embedder = get_embedding_model()
        self.client = QdrantClient(url=QDRANT_URL)
        self.graph = FileRelationshipGraph()
        #self._ensure_collection()
        self._reset_collection()

    def index(self, source_folder: str) -> FileRelationshipGraph:
        """
        Full re-index of a codebase folder.
        Returns the relationship graph for use in retrieval.
        """
        print(f"[INDEXER] Scanning: {source_folder}")

        files = self._scan_files(source_folder)
        print(f"[INDEXER] Found {len(files)} files to index")

        # Build relationship graph first
        self.graph.build(files)

        # Clear existing collection and re-index
        self._reset_collection()

        # Chunk, embed, store
        all_chunks = []
        for file_path, content in files.items():
            chunks = self.chunker.chunk_file(file_path, content)
            all_chunks.extend(chunks)

        print(f"[INDEXER] Total chunks to embed: {len(all_chunks)}")

        # Batch embed for efficiency
        self._embed_and_store(all_chunks)

        print(f"[INDEXER] Indexing complete")
        return self.graph

    # =========================================================
    # PRIVATE
    # =========================================================
    def _scan_files(self, source_folder: str) -> Dict[str, str]:
        folder = Path(source_folder)
        files = {}

        for path in folder.rglob("*"):
            if (
                path.is_file()
                and path.suffix.lower() in SUPPORTED_EXTENSIONS
                and not any(p in str(path) for p in [
                    ".git", "node_modules", "__pycache__",
                    ".venv", "venv", "dist", "build"
                ])
            ):
                try:
                    relative = str(path.relative_to(folder))
                    files[relative] = path.read_text(encoding="utf-8", errors="replace")
                except Exception as e:
                    print(f"[INDEXER] Skipping {path}: {e}")

        return files

    def _embed_and_store(self, chunks: List[FileChunk], batch_size: int = EMBEDDING_BATCH_SIZE) -> None:
        points = []

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            texts = [c.text for c in batch]

            print(f"[INDEXER] Embedding batch {i // batch_size + 1}/{(len(chunks) - 1) // batch_size + 1}")

            try:
                vectors = self.embedder.embed_documents(texts)
            except Exception as e:
                print(f"[INDEXER] Embedding error: {e}")
                continue

            for chunk, vector in zip(batch, vectors):
                point_id = int(hashlib.md5(
                    f"{chunk.file_path}:{chunk.start_line}".encode()
                ).hexdigest()[:8], 16)

                points.append(PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "file_path": chunk.file_path,
                        "language": chunk.language,
                        "chunk_type": chunk.chunk_type,
                        "symbol_name": chunk.symbol_name,
                        "start_line": chunk.start_line,
                        "end_line": chunk.end_line,
                        "text": chunk.text[:500],  # store preview only
                    }
                ))

        if points:
            self.client.upsert(
                collection_name=QDRANT_COLLECTION,
                points=points
            )
            print(f"[INDEXER] Stored {len(points)} chunks in Qdrant")

    def _ensure_collection(self) -> None:
        existing = [c.name for c in self.client.get_collections().collections]
        if QDRANT_COLLECTION not in existing:
            self.client.create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIM,
                    distance=Distance.COSINE
                )
            )
            print(f"[INDEXER] Created collection: {QDRANT_COLLECTION}")

    def _reset_collection(self) -> None:
        self.client.delete_collection(QDRANT_COLLECTION)
        self.client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(
                size=EMBEDDING_DIM,
                distance=Distance.COSINE
            )
        )
        print(f"[INDEXER] Reset collection: {QDRANT_COLLECTION}")