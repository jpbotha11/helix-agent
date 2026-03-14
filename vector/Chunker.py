"""
Language-agnostic chunker.
Splits files into meaningful chunks using structural markers
that work across Python, JS, Java, TS, and other languages.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


# =========================================================
# CHUNK MODEL
# =========================================================
@dataclass
class FileChunk:
    file_path: str
    language: str
    chunk_type: str        # "function" | "class" | "import" | "block" | "file"
    text: str
    start_line: int
    end_line: int
    symbol_name: str = ""  # function/class name if applicable
    file_hash: str = ""


# =========================================================
# LANGUAGE CONFIG
# =========================================================
LANGUAGE_MAP = {
    ".py":   "python",
    ".js":   "javascript",
    ".ts":   "typescript",
    ".java": "java",
    ".cs":   "csharp",
    ".go":   "go",
    ".rb":   "ruby",
    ".php":  "php",
    ".cpp":  "cpp",
    ".c":    "c",
    ".md":   "markdown",
    ".json": "json",
    ".yaml": "yaml",
    ".yml":  "yaml",
    ".toml": "toml",
    ".txt":  "text",
}

# Structural markers per language -- used to split files into logical chunks
STRUCTURAL_PATTERNS = {
    "python": [
        (r"^(async\s+)?def\s+\w+", "function"),
        (r"^class\s+\w+", "class"),
        (r"^(import|from)\s+", "import"),
    ],
    "javascript": [
        (r"^(async\s+)?function\s+\w+", "function"),
        (r"^(const|let|var)\s+\w+\s*=\s*(async\s+)?\(", "function"),
        (r"^class\s+\w+", "class"),
        (r"^(import|require|export)\s+", "import"),
    ],
    "typescript": [
        (r"^(async\s+)?function\s+\w+", "function"),
        (r"^(const|let|var)\s+\w+\s*=\s*(async\s+)?\(", "function"),
        (r"^class\s+\w+", "class"),
        (r"^interface\s+\w+", "class"),
        (r"^(import|export)\s+", "import"),
    ],
    "java": [
        (r"^\s*(public|private|protected|static).*\s+\w+\s*\(", "function"),
        (r"^\s*(public|private|protected)?\s*class\s+\w+", "class"),
        (r"^import\s+", "import"),
    ],
    "csharp": [
        (r"^\s*(public|private|protected|static).*\s+\w+\s*\(", "function"),
        (r"^\s*(public|private|protected)?\s*class\s+\w+", "class"),
        (r"^using\s+", "import"),
    ],
    "go": [
        (r"^func\s+\w+", "function"),
        (r"^type\s+\w+\s+struct", "class"),
        (r"^import\s+", "import"),
    ],
}

# Fallback for unknown languages -- chunk by blank line separation
DEFAULT_CHUNK_SIZE = 50  # lines


# =========================================================
# CHUNKER
# =========================================================
class FileChunker:

    def chunk_file(self, file_path: str, content: str) -> List[FileChunk]:
        path = Path(file_path)
        language = LANGUAGE_MAP.get(path.suffix.lower(), "text")
        lines = content.splitlines()

        if language in STRUCTURAL_PATTERNS:
            return self._chunk_by_structure(file_path, language, lines)
        else:
            return self._chunk_by_lines(file_path, language, lines)

    def _chunk_by_structure(
        self, file_path: str, language: str, lines: List[str]
    ) -> List[FileChunk]:
        patterns = STRUCTURAL_PATTERNS[language]
        chunks = []
        current_start = 0
        current_type = "block"
        current_name = ""
        imports = []

        i = 0
        while i < len(lines):
            line = lines[i]
            matched = False

            for pattern, chunk_type in patterns:
                if re.match(pattern, line.strip()):
                    # Save previous chunk
                    if i > current_start:
                        chunk_text = "\n".join(lines[current_start:i])
                        if chunk_text.strip():
                            chunks.append(FileChunk(
                                file_path=file_path,
                                language=language,
                                chunk_type=current_type,
                                text=chunk_text,
                                start_line=current_start + 1,
                                end_line=i,
                                symbol_name=current_name
                            ))

                    current_start = i
                    current_type = chunk_type
                    current_name = self._extract_symbol_name(line)
                    matched = True
                    break

            i += 1

        # Save final chunk
        if current_start < len(lines):
            chunk_text = "\n".join(lines[current_start:])
            if chunk_text.strip():
                chunks.append(FileChunk(
                    file_path=file_path,
                    language=language,
                    chunk_type=current_type,
                    text=chunk_text,
                    start_line=current_start + 1,
                    end_line=len(lines),
                    symbol_name=current_name
                ))

        # If file is small or no structure found, return as single chunk
        if not chunks:
            chunks.append(FileChunk(
                file_path=file_path,
                language=language,
                chunk_type="file",
                text="\n".join(lines),
                start_line=1,
                end_line=len(lines),
                symbol_name=""
            ))

        return chunks

    def _chunk_by_lines(
        self, file_path: str, language: str, lines: List[str]
    ) -> List[FileChunk]:
        chunks = []
        for i in range(0, len(lines), DEFAULT_CHUNK_SIZE):
            chunk_lines = lines[i:i + DEFAULT_CHUNK_SIZE]
            chunk_text = "\n".join(chunk_lines)
            if chunk_text.strip():
                chunks.append(FileChunk(
                    file_path=file_path,
                    language=language,
                    chunk_type="block",
                    text=chunk_text,
                    start_line=i + 1,
                    end_line=min(i + DEFAULT_CHUNK_SIZE, len(lines)),
                    symbol_name=""
                ))
        return chunks

    def _extract_symbol_name(self, line: str) -> str:
        match = re.search(r"\b(\w+)\s*[\(\{:]", line)
        return match.group(1) if match else ""