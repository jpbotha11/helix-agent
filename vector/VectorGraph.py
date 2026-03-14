"""
File relationship graph.
Parses imports/requires/includes across languages to build
a dependency graph so we know which files depend on which.

When a file is selected for modification, we can expand
to include all files that import it (dependents) to avoid
breaking the build.
"""

import re
from pathlib import Path
from typing import Dict, List, Set
from collections import defaultdict


# =========================================================
# IMPORT PATTERNS PER LANGUAGE
# =========================================================
IMPORT_PATTERNS = {
    "python": [
        r"^from\s+([\w.]+)\s+import",
        r"^import\s+([\w.]+)",
    ],
    "javascript": [
        r"from\s+['\"]([^'\"]+)['\"]",
        r"require\s*\(\s*['\"]([^'\"]+)['\"]\s*\)",
    ],
    "typescript": [
        r"from\s+['\"]([^'\"]+)['\"]",
        r"require\s*\(\s*['\"]([^'\"]+)['\"]\s*\)",
    ],
    "java": [
        r"^import\s+([\w.]+);",
    ],
    "csharp": [
        r"^using\s+([\w.]+);",
    ],
    "go": [
        r"['\"]([^'\"]+)['\"]",
    ],
}

LANGUAGE_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".java": "java",
    ".cs": "csharp",
    ".go": "go",
}


# =========================================================
# RELATIONSHIP GRAPH
# =========================================================
class FileRelationshipGraph:

    def __init__(self):
        # file -> set of files it imports
        self.dependencies: Dict[str, Set[str]] = defaultdict(set)
        # file -> set of files that import it
        self.dependents: Dict[str, Set[str]] = defaultdict(set)
        # all known files in the codebase
        self.all_files: Set[str] = set()

    def build(self, files: Dict[str, str]) -> None:
        """Build the full relationship graph from a dict of filepath -> content."""
        self.all_files = set(files.keys())

        for file_path, content in files.items():
            language = LANGUAGE_MAP.get(Path(file_path).suffix.lower())
            if not language:
                continue

            imports = self._extract_imports(content, language)
            resolved = self._resolve_imports(imports, file_path, files)

            for dep in resolved:
                self.dependencies[file_path].add(dep)
                self.dependents[dep].add(file_path)

        print(f"[GRAPH] Built relationship graph for {len(files)} files")
        print(f"[GRAPH] Found {sum(len(v) for v in self.dependencies.values())} relationships")

    def get_dependencies(self, file_path: str) -> Set[str]:
        """Files that this file imports."""
        return self.dependencies.get(file_path, set())

    def get_dependents(self, file_path: str) -> Set[str]:
        """Files that import this file -- these could break if we change it."""
        return self.dependents.get(file_path, set())

    def expand_impact(self, files: List[str], depth: int = 2) -> Set[str]:
        """
        Given a list of files to modify, expand to include:
        - All files they depend on (we need context)
        - All files that depend on them (they could break)

        depth controls how many hops to follow.
        """
        expanded = set(files)
        frontier = set(files)

        for _ in range(depth):
            next_frontier = set()
            for f in frontier:
                # Add direct dependencies (what this file imports)
                deps = self.get_dependencies(f)
                new_deps = deps - expanded
                next_frontier.update(new_deps)

                # Add dependents (what imports this file)
                dependents = self.get_dependents(f)
                new_dependents = dependents - expanded
                next_frontier.update(new_dependents)

            expanded.update(next_frontier)
            frontier = next_frontier

            if not frontier:
                break

        return expanded

    def summarize(self) -> Dict:
        """Return a summary of the graph for debugging."""
        return {
            "total_files": len(self.all_files),
            "files_with_dependencies": len(self.dependencies),
            "files_with_dependents": len(self.dependents),
            "most_depended_on": sorted(
                self.dependents.items(),
                key=lambda x: len(x[1]),
                reverse=True
            )[:5]
        }

    # =========================================================
    # PRIVATE
    # =========================================================
    def _extract_imports(self, content: str, language: str) -> List[str]:
        patterns = IMPORT_PATTERNS.get(language, [])
        imports = []
        for line in content.splitlines():
            for pattern in patterns:
                matches = re.findall(pattern, line.strip())
                imports.extend(matches)
        return imports

    def _resolve_imports(
        self, imports: List[str], current_file: str, all_files: Dict[str, str]
    ) -> List[str]:
        """
        Try to match import strings to actual files in the codebase.
        Handles relative imports, module paths, and direct file references.
        """
        resolved = []
        current_dir = str(Path(current_file).parent)

        for imp in imports:
            # Convert module path to file path (e.g. "app.models" -> "app/models.py")
            as_path = imp.replace(".", "/")

            candidates = [
                f"{as_path}.py",
                f"{as_path}.js",
                f"{as_path}.ts",
                f"{as_path}.java",
                f"{current_dir}/{as_path}.py",
                f"{current_dir}/{as_path}.js",
                f"{current_dir}/{as_path}.ts",
                f"./{imp}",
                f"./{imp}.py",
                f"./{imp}.js",
            ]

            for candidate in candidates:
                # Normalize path
                normalized = str(Path(candidate))
                if normalized in all_files:
                    resolved.append(normalized)
                    break

                # Also try matching by filename only
                for known_file in all_files:
                    if Path(known_file).stem == Path(imp).stem:
                        resolved.append(known_file)
                        break

        return list(set(resolved))