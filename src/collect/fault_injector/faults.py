"""Extensible fault injection library for frontend build failures."""

from __future__ import annotations

import json
import random
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, Field

from src.schema import ErrorCategory

IGNORED_DIRS = {"node_modules", ".git", "dist", "build", ".next", ".svelte-kit", "out"}
IMPORT_PATTERN = re.compile(r"""(from\s+['"])(\.\.?/[^'"]+)(['"])""")


class FaultNotApplicableError(RuntimeError):
    """Raised when a fault cannot be applied to a given repository."""


class FaultApplication(BaseModel):
    """Metadata produced when a fault is applied to a repository."""

    fault_id: str
    error_category: ErrorCategory
    root_cause: str
    fix_steps: list[str] = Field(min_length=1)
    files_to_change: list[str] = Field(default_factory=list)
    fault_params: dict[str, Any] = Field(default_factory=dict)
    expected_fix_text: str


class BaseFault(ABC):
    """Abstract base class for deterministic frontend build faults."""

    fault_id: ClassVar[str]
    error_category: ClassVar[ErrorCategory]

    @abstractmethod
    def apply(self, repo_dir: Path, rng: random.Random) -> FaultApplication:
        """Mutate `repo_dir` in-place to inject the fault and return ground-truth metadata."""

    @abstractmethod
    def verify_fix(self, repo_dir: Path, fault_params: dict[str, Any]) -> bool:
        """Return True if the injected fault has been resolved in `repo_dir`."""


FAULT_REGISTRY: dict[str, type[BaseFault]] = {}


def register_fault(cls: type[BaseFault]) -> type[BaseFault]:
    """Register a BaseFault subclass by its `fault_id`."""
    FAULT_REGISTRY[cls.fault_id] = cls
    return cls


def get_fault(fault_id: str) -> BaseFault:
    """Instantiate a registered fault by ID."""
    if fault_id not in FAULT_REGISTRY:
        available = ", ".join(sorted(FAULT_REGISTRY))
        raise KeyError(f"Unknown fault '{fault_id}'. Available: {available}")
    return FAULT_REGISTRY[fault_id]()


def _iter_source_files(repo_dir: Path, extensions: tuple[str, ...]) -> list[Path]:
    """Return sorted source files in `repo_dir` matching `extensions`, skipping ignored dirs."""
    matches: list[Path] = []
    for path in sorted(repo_dir.rglob("*")):
        if any(part in IGNORED_DIRS for part in path.parts):
            continue
        if path.is_file() and path.suffix in extensions and not path.name.endswith(".d.ts"):
            matches.append(path)
    return matches


@register_fault
class RemoveDependencyFault(BaseFault):
    """Remove a declared dependency from package.json to trigger dependency_missing."""

    fault_id: ClassVar[str] = "remove_dependency"
    error_category: ClassVar[ErrorCategory] = ErrorCategory.DEPENDENCY_MISSING

    def apply(self, repo_dir: Path, rng: random.Random) -> FaultApplication:
        pkg_path = repo_dir / "package.json"
        if not pkg_path.exists():
            raise FaultNotApplicableError(f"No package.json found in {repo_dir}")

        pkg_data: dict[str, Any] = json.loads(pkg_path.read_text(encoding="utf-8"))
        section = "dependencies" if pkg_data.get("dependencies") else "devDependencies"
        deps: dict[str, str] = pkg_data.get(section, {})
        if not deps:
            raise FaultNotApplicableError("package.json has no dependencies or devDependencies.")

        pkg_name = rng.choice(sorted(deps.keys()))
        original_version = deps.pop(pkg_name)
        pkg_path.write_text(json.dumps(pkg_data, indent=2) + "\n", encoding="utf-8")

        fix_text = f"Restore '{pkg_name}': '{original_version}' in package.json ({section}) and reinstall."
        return FaultApplication(
            fault_id=self.fault_id,
            error_category=self.error_category,
            root_cause=f"Package '{pkg_name}' was removed from {section} in package.json.",
            fix_steps=[
                f"Add '{pkg_name}' ({original_version}) back to {section} in package.json.",
                "Run the project's package manager install command to sync node_modules.",
            ],
            files_to_change=["package.json"],
            fault_params={
                "section": section,
                "removed_package": pkg_name,
                "original_version": original_version,
            },
            expected_fix_text=fix_text,
        )

    def verify_fix(self, repo_dir: Path, fault_params: dict[str, Any]) -> bool:
        pkg_path = repo_dir / "package.json"
        if not pkg_path.exists():
            return False
        pkg_data: dict[str, Any] = json.loads(pkg_path.read_text(encoding="utf-8"))
        section = fault_params["section"]
        removed_pkg = fault_params["removed_package"]
        return removed_pkg in pkg_data.get(section, {})


@register_fault
class WrongImportPathFault(BaseFault):
    """Break a relative import path in a source file to trigger import_path_error."""

    fault_id: ClassVar[str] = "wrong_import_path"
    error_category: ClassVar[ErrorCategory] = ErrorCategory.IMPORT_PATH_ERROR
    BROKEN_TARGET: ClassVar[str] = "./__launchforge_missing_module__"

    def apply(self, repo_dir: Path, rng: random.Random) -> FaultApplication:
        candidates: list[tuple[Path, str]] = []
        for src_file in _iter_source_files(repo_dir, (".ts", ".tsx", ".js", ".jsx")):
            text = src_file.read_text(encoding="utf-8")
            match = IMPORT_PATTERN.search(text)
            if match:
                candidates.append((src_file, match.group(2)))

        if not candidates:
            raise FaultNotApplicableError("No source files with relative ES imports found.")

        target_file, original_import = rng.choice(candidates)
        rel_path = target_file.relative_to(repo_dir).as_posix()
        content = target_file.read_text(encoding="utf-8")
        mutated = IMPORT_PATTERN.sub(rf"\g<1>{self.BROKEN_TARGET}\g<3>", content, count=1)
        target_file.write_text(mutated, encoding="utf-8")

        fix_text = f"Restore import '{self.BROKEN_TARGET}' back to '{original_import}' in {rel_path}."
        return FaultApplication(
            fault_id=self.fault_id,
            error_category=self.error_category,
            root_cause=f"Import path '{self.BROKEN_TARGET}' in {rel_path} does not resolve to any file.",
            fix_steps=[
                f"Open {rel_path} and replace '{self.BROKEN_TARGET}' with '{original_import}'.",
            ],
            files_to_change=[rel_path],
            fault_params={
                "file_path": rel_path,
                "original_import": original_import,
                "broken_import": self.BROKEN_TARGET,
            },
            expected_fix_text=fix_text,
        )

    def verify_fix(self, repo_dir: Path, fault_params: dict[str, Any]) -> bool:
        target_file = repo_dir / fault_params["file_path"]
        if not target_file.exists():
            return False
        text = target_file.read_text(encoding="utf-8")
        return (
            fault_params["broken_import"] not in text
            and fault_params["original_import"] in text
        )


@register_fault
class TypeScriptTypeErrorFault(BaseFault):
    """Inject a deterministic TypeScript type error to trigger type_error."""

    fault_id: ClassVar[str] = "ts_type_error"
    error_category: ClassVar[ErrorCategory] = ErrorCategory.TYPE_ERROR
    MARKER: ClassVar[str] = "__launchforge_type_error__"
    SNIPPET: ClassVar[str] = (
        "\n// Injected type error for build failure benchmark\n"
        'export const __launchforge_type_error__: number = "launchforge_invalid_number";\n'
    )

    def apply(self, repo_dir: Path, rng: random.Random) -> FaultApplication:
        ts_files = _iter_source_files(repo_dir, (".ts", ".tsx"))
        if not ts_files:
            raise FaultNotApplicableError("No TypeScript (.ts/.tsx) source files found.")

        target_file = rng.choice(ts_files)
        rel_path = target_file.relative_to(repo_dir).as_posix()
        original = target_file.read_text(encoding="utf-8")
        target_file.write_text(original + self.SNIPPET, encoding="utf-8")

        fix_text = f"Remove or fix the invalid string-to-number assignment for {self.MARKER} in {rel_path}."
        return FaultApplication(
            fault_id=self.fault_id,
            error_category=self.error_category,
            root_cause=f"Type 'string' is not assignable to type 'number' for {self.MARKER} in {rel_path}.",
            fix_steps=[
                f"Open {rel_path} and fix or remove the `{self.MARKER}` type mismatch.",
            ],
            files_to_change=[rel_path],
            fault_params={
                "file_path": rel_path,
                "marker": self.MARKER,
            },
            expected_fix_text=fix_text,
        )

    def verify_fix(self, repo_dir: Path, fault_params: dict[str, Any]) -> bool:
        target_file = repo_dir / fault_params["file_path"]
        if not target_file.exists():
            return False
        return fault_params["marker"] not in target_file.read_text(encoding="utf-8")
