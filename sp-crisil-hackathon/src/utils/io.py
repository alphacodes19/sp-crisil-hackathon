"""
Generic JSONL read/write helpers.

Kept source-agnostic and model-agnostic (works for UnifiedDocument today,
and later for the risk Signal model) so every pipeline stage reuses the
same persistence code instead of each inventing its own.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, List, Type, TypeVar, Union

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def read_jsonl(path: Union[str, Path], model: Type[T]) -> Iterator[T]:
    """Yield `model` instances parsed from a JSONL file, one per line."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield model.model_validate_json(line)


def write_jsonl(path: Union[str, Path], records: List[BaseModel], mode: str = "w") -> None:
    """Write a list of pydantic model instances to a JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode, encoding="utf-8") as f:
        for r in records:
            f.write(r.model_dump_json())
            f.write("\n")
