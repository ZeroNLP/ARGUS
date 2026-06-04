from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_lines(path: str | Path) -> list[str]:
    with Path(path).open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def should_skip(outputs: Iterable[str | Path], force: bool) -> bool:
    if force:
        return False
    return all(Path(p).exists() for p in outputs)


def limited(items: list[Any], max_samples: int | None) -> list[Any]:
    if max_samples is None:
        return items
    return items[: int(max_samples)]
