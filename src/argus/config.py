from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ProjectPaths:
    project_dir: Path
    output_dir: Path
    data_dir: Path
    activation_dir: Path
    probe_dir: Path
    vector_dir: Path
    epsilon_dir: Path
    result_dir: Path


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def select_modality_config(raw: dict[str, Any], modality: str | None = None) -> dict[str, Any]:
    if "modalities" not in raw:
        return copy.deepcopy(raw)

    selected = modality or os.environ.get("ARGUS_MODALITY") or raw.get("default_modality", "image")
    selected = str(selected).strip().lower()
    modalities = raw.get("modalities", {})
    if selected not in modalities:
        raise ValueError(f"Unknown modality {selected!r}; available: {sorted(modalities)}")

    common = {key: value for key, value in raw.items() if key not in {"modalities", "default_modality"}}
    config = deep_merge(common, modalities[selected])
    config.setdefault("data", {})
    config["data"]["modality"] = selected
    return config


def load_config(config_path: str | Path, output_dir: str | Path | None = None) -> dict[str, Any]:
    path = Path(config_path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    config = select_modality_config(raw)

    project_dir = path.parents[1]
    configured_output = output_dir if output_dir is not None else config.get("run", {}).get("output_dir", "outputs")
    out = Path(configured_output).expanduser()
    if not out.is_absolute():
        out = project_dir / out

    config["_config_path"] = str(path)
    config["_project_dir"] = str(project_dir)
    config["_output_dir"] = str(out.resolve())
    return config


def get_paths(config: dict[str, Any]) -> ProjectPaths:
    project_dir = Path(config["_project_dir"]).resolve()
    output_dir = Path(config["_output_dir"]).resolve()
    return ProjectPaths(
        project_dir=project_dir,
        output_dir=output_dir,
        data_dir=output_dir / "data",
        activation_dir=output_dir / "activations",
        probe_dir=output_dir / "probes",
        vector_dir=output_dir / "vectors",
        epsilon_dir=output_dir / "epsilon",
        result_dir=output_dir / "results",
    )


def cfg_get(config: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    value: Any = config
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def resolve_input_path(config: dict[str, Any], dotted_key: str) -> Path:
    raw = cfg_get(config, dotted_key)
    if raw is None:
        raise KeyError(f"Missing required config value: {dotted_key}")
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = Path(config["_project_dir"]) / path
    return path.resolve()
