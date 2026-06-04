from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import cfg_get


def get_modality(config: dict[str, Any]) -> str:
    modality = str(cfg_get(config, "data.modality", "image")).strip().lower()
    if modality not in {"image", "video", "audio"}:
        raise ValueError("data.modality must be one of: image, video, audio")
    return modality


def ensure_image_argus_stage(config: dict[str, Any], stage_name: str) -> None:
    modality = get_modality(config)
    if modality != "image":
        raise ValueError(f"{stage_name} is image-only in this release. Video/audio are limited to data construction and baselines.")


def dataset_prefix(config: dict[str, Any]) -> str:
    modality = get_modality(config)
    if modality == "image":
        return "image_vtqa2023"
    if modality == "video":
        return "video_msrvtt"
    return "audio_clotho_aqa"


def split_json_path(paths: Any, config: dict[str, Any], split: str, augmented: bool = False) -> Path:
    suffix = "_augmented" if augmented else ""
    return paths.data_dir / split / f"{dataset_prefix(config)}_{split}{suffix}.json"


def clean_media(item: dict[str, Any]) -> str:
    return str(item.get("clean_media") or item.get("clean_image") or item.get("clean_video_path") or item.get("clean_audio_path"))


def poison_media(item: dict[str, Any]) -> str:
    return str(item.get("poison_media") or item.get("poison_image") or item.get("poison_video_path") or item.get("poison_audio_path"))


def media_type(item: dict[str, Any], default: str = "image") -> str:
    return str(item.get("media_type") or item.get("modality") or default)


def video_options(config: dict[str, Any], prefix: str = "data") -> dict[str, Any]:
    options: dict[str, Any] = {}
    max_pixels = cfg_get(config, f"{prefix}.video_max_pixels", cfg_get(config, "data.video_max_pixels", 224 * 224))
    min_pixels = cfg_get(config, f"{prefix}.video_min_pixels", cfg_get(config, "data.video_min_pixels"))
    fps = cfg_get(config, f"{prefix}.video_fps", cfg_get(config, "data.video_fps", 1.0))
    nframes = cfg_get(config, f"{prefix}.video_nframes", cfg_get(config, "data.video_nframes"))
    if max_pixels is not None:
        max_pixels = int(max_pixels)
        options["max_pixels"] = max_pixels
    if min_pixels is not None:
        min_pixels = int(min_pixels)
        if max_pixels is not None:
            min_pixels = min(min_pixels, int(max_pixels))
        options["min_pixels"] = min_pixels
    if nframes is not None:
        options["nframes"] = int(nframes)
    elif fps is not None:
        options["fps"] = float(fps)
    return options


def media_block(path: str | Path, kind: str, options: dict[str, Any] | None = None) -> dict[str, Any]:
    if kind == "image":
        return {"type": "image", "image": str(path)}
    if kind == "video":
        return {"type": "video", "video": str(path), **(options or {})}
    if kind == "audio":
        return {"type": "audio", "audio_url": str(path)}
    raise ValueError(f"Unsupported media type: {kind}")
