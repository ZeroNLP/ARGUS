from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from .io_utils import load_json
from .media import clean_media, poison_media


ANSWER_VARIANT_KEYS = ["short_response", "medium_response", "long_response"]


def choose_augmented_answer(item: dict[str, Any], field: str) -> str:
    augmented_key = f"{field}_augmented"
    augmented = item.get(augmented_key)
    if isinstance(augmented, dict):
        key = random.choice(ANSWER_VARIANT_KEYS)
        if augmented.get(key):
            return str(augmented[key])
    return str(item[field])


def augmented_answer_variants(item: dict[str, Any], field: str) -> list[tuple[str, str]]:
    augmented = item.get(f"{field}_augmented")
    if not isinstance(augmented, dict):
        return [("original", str(item[field]))]
    variants = [(key, str(augmented[key])) for key in ANSWER_VARIANT_KEYS if augmented.get(key)]
    return variants or [("original", str(item[field]))]


def paired_augmented_answer_variants(item: dict[str, Any]) -> list[tuple[str, str, str]]:
    first = dict(augmented_answer_variants(item, "first_answer"))
    second = dict(augmented_answer_variants(item, "second_answer"))
    pairs = []
    for key in ANSWER_VARIANT_KEYS:
        if key in first and key in second:
            pairs.append((key, first[key], second[key]))
    if pairs:
        return pairs
    return [("original", str(item["first_answer"]), str(item["second_answer"]))]


def shard_items(data: list[dict[str, Any]], shard_index: int = 0, num_shards: int = 1) -> list[dict[str, Any]]:
    if num_shards <= 1:
        return data
    return [item for index, item in enumerate(data) if index % num_shards == shard_index]


def collect_detection_activations(
    runner: Any,
    data_json: Path,
    output_path: Path,
    shard_index: int = 0,
    num_shards: int = 1,
    default_media_type: str = "image",
) -> Path:

    data = shard_items(load_json(data_json), shard_index, num_shards)
    results = []
    for item in tqdm(data, desc=f"detection activations: {data_json.name}"):
        clean = runner.activation(item["clean_prompt"], clean_media(item), media_type="image")
        injected = runner.activation(item["clean_prompt"], poison_media(item), media_type="image")
        results.append({"id": item["id"], "clean_activation": clean, "image_inject_activation": injected})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(results, output_path)
    return output_path


def collect_edit_activations(
    runner: Any,
    data_json: Path,
    output_path: Path,
    use_augmented: bool = True,
    shard_index: int = 0,
    num_shards: int = 1,
    default_media_type: str = "image",
) -> Path:

    data = shard_items(load_json(data_json), shard_index, num_shards)
    results = []
    for item in tqdm(data, desc=f"edit activations: {data_json.name}"):
        answer_rows = paired_augmented_answer_variants(item) if use_augmented else [("original", str(item["first_answer"]), str(item["second_answer"]))]
        for variant, first_answer, second_answer in answer_rows:
            first = runner.activation(item["clean_prompt"], poison_media(item), prefix=first_answer, media_type="image")
            second = runner.activation(item["clean_prompt"], poison_media(item), prefix=second_answer, media_type="image")
            results.append(
                {
                    "id": item["id"],
                    "answer_variant": variant,
                    "first_activation": first,
                    "second_activation": second,
                }
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(results, output_path)
    return output_path
