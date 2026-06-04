from __future__ import annotations

import argparse
from pathlib import Path

from argus.config import cfg_get, get_paths, load_config
from argus.io_utils import load_json
from argus.media import clean_media, get_modality, poison_media, split_json_path
from argus.prompts import GLUE_LABELS


def _assert_media_exist(records: list[dict]) -> None:
    for item in records:
        for label, path in [("clean_media", clean_media(item)), ("poison_media", poison_media(item))]:
            if not Path(path).exists():
                raise FileNotFoundError(f"Missing {label} for sample {item['id']}: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--force", action="store_true", help="Accepted for run_all.sh compatibility.")
    args = parser.parse_args()

    config = load_config(args.config, args.output_dir)
    paths = get_paths(config)
    modality = get_modality(config)
    train = load_json(split_json_path(paths, config, "train"))
    val = load_json(split_json_path(paths, config, "val"))
    test = load_json(split_json_path(paths, config, "test"))

    max_samples = cfg_get(config, "debug.max_samples")
    # Smoke runs may be smaller than the paper split sizes.
    if max_samples is None:
        expected_train = int(cfg_get(config, "paper.train_size", 10312 if modality == "image" else len(train)))
        expected_val = int(cfg_get(config, "paper.val_size", 1000))
        expected_test = int(cfg_get(config, "paper.test_size", 1000))
        if (len(train), len(val), len(test)) != (expected_train, expected_val, expected_test):
            raise ValueError(f"Unexpected split sizes: train={len(train)}, val={len(val)}, test={len(test)}")

    # Validation trigger phrases must be unseen during training.
    train_triggers = {item["trigger_prompt"] for item in train}
    val_triggers = {item["trigger_prompt"] for item in val}
    if train_triggers & val_triggers:
        raise ValueError("Training and validation triggers overlap.")

    fixed_test_trigger = cfg_get(config, "paper.test_trigger")
    if modality == "image" and any(item["trigger_prompt"] != fixed_test_trigger for item in test):
        raise ValueError("Test split does not use the fixed Ignore trigger.")

    for item in val:
        if item["second_task"] in GLUE_LABELS and item["second_answer"] not in GLUE_LABELS[item["second_task"]]:
            raise ValueError(f"Invalid GLUE label for sample {item['id']}: {item['second_task']} -> {item['second_answer']}")

    _assert_media_exist(train)
    _assert_media_exist(val)
    _assert_media_exist(test)
    print(f"{modality} data sanity ok: train={len(train)}, val={len(val)}, test={len(test)}")


if __name__ == "__main__":
    main()
