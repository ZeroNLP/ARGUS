from __future__ import annotations

import argparse
from pathlib import Path

from argus.config import cfg_get, get_paths, load_config, resolve_input_path
from argus.media import get_modality


def looks_like_hf_repo(value: object) -> bool:
    text = str(value or "")
    return "/" in text and not text.startswith(("/", ".", "~")) and not Path(text).exists()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config, args.output_dir)
    paths = get_paths(config)
    modality = get_modality(config)
    required = ["model.path", "data.alpaca_path", "data.glue_dir", "data.trigger_path"]
    if modality == "image":
        required.extend(["data.vtqa_path", "data.font_path"])
    elif modality == "video":
        required.extend(["data.msrvtt_path", "data.font_path"])
    else:
        required.extend(["data.clotho_train_csv", "data.clotho_val_csv", "data.clotho_test_csv", "data.audio_files_dir"])

    missing = []
    for key in required:
        if key == "data.vtqa_path" and looks_like_hf_repo(cfg_get(config, key)):
            print(f"{key}: {cfg_get(config, key)} [{cfg_get(config, 'data.vtqa_config', 'en-image')}]")
            continue
        path = resolve_input_path(config, key)
        status = "ok" if path.exists() else "missing"
        print(f"{key}: {path} [{status}]")
        if not path.exists():
            missing.append(str(path))

    if missing and not args.allow_missing:
        raise SystemExit("Missing required inputs. Fill configs/dataset_baselines.yaml or configs/argus.yaml, or use --allow-missing for structure checks.")

    print(f"modality: {modality}")
    print(f"output_dir: {paths.output_dir}")
    print(f"debug.max_samples: {cfg_get(config, 'debug.max_samples')}")
    if modality != "image":
        print("stage order: create_data -> validate_data. Use scripts/run_baselines.py for baselines.")
    else:
        print("stage order: create_data -> validate_data -> augment_answers -> collect_detection_activations -> collect_edit_activations -> train_detection_probe -> train_edit_probe -> Preliminary experiments -> train_auxiliary_probes -> train_vector -> train_final_steering_probe -> estimate_epsilon -> run_no_defense_eval -> run_defense_eval")


if __name__ == "__main__":
    main()
