from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
ARGUS_MODALITIES = {"image"}
PUBLIC_MODALITIES = {"image", "video", "audio"}


def stage_path(name: str) -> str:
    return str(SRC_DIR / "argus" / "stages" / f"{name}.py")


def run_stage(name: str, command: list[str], env: dict[str, str]) -> None:
    print()
    print(f"==== {name} ====")
    subprocess.run(command, check=True, env=env)


def deep_merge(base: dict, override: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_runner_config(config_path: str, modality: str | None = None) -> dict:
    config_file = Path(config_path).expanduser()
    if not config_file.is_absolute():
        config_file = (Path.cwd() / config_file).resolve()
    with config_file.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if "modalities" not in raw:
        return raw

    selected = modality or os.environ.get("ARGUS_MODALITY") or raw.get("default_modality", "image")
    selected = str(selected).strip().lower()
    if selected not in raw["modalities"]:
        raise ValueError(f"Unknown modality {selected!r}; available: {sorted(raw['modalities'])}")
    common = {key: value for key, value in raw.items() if key not in {"modalities", "default_modality"}}
    config = deep_merge(common, raw["modalities"][selected])
    config.setdefault("data", {})
    config["data"]["modality"] = selected
    return config


def is_combined_public_config(config_path: str) -> bool:
    config_file = Path(config_path).expanduser()
    if not config_file.is_absolute():
        config_file = (Path.cwd() / config_file).resolve()
    with config_file.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return "modalities" in raw


def read_config_device(config_path: str, modality: str | None = None) -> str | None:
    config = load_runner_config(config_path, modality)
    device = config.get("run", {}).get("device")
    if device is None or str(device).strip() == "":
        return None
    return str(device)


def read_config_modality(config_path: str, modality: str | None = None) -> str:
    config = load_runner_config(config_path, modality)
    modality = str(config.get("data", {}).get("modality", "image")).strip().lower()
    if modality not in PUBLIC_MODALITIES:
        raise ValueError("data.modality must be one of: image, video, audio")
    return modality


def dataset_prefix_for_modality(modality: str) -> str:
    if modality == "image":
        return "image_vtqa2023"
    if modality == "video":
        return "video_msrvtt"
    return "audio_clotho_aqa"


def split_gpu_ids(raw: str | None) -> list[str]:
    if raw is None:
        return []
    return [part.strip() for part in str(raw).split(",") if part.strip()]


def resolve_output_dir(config_path: str, output_dir: str | None, modality: str | None = None) -> Path:
    config_file = Path(config_path).expanduser()
    if not config_file.is_absolute():
        config_file = (Path.cwd() / config_file).resolve()
    config = load_runner_config(str(config_file), modality)
    configured = output_dir if output_dir is not None else config.get("run", {}).get("output_dir", "outputs")
    out = Path(configured).expanduser()
    if not out.is_absolute():
        out = config_file.parents[1] / out
    return out.resolve()


def merge_augmented_shards(shard_paths: list[Path], output_path: Path) -> None:
    merged = []
    for shard_path in shard_paths:
        with shard_path.open("r", encoding="utf-8") as f:
            merged.extend(json.load(f))
    merged.sort(key=lambda item: int(item.pop("_augmentation_index", item.get("id", 0))))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print(f"merged {len(merged)} augmented samples -> {output_path}")


def summarize_score_rows(rows: list[dict]) -> dict[str, float]:
    total = max(len(rows), 1)
    return {
        "UIA": round(sum(int(x.get("first_success", False)) for x in rows) / total, 6),
        "AIA": round(sum(int(x.get("second_success", False)) for x in rows) / total, 6),
        "AIFR": round(sum(int(x.get("attacker_followed", False)) for x in rows) / total, 6),
    }


def summarize_condition(rows: list[dict], **filters: object) -> dict[str, float]:
    selected = [row for row in rows if all(row.get(key) == value for key, value in filters.items())]
    return summarize_score_rows(selected)


def merge_activation_shards(shard_paths: list[Path], output_path: Path) -> None:
    import torch

    merged = []
    for shard_path in shard_paths:
        merged.extend(torch.load(shard_path, map_location="cpu"))
    merged.sort(key=lambda item: int(item.get("id", 0)))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, output_path)
    print(f"merged {len(merged)} activation rows -> {output_path}")


def run_parallel_stage(
    name: str,
    python: str,
    stage: str,
    common: list[str],
    base_env: dict[str, str],
    gpu_ids: list[str],
    shard_paths: list[Path],
    extra_args: list[str],
) -> None:

    processes: list[tuple[str, subprocess.Popen]] = []
    for index, gpu_id in enumerate(gpu_ids):
        env = base_env.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        command = [
            python,
            stage_path(stage),
            *common,
            *extra_args,
            "--shard-index",
            str(index),
            "--num-shards",
            str(len(gpu_ids)),
            "--shard-output",
            str(shard_paths[index]),
        ]
        print(f"launch {name} shard {index + 1}/{len(gpu_ids)} on CUDA_VISIBLE_DEVICES={gpu_id}")
        processes.append((gpu_id, subprocess.Popen(command, env=env)))
    failed = []
    for gpu_id, process in processes:
        return_code = process.wait()
        if return_code != 0:
            failed.append((gpu_id, return_code))
    if failed:
        raise subprocess.CalledProcessError(failed[0][1], f"{name} failed on GPUs {failed}")


def run_parallel_augmentation(
    python: str,
    common: list[str],
    base_env: dict[str, str],
    gpu_ids: list[str],
    output_dir: Path,
    force: bool,
    modality: str,
) -> None:

    final_path = output_dir / "data" / "train" / f"{dataset_prefix_for_modality(modality)}_train_augmented.json"
    if final_path.exists() and not force:
        print(f"skip existing {final_path}")
        return

    shard_dir = output_dir / "data" / "train" / "augmentation_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_paths = [shard_dir / f"shard_{index}.json" for index in range(len(gpu_ids))]
    processes: list[tuple[str, subprocess.Popen]] = []
    for index, gpu_id in enumerate(gpu_ids):
        env = base_env.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        command = [
            python,
            stage_path("generate_augmented_answers"),
            *common,
            "--shard-index",
            str(index),
            "--num-shards",
            str(len(gpu_ids)),
            "--shard-output",
            str(shard_paths[index]),
        ]
        print(f"launch augment shard {index + 1}/{len(gpu_ids)} on CUDA_VISIBLE_DEVICES={gpu_id}")
        processes.append((gpu_id, subprocess.Popen(command, env=env)))

    failed = []
    for gpu_id, process in processes:
        return_code = process.wait()
        if return_code != 0:
            failed.append((gpu_id, return_code))
    if failed:
        raise subprocess.CalledProcessError(failed[0][1], f"parallel augmentation failed on GPUs {failed}")
    merge_augmented_shards(shard_paths, final_path)


def run_parallel_activation_collection(
    python: str,
    common: list[str],
    base_env: dict[str, str],
    gpu_ids: list[str],
    output_dir: Path,
    force: bool,
    mode: str,
    split: str,
) -> None:

    final_path = output_dir / "activations" / f"mllm_{mode}_{split}.pt"
    if final_path.exists() and not force:
        print(f"skip existing {final_path}")
        return
    shard_dir = output_dir / "activations" / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_paths = [shard_dir / f"{mode}_{split}_shard_{index}.pt" for index in range(len(gpu_ids))]
    run_parallel_stage(
        f"collect_{mode}_activations_{split}",
        python,
        "collect_activations",
        common,
        base_env,
        gpu_ids,
        shard_paths,
        ["--mode", mode, "--split", split],
    )
    merge_activation_shards(shard_paths, final_path)


def format_alpha(alpha: float) -> str:
    return str(float(alpha)).replace("-", "m").replace(".", "p")


def preliminary_condition_filename(kind: str, layers: list[int], alpha: float | None, direction: str | None) -> str:
    if kind == "no_steering":
        return "no_steering.json"
    layer_text = "_".join(str(layer) for layer in layers)
    alpha_text = format_alpha(float(alpha if alpha is not None else 0.0))
    direction_text = direction or "none"
    return f"{kind}_layers_{layer_text}_{direction_text}_alpha_{alpha_text}.json"


def load_pipeline_config(config_path: str) -> dict:
    config_file = Path(config_path).expanduser()
    if not config_file.is_absolute():
        config_file = (Path.cwd() / config_file).resolve()
    with config_file.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def default_multi_strength_grid(initial_alpha: float, top_n: int) -> list[float]:
    scales = {
        2: [0.75, 1.0, 1.25],
        3: [0.5, 0.75, 1.0],
        4: [0.375, 0.5, 0.75],
        5: [0.25, 0.375, 0.5],
    }
    return [round(initial_alpha * scale, 6) for scale in scales[top_n]]


def multi_strength_grid(config: dict, top_n: int, initial_alpha: float) -> list[float]:
    configured = config.get("preliminary", {}).get("multi_strength_grid", {}).get(str(top_n))
    if configured is None:
        configured = config.get("preliminary", {}).get("multi_strength_grid", {}).get(top_n)
    if configured is not None:
        return [float(value) for value in configured]
    return default_multi_strength_grid(initial_alpha, top_n)


def merge_preliminary_condition_shards(shard_paths: list[Path], output_path: Path) -> None:
    condition = None
    results = []
    for path in shard_paths:
        with path.open("r", encoding="utf-8") as f:
            shard = json.load(f)
        condition = shard.get("condition", condition)
        results.extend(shard.get("results", []))
    results.sort(key=lambda item: int(item.get("id", 0)))
    metrics = summarize_score_rows(results)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump({"condition": condition or {}, "metrics": metrics, "results": results}, f, ensure_ascii=False, indent=2)
    print(f"merged preliminary condition -> {output_path}")


def json_path_is_valid(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8") as f:
            json.load(f)
        return True
    except Exception:
        return False


def run_parallel_preliminary_condition(
    python: str,
    common: list[str],
    base_env: dict[str, str],
    gpu_ids: list[str],
    output_dir: Path,
    force: bool,
    kind: str,
    layers: list[int],
    alpha: float | None,
    direction: str | None,
) -> Path:

    filename = preliminary_condition_filename(kind, layers, alpha, direction)
    final_path = output_dir / "results" / "preliminary_conditions" / filename
    if final_path.exists() and not force:
        if json_path_is_valid(final_path):
            print(f"skip existing {final_path}")
            return final_path
        print(f"rerun incomplete or invalid preliminary condition: {final_path}")

    shard_dir = output_dir / "results" / "preliminary_conditions" / "shards" / final_path.stem
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_paths = [shard_dir / f"shard_{index}.json" for index in range(len(gpu_ids))]
    extra_args = ["--condition-kind", kind]
    if layers:
        extra_args.extend(["--layers", ",".join(str(layer) for layer in layers)])
    if alpha is not None:
        extra_args.extend(["--alpha", str(alpha)])
    if direction is not None:
        extra_args.extend(["--direction", direction])
    run_parallel_stage(
        f"preliminary_{final_path.stem}",
        python,
        "preliminary_experiments",
        common,
        base_env,
        gpu_ids,
        shard_paths,
        extra_args,
    )
    merge_preliminary_condition_shards(shard_paths, final_path)
    return final_path


def fixed_selection_from_condition_files(output_dir: Path, config: dict) -> dict:
    preliminary_cfg = config.get("preliminary", {})
    layer_start = int(preliminary_cfg.get("layer_start", 8))
    layer_end = int(preliminary_cfg.get("layer_end", 18))
    initial_alpha = float(preliminary_cfg.get("initial_strength", 10))
    condition_dir = output_dir / "results" / "preliminary_conditions"
    gaps = []
    for layer in range(layer_start, layer_end + 1):
        attack_path = condition_dir / preliminary_condition_filename("single_fixed", [layer], initial_alpha, "attack")
        defense_path = condition_dir / preliminary_condition_filename("single_fixed", [layer], initial_alpha, "defense")
        with attack_path.open("r", encoding="utf-8") as f:
            attack = json.load(f)["metrics"]
        with defense_path.open("r", encoding="utf-8") as f:
            defense = json.load(f)["metrics"]
        gaps.append({"layer": layer, "aia_gap": round(attack["AIA"] - defense["AIA"], 6)})
    ranked_layers = [row["layer"] for row in sorted(gaps, key=lambda row: (-row["aia_gap"], row["layer"]))]
    return {"aia_gaps": gaps, "ranked_layers": ranked_layers, "best_single_layer": ranked_layers[0]}


def run_parallel_preliminary(
    python: str,
    common: list[str],
    base_env: dict[str, str],
    gpu_ids: list[str],
    output_dir: Path,
    config_path: str,
    force: bool,
) -> None:

    final_path = output_dir / "results" / "preliminary_experiments.json"
    prediction_path = output_dir / "results" / "preliminary_experiments_predictions.json"
    if final_path.exists() and prediction_path.exists() and not force:
        with final_path.open("r", encoding="utf-8") as f:
            existing = json.load(f)
        if "multi_layer_results" in existing:
            print(f"skip existing {final_path}")
            return

    config = load_pipeline_config(config_path)
    preliminary_cfg = config.get("preliminary", {})
    layer_start = int(preliminary_cfg.get("layer_start", 8))
    layer_end = int(preliminary_cfg.get("layer_end", 18))
    initial_alpha = float(preliminary_cfg.get("initial_strength", 10))
    strength_grid = [float(x) for x in preliminary_cfg.get("strength_grid", [5, 10, 15, 20, 25, 30, 35, 40])]
    topn_values = [int(x) for x in preliminary_cfg.get("multi_topn_values", [2, 3, 4, 5])]

    run_parallel_preliminary_condition(python, common, base_env, gpu_ids, output_dir, force, "no_steering", [], None, None)
    for layer in range(layer_start, layer_end + 1):
        for direction in ["attack", "defense"]:
            run_parallel_preliminary_condition(
                python,
                common,
                base_env,
                gpu_ids,
                output_dir,
                force,
                "single_fixed",
                [layer],
                initial_alpha,
                direction,
            )

    fixed_selection = fixed_selection_from_condition_files(output_dir, config)
    best_single_layer = int(fixed_selection["best_single_layer"])
    print(f"preliminary fixed sweep selected best_single_layer={best_single_layer}")
    for alpha in strength_grid:
        run_parallel_preliminary_condition(
            python,
            common,
            base_env,
            gpu_ids,
            output_dir,
            force,
            "single_sensitivity",
            [best_single_layer],
            alpha,
            "defense",
        )

    ranked_layers = [int(layer) for layer in fixed_selection["ranked_layers"]]
    for top_n in topn_values:
        layers = ranked_layers[:top_n]
        for alpha in multi_strength_grid(config, top_n, initial_alpha):
            run_parallel_preliminary_condition(
                python,
                common,
                base_env,
                gpu_ids,
                output_dir,
                force,
                "multi_topn",
                layers,
                alpha,
                "defense",
            )

    run_stage("preliminary_summary", [python, stage_path("preliminary_experiments"), *common, "--summary-only"], base_env)


def merge_eval_shards(shard_paths: list[Path], output_path: Path, mode: str) -> None:
    results = []
    clean_score_rows = []
    inject_score_rows = []
    post_filtering_enabled = False
    for path in shard_paths:
        with path.open("r", encoding="utf-8") as f:
            shard = json.load(f)
        post_filtering_enabled = post_filtering_enabled or bool(shard.get("post_filtering_enabled", False))
        results.extend(shard.get("results", []))
    results.sort(key=lambda item: int(item.get("id", 0)))
    for item in results:
        clean_score_rows.append(
            {
                "first_success": item.get("clean_first_success", False),
                "second_success": False,
                "attacker_followed": False,
            }
        )
        inject_score_rows.append(
            {
                "first_success": item.get("first_success", False),
                "second_success": item.get("second_success", False),
                "attacker_followed": item.get("attacker_followed", False),
            }
        )
    clean_metrics = summarize_score_rows(clean_score_rows)
    inject_metrics = summarize_score_rows(inject_score_rows)
    summary = {
        "mode": mode,
        "defense_enabled": mode == "defense",
        "post_filtering_enabled": post_filtering_enabled,
        "UIA_clean": clean_metrics["UIA"],
        "UIA_inject": inject_metrics["UIA"],
        "AIA": inject_metrics["AIA"],
        "AIFR": inject_metrics["AIFR"],
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"merged eval results -> {output_path}")


def run_parallel_eval(
    python: str,
    common: list[str],
    base_env: dict[str, str],
    gpu_ids: list[str],
    output_dir: Path,
    force: bool,
    mode: str,
    modality: str,
) -> None:

    output_name = f"{modality}_no_defense.json" if mode == "no_defense" else f"{modality}_argus.json"
    final_path = output_dir / "results" / output_name
    if final_path.exists() and not force:
        print(f"skip existing {final_path}")
        return
    shard_dir = output_dir / "results" / "eval_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_paths = [shard_dir / f"{mode}_{index}.json" for index in range(len(gpu_ids))]
    run_parallel_stage(
        f"run_{mode}_eval",
        python,
        "run_defense_eval",
        common,
        base_env,
        gpu_ids,
        shard_paths,
        ["--mode", mode],
    )
    merge_eval_shards(shard_paths, final_path, mode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ARGUS image pipeline or public dataset build.")
    parser.add_argument("--config", default=str(ROOT_DIR / "configs" / "argus.yaml"))
    parser.add_argument("--output-dir")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-missing", action="store_true", help="Allow dry-run config checks to pass with placeholder paths.")
    parser.add_argument("--gpu-id", help="Override run.device and set CUDA_VISIBLE_DEVICES for all stages, e.g. 0 or 0,1,2.")
    parser.add_argument("--modality", choices=sorted(PUBLIC_MODALITIES), help="Select a modality block from configs/dataset_baselines.yaml.")
    parser.add_argument("--only-data", action="store_true", help="Run only dataset construction and validation.")
    parser.add_argument("--skip-data", action="store_true", help="Skip dataset construction and validation.")
    args = parser.parse_args()
    if args.only_data and args.skip_data:
        raise ValueError("--only-data and --skip-data cannot be used together")

    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(SRC_DIR) if not existing else str(SRC_DIR) + os.pathsep + existing
    if args.modality:
        env["ARGUS_MODALITY"] = args.modality
    gpu_id = args.gpu_id if args.gpu_id is not None else read_config_device(args.config, args.modality)
    modality = read_config_modality(args.config, args.modality)
    if is_combined_public_config(args.config) and not args.only_data and not args.dry_run:
        raise ValueError("configs/dataset_baselines.yaml is for dataset construction and baselines. Use configs/argus.yaml for the ARGUS pipeline.")
    if modality not in ARGUS_MODALITIES and not args.only_data and not args.dry_run:
        raise ValueError("The ARGUS activation/probe/vector/defense pipeline is image-only in this release. Use scripts/build_dataset.py and scripts/run_baselines.py for video/audio.")
    if gpu_id is not None:
        # Child stages must import GPU libraries after this restriction is set.
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        print(f"CUDA_VISIBLE_DEVICES={gpu_id}")
    gpu_ids = split_gpu_ids(gpu_id)

    common = ["--config", args.config]
    if args.output_dir:
        common.extend(["--output-dir", args.output_dir])
    if args.force:
        common.append("--force")

    python = sys.executable
    if args.dry_run:
        command = [python, stage_path("validate_config"), "--config", args.config]
        if args.output_dir:
            command.extend(["--output-dir", args.output_dir])
        if args.allow_missing:
            command.append("--allow-missing")
        subprocess.run(command, check=True, env=env)
        return

    output_root = resolve_output_dir(args.config, args.output_dir, args.modality)
    stages = [
        ("create_data", [python, stage_path("create_data"), *common]),
        ("validate_data", [python, stage_path("validate_data"), *common]),
        ("augment_answers", None),
        ("collect_detection_activations_train", None),
        ("collect_detection_activations_val", None),
        ("collect_edit_activations_train", None),
        ("collect_edit_activations_val", None),
        ("train_detection_probe", [python, stage_path("train_probe"), *common, "--kind", "detection"]),
        ("train_edit_probe", [python, stage_path("train_probe"), *common, "--kind", "edit"]),
        ("Preliminary experiments", None),
        ("train_auxiliary_probes", [python, stage_path("train_auxiliary_probes"), *common]),
        ("train_vector", [python, stage_path("train_vector"), *common]),
        ("train_final_steering_probe", [python, stage_path("train_post_filter_probe"), *common]),
        ("estimate_epsilon", [python, stage_path("estimate_epsilon"), *common]),
        ("run_no_defense_eval", None),
        ("run_defense_eval", None),
    ]
    if args.only_data:
        stages = stages[:2]
    elif args.skip_data:
        stages = stages[2:]

    for name, command in stages:
        if name == "augment_answers" and len(gpu_ids) > 1:
            print()
            print("==== augment_answers ====")
            run_parallel_augmentation(python, common, env, gpu_ids, output_root, args.force, modality)
        elif name == "augment_answers":
            run_stage(name, [python, stage_path("generate_augmented_answers"), *common], env)
        elif name.startswith("collect_") and len(gpu_ids) > 1:
            _, mode, _, split = name.split("_", 3)
            print()
            print(f"==== {name} ====")
            run_parallel_activation_collection(python, common, env, gpu_ids, output_root, args.force, mode, split)
        elif name.startswith("collect_"):
            _, mode, _, split = name.split("_", 3)
            run_stage(name, [python, stage_path("collect_activations"), *common, "--mode", mode, "--split", split], env)
        elif name == "Preliminary experiments" and len(gpu_ids) > 1:
            print()
            print("==== Preliminary experiments ====")
            run_parallel_preliminary(python, common, env, gpu_ids, output_root, args.config, args.force)
        elif name == "Preliminary experiments":
            run_stage(name, [python, stage_path("preliminary_experiments"), *common], env)
        elif name == "run_no_defense_eval" and len(gpu_ids) > 1:
            print()
            print("==== run_no_defense_eval ====")
            run_parallel_eval(python, common, env, gpu_ids, output_root, args.force, "no_defense", modality)
        elif name == "run_defense_eval" and len(gpu_ids) > 1:
            print()
            print("==== run_defense_eval ====")
            run_parallel_eval(python, common, env, gpu_ids, output_root, args.force, "defense", modality)
        elif name == "run_no_defense_eval":
            run_stage(name, [python, stage_path("run_defense_eval"), *common, "--mode", "no_defense"], env)
        elif name == "run_defense_eval":
            run_stage(name, [python, stage_path("run_defense_eval"), *common, "--mode", "defense"], env)
        else:
            run_stage(name, command, env)


if __name__ == "__main__":
    main()
