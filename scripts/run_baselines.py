from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from run_all import load_runner_config, read_config_device, read_config_modality, resolve_output_dir, split_gpu_ids, stage_path


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
BASELINES = ["none", "system_prompt", "ignore", "noise", "remove"]


def parse_baselines(raw: str) -> list[str]:
    if raw.strip().lower() == "all":
        return BASELINES
    names = [part.strip() for part in raw.split(",") if part.strip()]
    invalid = [name for name in names if name not in BASELINES]
    if invalid:
        raise ValueError(f"Unknown baselines {invalid}; valid choices are {BASELINES}")
    return names


def load_yaml_config(config_path: str, modality: str) -> dict:
    return load_runner_config(config_path, modality)


def configured_baselines(config: dict, raw: str) -> list[str]:
    if raw.strip().lower() != "all":
        return parse_baselines(raw)
    enabled = config.get("baselines", {}).get("enabled")
    if enabled:
        return parse_baselines(",".join(str(item) for item in enabled))
    return BASELINES


def baseline_available(config: dict, baseline: str) -> bool:
    if baseline != "remove":
        return True
    remove = config.get("baselines", {}).get("remove", {})
    return bool(remove.get("clean_dir") and remove.get("poison_dir"))


def run_stage(name: str, command: list[str], env: dict[str, str]) -> None:
    print()
    print(f"==== {name} ====")
    subprocess.run(command, check=True, env=env)


def summarize_score_rows(rows: list[dict]) -> dict[str, float]:
    total = max(len(rows), 1)
    return {
        "UIA": round(sum(int(row.get("first_success", False)) for row in rows) / total, 6),
        "AIA": round(sum(int(row.get("second_success", False)) for row in rows) / total, 6),
        "AIFR": round(sum(int(row.get("attacker_followed", False)) for row in rows) / total, 6),
    }


def merge_baseline_shards(baseline: str, shard_paths: list[Path], output_path: Path) -> None:
    results = []
    for shard_path in shard_paths:
        with shard_path.open("r", encoding="utf-8") as f:
            shard = json.load(f)
        results.extend(shard.get("results", []))
    results.sort(key=lambda item: int(item.get("id", 0)))
    clean_rows = [{"first_success": row.get("clean_first_success", False)} for row in results]
    inject_rows = [
        {
            "first_success": row.get("first_success", False),
            "second_success": row.get("second_success", False),
            "attacker_followed": row.get("attacker_followed", False),
        }
        for row in results
    ]
    clean_metrics = summarize_score_rows(clean_rows)
    inject_metrics = summarize_score_rows(inject_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "baseline": baseline,
                "UIA_clean": clean_metrics["UIA"],
                "UIA_inject": inject_metrics["UIA"],
                "AIA": inject_metrics["AIA"],
                "AIFR": inject_metrics["AIFR"],
                "results": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"merged baseline results -> {output_path}")


def run_parallel_baseline(
    python: str,
    common: list[str],
    env: dict[str, str],
    gpu_ids: list[str],
    output_root: Path,
    modality: str,
    baseline: str,
    force: bool,
) -> None:

    final_path = output_root / "results" / "baselines" / f"{modality}_{baseline}.json"
    if final_path.exists() and not force:
        print(f"skip existing {final_path}")
        return
    shard_dir = output_root / "results" / "baseline_shards" / baseline
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_paths = [shard_dir / f"shard_{index}.json" for index in range(len(gpu_ids))]
    processes: list[tuple[str, subprocess.Popen]] = []
    for index, gpu_id in enumerate(gpu_ids):
        shard_env = env.copy()
        shard_env["CUDA_VISIBLE_DEVICES"] = gpu_id
        command = [
            python,
            stage_path("run_baseline_eval"),
            *common,
            "--baseline",
            baseline,
            "--shard-index",
            str(index),
            "--num-shards",
            str(len(gpu_ids)),
            "--shard-output",
            str(shard_paths[index]),
        ]
        print(f"launch {baseline} baseline shard {index + 1}/{len(gpu_ids)} on CUDA_VISIBLE_DEVICES={gpu_id}")
        processes.append((gpu_id, subprocess.Popen(command, env=shard_env)))
    failed = []
    for gpu_id, process in processes:
        return_code = process.wait()
        if return_code != 0:
            failed.append((gpu_id, return_code))
    if failed:
        raise subprocess.CalledProcessError(failed[0][1], f"{baseline} baseline failed on GPUs {failed}")
    merge_baseline_shards(baseline, shard_paths, final_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all configured ARGUS baseline defenses.")
    parser.add_argument("--config", default=str(ROOT_DIR / "configs" / "dataset_baselines.yaml"))
    parser.add_argument("--output-dir")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--gpu-id", help="Override run.device and set CUDA_VISIBLE_DEVICES, e.g. 0 or 0,1,2.")
    parser.add_argument("--modality", default="image", choices=["image", "video", "audio"])
    parser.add_argument("--baselines", default="all", help="Comma-separated subset or 'all'.")
    args = parser.parse_args()

    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(SRC_DIR) if not existing else str(SRC_DIR) + os.pathsep + existing
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env["ARGUS_MODALITY"] = args.modality
    gpu_id = args.gpu_id if args.gpu_id is not None else read_config_device(args.config, args.modality)
    if gpu_id:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        print(f"CUDA_VISIBLE_DEVICES={gpu_id}")
    gpu_ids = split_gpu_ids(gpu_id)
    modality = read_config_modality(args.config, args.modality)
    config = load_yaml_config(args.config, args.modality)
    output_root = resolve_output_dir(args.config, args.output_dir, args.modality)
    common = ["--config", args.config]
    if args.output_dir:
        common.extend(["--output-dir", args.output_dir])
    if args.force:
        common.append("--force")
    python = sys.executable

    for baseline in configured_baselines(config, args.baselines):
        if not baseline_available(config, baseline):
            print(f"skip baseline_{baseline}: configure baselines.remove.clean_dir and baselines.remove.poison_dir first")
            continue
        if len(gpu_ids) > 1:
            print()
            print(f"==== baseline_{baseline} ====")
            run_parallel_baseline(python, common, env, gpu_ids, output_root, modality, baseline, args.force)
        else:
            run_stage(
                f"baseline_{baseline}",
                [python, stage_path("run_baseline_eval"), *common, "--baseline", baseline],
                env,
            )


if __name__ == "__main__":
    main()
