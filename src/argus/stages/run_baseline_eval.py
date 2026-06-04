from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from argus.baselines import BASELINE_NAMES, build_baseline_conversation
from argus.config import cfg_get, get_paths, load_config, resolve_input_path
from argus.eval_metrics import score_response, summarize_scores
from argus.io_utils import load_json, save_json, seed_everything
from argus.media import get_modality, split_json_path, video_options
from argus.modeling import AudioLanguageRunner, VisionLanguageRunner


def selected_baselines(raw: str | None) -> list[str]:
    if raw is None or raw.strip().lower() == "all":
        return BASELINE_NAMES
    names = [part.strip() for part in raw.split(",") if part.strip()]
    invalid = [name for name in names if name not in BASELINE_NAMES]
    if invalid:
        raise ValueError(f"Unknown baselines {invalid}; valid choices are {BASELINE_NAMES}")
    return names


def summarize_baseline_results(baseline: str, results: list[dict]) -> dict:
    clean_rows = [{"first_success": row["clean_first_success"], "second_success": False, "attacker_followed": False} for row in results]
    inject_rows = [
        {
            "first_success": row["first_success"],
            "second_success": row["second_success"],
            "attacker_followed": row["attacker_followed"],
        }
        for row in results
    ]
    clean_metrics = summarize_scores(clean_rows)
    inject_metrics = summarize_scores(inject_rows)
    return {
        "baseline": baseline,
        "UIA_clean": clean_metrics["UIA"],
        "UIA_inject": inject_metrics["UIA"],
        "AIA": inject_metrics["AIA"],
        "AIFR": inject_metrics["AIFR"],
        "results": results,
    }


def run_one_baseline(args: argparse.Namespace, baseline: str) -> Path:
    config = load_config(args.config, args.output_dir)
    seed_everything(int(cfg_get(config, "run.seed", 42)))
    paths = get_paths(config)
    modality = get_modality(config)
    output_path = Path(args.shard_output) if args.shard_output else paths.result_dir / "baselines" / f"{modality}_{baseline}.json"
    if output_path.exists() and not args.force:
        print(f"skip existing {output_path}")
        return output_path

    if modality == "audio":
        runner = AudioLanguageRunner(resolve_input_path(config, "model.path"), device=cfg_get(config, "run.device"))
    else:
        runner = VisionLanguageRunner(
            resolve_input_path(config, "model.path"),
            device=cfg_get(config, "run.device"),
            video_options=video_options(config),
        )
    data = load_json(split_json_path(paths, config, "test"))
    if args.num_shards > 1:
        data = [item for index, item in enumerate(data) if index % args.num_shards == args.shard_index]
    media_output_root = paths.output_dir / "baseline_media"
    generation_kwargs = {
        "max_new_tokens": int(cfg_get(config, "generation.max_new_tokens", 128)),
        "do_sample": bool(cfg_get(config, "generation.do_sample", False)),
        "temperature": float(cfg_get(config, "generation.temperature", 0.0)),
    }

    results = []
    for item in tqdm(data, desc=f"{baseline} baseline"):
        clean_conversation = build_baseline_conversation(baseline, item, "clean", modality, config, media_output_root)
        clean_response = runner.generate_conversation(clean_conversation, **generation_kwargs)
        clean_scores = score_response(item, clean_response)

        inject_conversation = build_baseline_conversation(baseline, item, "inject", modality, config, media_output_root)
        inject_response = runner.generate_conversation(inject_conversation, **generation_kwargs)
        inject_scores = score_response(item, inject_response)

        results.append(
            {
                "id": item["id"],
                "prompt": item["clean_prompt"],
                "clean_response": clean_response,
                "inject_response": inject_response,
                "response": inject_response,
                "clean_first_success": clean_scores["first_success"],
                **inject_scores,
            }
        )

    save_json(summarize_baseline_results(baseline, results), output_path)
    print(output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Qwen2-VL baseline defenses on the ARGUS test split.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--baseline", default="all", help="Comma-separated list or 'all'.")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-output")
    args = parser.parse_args()
    if args.num_shards < 1:
        raise ValueError("--num-shards must be at least 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must be in [0, num_shards)")
    if args.shard_output and len(selected_baselines(args.baseline)) != 1:
        raise ValueError("--shard-output can only be used with one baseline")

    for baseline in selected_baselines(args.baseline):
        run_one_baseline(args, baseline)


if __name__ == "__main__":
    main()
