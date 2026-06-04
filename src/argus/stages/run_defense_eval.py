from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from argus.config import cfg_get, get_paths, load_config, resolve_input_path
from argus.defense import ImageDefenseController
from argus.eval_metrics import score_response, summarize_scores
from argus.io_utils import load_json, save_json, seed_everything
from argus.media import clean_media, ensure_image_argus_stage, get_modality, poison_media, split_json_path
from argus.modeling import VisionLanguageRunner, get_language_layers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--mode", choices=["no_defense", "defense"], default="defense")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-output")
    args = parser.parse_args()
    if args.num_shards < 1:
        raise ValueError("--num-shards must be at least 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must be in [0, num_shards)")

    config = load_config(args.config, args.output_dir)
    seed_everything(int(cfg_get(config, "run.seed", 42)))
    ensure_image_argus_stage(config, "run_defense_eval")
    paths = get_paths(config)
    modality = get_modality(config)
    output_name = f"{modality}_no_defense.json" if args.mode == "no_defense" else f"{modality}_argus.json"
    output_path = paths.result_dir / output_name
    if args.num_shards > 1:
        output_path = Path(args.shard_output) if args.shard_output else paths.result_dir / "eval_shards" / f"{args.mode}_{args.shard_index}.json"
    if output_path.exists() and not args.force:
        print(f"skip existing {output_path}")
        return

    runner = VisionLanguageRunner(
        resolve_input_path(config, "model.path"),
        device=cfg_get(config, "run.device"),
    )
    controller = ImageDefenseController(config, paths) if args.mode == "defense" else None
    handles = []
    if controller is not None:
        layer_index = load_json(paths.vector_dir / "layer_index.json")
        # Hooks cover detection, steering, and optional post-filter layers.
        max_hook_layer = max(max(int(x) for x in layer_index), controller.detect_layer)
        if controller.enable_post_filtering:
            max_hook_layer = max(max_hook_layer, int(cfg_get(config, "paper.post_filter_layer", 20)))
        layers = get_language_layers(runner.model)
        for idx in range(max_hook_layer + 1):
            target = layers[idx]
            handles.append(target.register_forward_hook(controller.hook(idx)))

    data = load_json(split_json_path(paths, config, "test"))
    if args.num_shards > 1:
        data = [item for index, item in enumerate(data) if index % args.num_shards == args.shard_index]
    results = []
    clean_scored_rows = []
    inject_scored_rows = []
    for item in tqdm(data, desc=f"{args.mode} eval"):
        # The user prompt stays clean; the attack lives only in the media.
        if controller is not None:
            controller.reset()
        clean_completion = runner.generate(
            item["clean_prompt"],
            clean_media(item),
            max_new_tokens=int(cfg_get(config, "generation.max_new_tokens", 128)),
            do_sample=bool(cfg_get(config, "generation.do_sample", False)),
            temperature=float(cfg_get(config, "generation.temperature", 0.0)),
            media_type="image",
        )
        clean_rejected = controller.post_filter_rejects() if controller is not None else False
        clean_attack_detected = controller.attack_detected if controller is not None else False
        if clean_rejected:
            clean_completion = "Sorry!"
        clean_scores = score_response(item, clean_completion)
        clean_scored_rows.append(clean_scores)

        if controller is not None:
            controller.reset()
        inject_completion = runner.generate(
            item["clean_prompt"],
            poison_media(item),
            max_new_tokens=int(cfg_get(config, "generation.max_new_tokens", 128)),
            do_sample=bool(cfg_get(config, "generation.do_sample", False)),
            temperature=float(cfg_get(config, "generation.temperature", 0.0)),
            media_type="image",
        )
        rejected = controller.post_filter_rejects() if controller is not None else False
        inject_attack_detected = controller.attack_detected if controller is not None else False
        if rejected:
            inject_completion = "Sorry!"
        inject_scores = score_response(item, inject_completion)
        inject_scored_rows.append(inject_scores)
        results.append(
            {
                "id": item["id"],
                "prompt": item["clean_prompt"],
                "clean_response": clean_completion,
                "inject_response": inject_completion,
                "response": inject_completion,
                "clean_attack_detected": clean_attack_detected,
                "attack_detected": inject_attack_detected,
                "clean_post_filter_rejected": clean_rejected,
                "post_filter_rejected": rejected,
                "clean_first_success": clean_scores["first_success"],
                **inject_scores,
            }
        )

    clean_metrics = summarize_scores(clean_scored_rows)
    inject_metrics = summarize_scores(inject_scored_rows)
    summary = {
        "mode": args.mode,
        "defense_enabled": args.mode == "defense",
        "post_filtering_enabled": bool(controller.enable_post_filtering) if controller is not None else False,
        "UIA_clean": clean_metrics["UIA"],
        "UIA_inject": inject_metrics["UIA"],
        "AIA": inject_metrics["AIA"],
        "AIFR": inject_metrics["AIFR"],
        "results": results,
    }
    save_json(summary, output_path)
    for handle in handles:
        handle.remove()
    print(output_path)


if __name__ == "__main__":
    main()
