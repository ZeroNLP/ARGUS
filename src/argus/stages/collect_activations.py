from __future__ import annotations

import argparse
from pathlib import Path

from argus.activations import collect_detection_activations, collect_edit_activations
from argus.config import cfg_get, get_paths, load_config, resolve_input_path
from argus.io_utils import seed_everything
from argus.media import ensure_image_argus_stage, split_json_path
from argus.modeling import VisionLanguageRunner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--mode", choices=["detection", "edit"], required=True)
    parser.add_argument("--split", choices=["train", "val"], required=True)
    parser.add_argument("--force", action="store_true")
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
    paths = get_paths(config)
    ensure_image_argus_stage(config, "collect_activations")
    suffix = f"mllm_{args.mode}_{args.split}.pt"
    output_path = paths.activation_dir / suffix
    if args.num_shards > 1:
        shard_name = f"{args.mode}_{args.split}_shard_{args.shard_index}.pt"
        output_path = Path(args.shard_output) if args.shard_output else paths.activation_dir / "shards" / shard_name
    if output_path.exists() and not args.force:
        print(f"skip existing {output_path}")
        return

    # Train uses augmented answers; validation keeps held-out GLUE answers.
    if args.mode == "edit" and args.split == "train":
        data_json = split_json_path(paths, config, "train", augmented=True)
    else:
        data_json = split_json_path(paths, config, args.split)
    runner = VisionLanguageRunner(
        resolve_input_path(config, "model.path"),
        device=cfg_get(config, "run.device"),
    )
    if args.mode == "detection":
        collect_detection_activations(runner, data_json, output_path, args.shard_index, args.num_shards, "image")
    else:
        collect_edit_activations(
            runner,
            data_json,
            output_path,
            use_augmented=(args.split == "train"),
            shard_index=args.shard_index,
            num_shards=args.num_shards,
            default_media_type="image",
        )
    print(output_path)


if __name__ == "__main__":
    main()
