from __future__ import annotations

import argparse

import numpy as np
import torch

from argus.config import cfg_get, get_paths, load_config
from argus.io_utils import load_json, save_json, seed_everything
from argus.media import ensure_image_argus_stage, get_modality
from argus.probes import load_pickle
from argus.vector_utils import combined_vectors_from_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config, args.output_dir)
    seed_everything(int(cfg_get(config, "run.seed", 42)))
    ensure_image_argus_stage(config, "estimate_epsilon")
    paths = get_paths(config)
    modality = get_modality(config)
    output_path = paths.epsilon_dir / f"epsilon_{modality}.json"
    if output_path.exists() and not args.force:
        print(f"skip existing {output_path}")
        return

    vectors = combined_vectors_from_checkpoint(
        paths.vector_dir,
        paths.probe_dir / "edit",
        paths.probe_dir / "auxiliary",
        int(cfg_get(config, "paper.auxiliary_probe_count", 3)),
    )
    dataset = torch.load(paths.activation_dir / "mllm_edit_train.pt", map_location="cpu")
    stats = {}
    for layer, vector in vectors.items():
        # Epsilon is estimated only from already-safe user-following examples.
        probe = load_pickle(paths.probe_dir / "final_steering" / f"mllm_layer{layer}.pickle")
        w = np.asarray(probe["w"]).squeeze()
        b = float(probe["b"])
        activations = np.vstack([item["first_activation"][layer].float().numpy() for item in dataset])
        scores = activations @ w + b
        negative_scores = scores[scores < 0]
        if len(negative_scores) == 0:
            raise ValueError(f"No safely classified user-following samples for layer {layer}; cannot estimate epsilon.")
        stats[str(layer)] = {
            "median_score": float(-np.median(negative_scores)),
            "mean_score": float(-np.mean(negative_scores)),
            "std_score": float(np.std(negative_scores)),
            "count": int(len(negative_scores)),
            "alpha": float(np.linalg.norm(vector)),
        }
    preliminary = load_json(paths.result_dir / "preliminary_experiments.json")
    save_json({"stats": stats, "alpha": float(preliminary["best_strength"])}, output_path)
    print(output_path)


if __name__ == "__main__":
    main()
