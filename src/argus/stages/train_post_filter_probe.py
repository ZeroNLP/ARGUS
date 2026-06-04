from __future__ import annotations

import argparse

import numpy as np

from argus.config import cfg_get, get_paths, load_config
from argus.io_utils import seed_everything
from argus.media import ensure_image_argus_stage
from argus.probes import ParallelLogisticRegression, load_edit_eval_xy, load_edit_xy, save_metrics, save_pickle
from argus.vector_utils import combined_vectors_from_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config, args.output_dir)
    seed_everything(int(cfg_get(config, "run.seed", 42)))
    ensure_image_argus_stage(config, "train_post_filter_probe")
    paths = get_paths(config)
    out_dir = paths.probe_dir / "final_steering"
    metrics_path = out_dir / "mllm_metrics.json"
    if metrics_path.exists() and not args.force:
        print(f"skip existing {metrics_path}")
        return

    vectors = combined_vectors_from_checkpoint(
        paths.vector_dir,
        paths.probe_dir / "edit",
        paths.probe_dir / "auxiliary",
        int(cfg_get(config, "paper.auxiliary_probe_count", 3)),
    )
    train_path = paths.activation_dir / "mllm_edit_train.pt"
    val_path = paths.activation_dir / "mllm_edit_val.pt"
    ratio = float(cfg_get(config, "paper.probe_train_ratio", 0.8))
    epochs = int(cfg_get(config, "paper.final_steering_probe_epochs", cfg_get(config, "paper.post_filter_probe_epochs", 2000)))
    metrics = []
    for layer, direction in vectors.items():
        # Keep epsilon estimation and inference in the same probe geometry.
        train_x, train_y, val_x, val_y = load_edit_xy(train_path, layer, ratio)
        model = ParallelLogisticRegression(direction, epochs=epochs, seed=int(cfg_get(config, "run.seed", 42)))
        model.fit(train_x, train_y)
        internal_acc = model.score(val_x, val_y)
        external_acc = None
        if val_path.exists():
            ext_x, ext_y = load_edit_eval_xy(val_path, layer)
            external_acc = model.score(ext_x, ext_y)
        save_pickle({"w": model.coef_, "b": model.intercept_, "k": model.k_}, out_dir / f"mllm_layer{layer}.pickle")
        item = {"layer": layer, "eval_acc": round(float(internal_acc), 6)}
        if external_acc is not None:
            item["external_eval_acc"] = round(float(external_acc), 6)
        metrics.append(item)
        print(item)
    save_metrics(metrics, metrics_path)
    print(metrics_path)


if __name__ == "__main__":
    main()
