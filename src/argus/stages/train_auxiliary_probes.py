from __future__ import annotations

import argparse

import numpy as np

from argus.config import cfg_get, get_paths, load_config
from argus.io_utils import seed_everything
from argus.media import ensure_image_argus_stage
from argus.probes import (
    OrthogonalLogisticRegression,
    load_edit_xy,
    load_pickle,
    normalized_probe_vector,
    save_metrics,
    save_pickle,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config, args.output_dir)
    seed_everything(int(cfg_get(config, "run.seed", 42)))
    ensure_image_argus_stage(config, "train_auxiliary_probes")
    paths = get_paths(config)
    count = int(cfg_get(config, "paper.auxiliary_probe_count", 3))
    epochs = int(cfg_get(config, "paper.auxiliary_probe_epochs", 2000))
    ratio = float(cfg_get(config, "paper.probe_train_ratio", 0.8))
    train_path = paths.activation_dir / "mllm_edit_train.pt"
    root = paths.probe_dir / "auxiliary"
    done = root / "auxiliary_metrics.json"
    if done.exists() and not args.force:
        print(f"skip existing {done}")
        return

    all_metrics = []
    for layer in range(28):
        train_x, train_y, val_x, val_y = load_edit_xy(train_path, layer, ratio)
        # Auxiliary probes are trained orthogonal to earlier basis vectors.
        basis = [normalized_probe_vector(load_pickle(paths.probe_dir / "edit" / f"mllm_layer{layer}.pickle"))]
        for aux_index in range(1, count + 1):
            model = OrthogonalLogisticRegression(np.stack(basis), epochs=epochs, seed=int(cfg_get(config, "run.seed", 42)) + aux_index)
            model.fit(train_x, train_y)
            acc = model.score(val_x, val_y)
            save_pickle({"w": model.coef_, "b": model.intercept_}, root / f"aux{aux_index}" / f"mllm_layer{layer}.pickle")
            basis.append(-model.coef_ / np.linalg.norm(model.coef_))
            metric = {"layer": layer, "auxiliary": aux_index, "eval_acc": round(acc, 6)}
            all_metrics.append(metric)
            print(metric)
    save_metrics(all_metrics, done)
    print(done)


if __name__ == "__main__":
    main()
