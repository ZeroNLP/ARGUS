from __future__ import annotations

import argparse

from argus.config import cfg_get, get_paths, load_config
from argus.io_utils import seed_everything
from argus.media import ensure_image_argus_stage
from argus.probes import (
    load_detection_xy,
    load_edit_eval_xy,
    load_edit_xy,
    save_metrics,
    save_pickle,
    train_logistic,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--kind", choices=["detection", "edit"], required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config, args.output_dir)
    seed_everything(int(cfg_get(config, "run.seed", 42)))
    ensure_image_argus_stage(config, "train_probe")
    paths = get_paths(config)
    layer_count = 28
    ratio = float(cfg_get(config, "paper.probe_train_ratio", 0.8))
    out_dir = paths.probe_dir / args.kind
    metrics_path = out_dir / "mllm_metrics.json"
    if metrics_path.exists() and not args.force:
        print(f"skip existing {metrics_path}")
        return

    train_path = paths.activation_dir / f"mllm_{args.kind}_train.pt"
    val_path = paths.activation_dir / f"mllm_{args.kind}_val.pt"
    metrics = []
    for layer in range(layer_count):
        # Detection separates clean/injected; edit separates user/attacker answers.
        if args.kind == "detection":
            train_x, train_y, val_x, val_y = load_detection_xy(train_path, layer, ratio)
        else:
            train_x, train_y, val_x, val_y = load_edit_xy(train_path, layer, ratio)
        model = train_logistic(train_x, train_y)
        internal_acc = model.score(val_x, val_y)
        if val_path.exists() and args.kind == "edit":
            ext_x, ext_y = load_edit_eval_xy(val_path, layer)
            external_acc = model.score(ext_x, ext_y)
        else:
            external_acc = None
        save_pickle(model, out_dir / f"mllm_layer{layer}.pickle")
        item = {"layer": layer, "eval_acc": round(float(internal_acc), 6)}
        if external_acc is not None:
            item["external_eval_acc"] = round(float(external_acc), 6)
        metrics.append(item)
        print(item)
    save_metrics(metrics, metrics_path)
    print(metrics_path)


if __name__ == "__main__":
    main()
