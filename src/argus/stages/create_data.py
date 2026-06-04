from __future__ import annotations

import argparse

from argus.config import get_paths, load_config
from argus.data_builder import build_datasets
from argus.io_utils import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config, args.output_dir)
    seed_everything(int(config.get("run", {}).get("seed", 42)))
    paths = get_paths(config)
    outputs = build_datasets(config, paths.data_dir, force=args.force)
    for split, path in outputs.items():
        print(f"{split}: {path}")


if __name__ == "__main__":
    main()
