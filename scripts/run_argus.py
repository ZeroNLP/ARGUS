from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the image-only ARGUS pipeline.")
    parser.add_argument("--config", default=str(ROOT_DIR / "configs" / "argus.yaml"))
    parser.add_argument("--output-dir")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--gpu-id")
    args = parser.parse_args()

    command = [sys.executable, str(ROOT_DIR / "scripts" / "run_all.py"), "--skip-data", "--config", args.config]
    if args.output_dir:
        command.extend(["--output-dir", args.output_dir])
    if args.force:
        command.append("--force")
    if args.dry_run:
        command.append("--dry-run")
    if args.allow_missing:
        command.append("--allow-missing")
    if args.gpu_id:
        command.extend(["--gpu-id", args.gpu_id])
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
