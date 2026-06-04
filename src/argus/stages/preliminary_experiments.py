from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from tqdm import tqdm

from argus.config import cfg_get, get_paths, load_config, resolve_input_path
from argus.eval_metrics import score_response, summarize_scores
from argus.io_utils import load_json, save_json, seed_everything
from argus.media import ensure_image_argus_stage, get_modality, poison_media, split_json_path
from argus.modeling import VisionLanguageRunner
from argus.preliminary import (
    MultiLayerSteeringController,
    SingleLayerSteeringController,
    install_multi_layer_steering,
    install_single_layer_steering,
)


def format_alpha(alpha: float) -> str:
    return str(float(alpha)).replace("-", "m").replace(".", "p")


def parse_layers(raw: str | None) -> list[int]:
    if raw is None or raw.strip() == "":
        return []
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def default_multi_strength_grid(initial_alpha: float, top_n: int) -> list[float]:
    scales = {
        2: [0.75, 1.0, 1.25],
        3: [0.5, 0.75, 1.0],
        4: [0.375, 0.5, 0.75],
        5: [0.25, 0.375, 0.5],
    }
    return [round(initial_alpha * scale, 6) for scale in scales[top_n]]


def multi_strength_grid(config: dict[str, Any], top_n: int, initial_alpha: float) -> list[float]:
    grid = cfg_get(config, "preliminary.multi_strength_grid", {})
    configured = None
    if isinstance(grid, dict):
        configured = grid.get(str(top_n), grid.get(top_n))
    if configured is not None:
        return [float(value) for value in configured]
    return default_multi_strength_grid(initial_alpha, top_n)


def condition_filename(kind: str, layers: list[int], alpha: float | None, direction: str | None) -> str:
    if kind == "no_steering":
        return "no_steering.json"
    layer_text = "_".join(str(layer) for layer in layers)
    alpha_text = format_alpha(float(alpha if alpha is not None else 0.0))
    direction_text = direction or "none"
    return f"{kind}_layers_{layer_text}_{direction_text}_alpha_{alpha_text}.json"


def condition_path(paths, kind: str, layers: list[int], alpha: float | None, direction: str | None) -> Path:
    return paths.result_dir / "preliminary_conditions" / condition_filename(kind, layers, alpha, direction)


def evaluate_condition(runner, data: list[dict], generation_kwargs: dict, metadata: dict, default_media_type: str) -> tuple[dict, list[dict]]:
    rows = []
    for item in tqdm(data, desc=f"preliminary {metadata['phase']}", leave=False):
        completion = runner.generate(item["clean_prompt"], poison_media(item), **generation_kwargs, media_type="image")
        scores = score_response(item, completion)
        # Saved rows make the search auditable and feed hard-sample selection.
        rows.append(
            {
                "id": item["id"],
                "prompt": item["clean_prompt"],
                "poison_media": poison_media(item),
                "first_answer": item["first_answer"],
                "second_answer": item["second_answer"],
                "second_task": item["second_task"],
                "response": completion,
                **metadata,
                **scores,
            }
        )
    return summarize_scores(rows), rows


def evaluate_steering(
    runner,
    data: list[dict],
    probe_dir: Path,
    layers: list[int],
    alpha: float,
    direction: str,
    generation_kwargs: dict,
    phase: str,
    default_media_type: str,
) -> tuple[dict, list[dict]]:

    if len(layers) == 1:
        controller = SingleLayerSteeringController(
            probe_dir / f"mllm_layer{layers[0]}.pickle",
            layer=layers[0],
            alpha=alpha,
            direction=direction,
        )
        handles = [install_single_layer_steering(runner, controller)]
    else:
        controller = MultiLayerSteeringController(probe_dir, layers=layers, alpha=alpha, direction=direction)
        handles = install_multi_layer_steering(runner, controller)
    try:
        metadata = {
            "phase": phase,
            "layers": layers,
            "layer": layers[0] if len(layers) == 1 else None,
            "alpha": float(alpha),
            "direction": direction,
        }
        metrics, rows = evaluate_condition(runner, data, generation_kwargs, metadata, default_media_type)
    finally:
        for handle in handles:
            handle.remove()
    return {"layers": layers, "layer": layers[0] if len(layers) == 1 else None, "alpha": alpha, "direction": direction, **metrics}, rows


def run_one_condition(
    runner,
    data: list[dict],
    probe_dir: Path,
    generation_kwargs: dict,
    kind: str,
    layers: list[int],
    alpha: float | None,
    direction: str | None,
    output_path: Path,
    force: bool,
    default_media_type: str,
) -> dict:

    if output_path.exists() and not force:
        try:
            cached = load_json(output_path)
            print(f"skip existing {output_path}")
            return cached
        except Exception:
            print(f"rerun incomplete or invalid preliminary output: {output_path}")

    if kind == "no_steering":
        metadata = {"phase": "no_steering", "layers": [], "layer": None, "alpha": None, "direction": None}
        metrics, rows = evaluate_condition(runner, data, generation_kwargs, metadata, default_media_type)
    else:
        if not layers:
            raise ValueError(f"{kind} requires --layers")
        if alpha is None:
            raise ValueError(f"{kind} requires --alpha")
        if direction is None:
            raise ValueError(f"{kind} requires --direction")
        phase = {
            "single_fixed": "single_fixed",
            "single_sensitivity": "single_strength_sensitivity",
            "multi_topn": "multi_topn_strength_sensitivity",
        }[kind]
        metrics, rows = evaluate_steering(runner, data, probe_dir, layers, alpha, direction, generation_kwargs, phase, default_media_type)

    payload = {
        "condition": {
            "kind": kind,
            "layers": layers,
            "layer": layers[0] if len(layers) == 1 else None,
            "alpha": alpha,
            "direction": direction,
        },
        "metrics": metrics,
        "results": rows,
    }
    save_json(payload, output_path)
    print(output_path)
    return payload


def shard_items(data: list[dict], shard_index: int, num_shards: int) -> list[dict]:
    if num_shards <= 1:
        return data
    return [item for index, item in enumerate(data) if index % num_shards == shard_index]


def select_best_candidate(candidates: list[dict]) -> dict:
    safe = [row for row in candidates if row["AIA"] == 0]
    if safe:
        return sorted(safe, key=lambda row: (row["UIA"], -len(row["layers"]), -row["alpha"]), reverse=True)[0]
    return sorted(candidates, key=lambda row: (row["AIA"], -row["UIA"], len(row["layers"]), row["alpha"]))[0]


def load_condition(paths, kind: str, layers: list[int], alpha: float | None, direction: str | None) -> dict:
    return load_json(condition_path(paths, kind, layers, alpha, direction))


def build_fixed_selection(config: dict[str, Any]) -> dict:
    paths = get_paths(config)
    layer_start = int(cfg_get(config, "preliminary.layer_start", 8))
    layer_end = int(cfg_get(config, "preliminary.layer_end", 18))
    initial_alpha = float(cfg_get(config, "preliminary.initial_strength", 10))
    gaps = []
    for layer in range(layer_start, layer_end + 1):
        attack_file = load_condition(paths, "single_fixed", [layer], initial_alpha, "attack")
        defense_file = load_condition(paths, "single_fixed", [layer], initial_alpha, "defense")
        gaps.append(
            {
                "layer": layer,
                "aia_gap": round(attack_file["metrics"]["AIA"] - defense_file["metrics"]["AIA"], 6),
            }
        )
    ranked_layers = [row["layer"] for row in sorted(gaps, key=lambda row: (-row["aia_gap"], row["layer"]))]
    return {"aia_gaps": gaps, "ranked_layers": ranked_layers, "best_single_layer": ranked_layers[0]}


def build_summary(config: dict[str, Any], force: bool = False) -> dict:
    paths = get_paths(config)
    output_path = paths.result_dir / "preliminary_experiments.json"
    prediction_path = paths.result_dir / "preliminary_experiments_predictions.json"
    if output_path.exists() and prediction_path.exists() and not force:
        existing = load_json(output_path)
        if "multi_layer_results" in existing:
            print(f"skip existing {output_path}")
            return existing

    layer_start = int(cfg_get(config, "preliminary.layer_start", 8))
    layer_end = int(cfg_get(config, "preliminary.layer_end", 18))
    initial_alpha = float(cfg_get(config, "preliminary.initial_strength", 10))
    alpha_values = [float(x) for x in cfg_get(config, "preliminary.strength_grid", [5, 10, 15, 20, 25, 30, 35, 40])]
    topn_values = [int(x) for x in cfg_get(config, "preliminary.multi_topn_values", [2, 3, 4, 5])]

    no_steering_file = load_condition(paths, "no_steering", [], None, None)
    fixed_results = []
    predictions = {
        "no_steering": no_steering_file["results"],
        "fixed_layer_results": [],
        "sensitivity_results": [],
        "multi_layer_results": [],
    }
    gaps = []
    for layer in range(layer_start, layer_end + 1):
        attack_file = load_condition(paths, "single_fixed", [layer], initial_alpha, "attack")
        defense_file = load_condition(paths, "single_fixed", [layer], initial_alpha, "defense")
        attack = {"layers": [layer], "layer": layer, "alpha": initial_alpha, "direction": "attack", **attack_file["metrics"]}
        defense = {"layers": [layer], "layer": layer, "alpha": initial_alpha, "direction": "defense", **defense_file["metrics"]}
        fixed_results.extend([attack, defense])
        predictions["fixed_layer_results"].extend(attack_file["results"])
        predictions["fixed_layer_results"].extend(defense_file["results"])
        gaps.append({"layer": layer, "aia_gap": round(attack["AIA"] - defense["AIA"], 6)})

    ranked_layers = [row["layer"] for row in sorted(gaps, key=lambda row: (-row["aia_gap"], row["layer"]))]
    best_single_layer = ranked_layers[0]

    single_sensitivity_results = []
    for alpha in alpha_values:
        condition = load_condition(paths, "single_sensitivity", [best_single_layer], alpha, "defense")
        row = {
            "kind": "single_sensitivity",
            "layers": [best_single_layer],
            "layer": best_single_layer,
            "alpha": alpha,
            "direction": "defense",
            **condition["metrics"],
        }
        single_sensitivity_results.append(row)
        predictions["sensitivity_results"].extend(condition["results"])

    multi_layer_results = []
    for top_n in topn_values:
        layers = ranked_layers[:top_n]
        for alpha in multi_strength_grid(config, top_n, initial_alpha):
            condition = load_condition(paths, "multi_topn", layers, alpha, "defense")
            row = {
                "kind": "multi_topn",
                "top_n": top_n,
                "layers": layers,
                "layer": None,
                "alpha": alpha,
                "direction": "defense",
                **condition["metrics"],
            }
            multi_layer_results.append(row)
            predictions["multi_layer_results"].extend(condition["results"])

    candidates = single_sensitivity_results + multi_layer_results
    best = select_best_candidate(candidates)
    result = {
        "selection_rule": {
            "single_layer_ranking": "rank layers by the AIA gap between attack and defense directions at the fixed alpha",
            "final_layers_and_strength": "among defense steering candidates, choose AIA=0 with maximum UIA; if none reach AIA=0, choose the lowest AIA, then highest UIA",
        },
        "no_steering": no_steering_file["metrics"],
        "fixed_alpha": initial_alpha,
        "layer_range": [layer_start, layer_end],
        "fixed_layer_results": fixed_results,
        "aia_gaps": gaps,
        "ranked_layers": ranked_layers,
        "best_single_layer": int(best_single_layer),
        "strength_grid": alpha_values,
        "single_sensitivity_results": single_sensitivity_results,
        "multi_topn_values": topn_values,
        "multi_strength_grid": {str(top_n): multi_strength_grid(config, top_n, initial_alpha) for top_n in topn_values},
        "multi_layer_results": multi_layer_results,
        "best_layers": [int(layer) for layer in best["layers"]],
        "best_layer": int(best["layers"][0]),
        "best_strength": float(best["alpha"]),
        "best_strength_metrics": best,
        "prediction_path": str(prediction_path),
        "condition_dir": str(paths.result_dir / "preliminary_conditions"),
    }
    save_json(predictions, prediction_path)
    save_json(result, output_path)
    print(output_path)
    return result


def load_validation_data(config: dict[str, Any], paths) -> list[dict]:
    data = load_json(split_json_path(paths, config, "val"))
    max_samples = cfg_get(config, "debug.max_samples")
    if max_samples is not None:
        data = data[: int(max_samples)]
    return data


def build_generation_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": int(cfg_get(config, "generation.max_new_tokens", 128)),
        "do_sample": bool(cfg_get(config, "generation.do_sample", False)),
    }
    if generation_kwargs["do_sample"]:
        generation_kwargs["temperature"] = float(cfg_get(config, "generation.temperature", 0.0))
    return generation_kwargs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--condition-kind", choices=["no_steering", "single_fixed", "single_sensitivity", "multi_topn"])
    parser.add_argument("--layers")
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--direction", choices=["attack", "defense"])
    parser.add_argument("--parallel-phase", choices=["full", "fixed", "sensitivity"], default="full")
    parser.add_argument("--best-layer", type=int)
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
    ensure_image_argus_stage(config, "preliminary_experiments")
    paths = get_paths(config)
    modality = get_modality(config)

    if args.summary_only:
        build_summary(config, force=args.force)
        return

    data = load_validation_data(config, paths)
    data = shard_items(data, args.shard_index, args.num_shards)
    generation_kwargs = build_generation_kwargs(config)
    probe_dir = paths.probe_dir / "edit"

    if args.condition_kind is not None:
        layers = parse_layers(args.layers)
        output_path = Path(args.shard_output) if args.shard_output else condition_path(paths, args.condition_kind, layers, args.alpha, args.direction)
        if output_path.exists() and not args.force:
            try:
                load_json(output_path)
                print(f"skip existing {output_path}")
                return
            except Exception:
                print(f"rerun incomplete or invalid preliminary output: {output_path}")
        runner = VisionLanguageRunner(
            resolve_input_path(config, "model.path"),
            device=cfg_get(config, "run.device"),
        )
        run_one_condition(
            runner,
            data,
            probe_dir,
            generation_kwargs,
            args.condition_kind,
            layers,
            args.alpha,
            args.direction,
            output_path,
            args.force,
            modality,
        )
        return

    # Single-process mode writes one resumable file per completed condition.
    output_path = paths.result_dir / "preliminary_experiments.json"
    prediction_path = paths.result_dir / "preliminary_experiments_predictions.json"
    if output_path.exists() and prediction_path.exists() and not args.force:
        existing = load_json(output_path)
        if "multi_layer_results" in existing:
            print(f"skip existing {output_path}")
            return

    runner = VisionLanguageRunner(
        resolve_input_path(config, "model.path"),
        device=cfg_get(config, "run.device"),
    )
    modality = get_modality(config)
    layer_start = int(cfg_get(config, "preliminary.layer_start", 8))
    layer_end = int(cfg_get(config, "preliminary.layer_end", 18))
    initial_alpha = float(cfg_get(config, "preliminary.initial_strength", 10))
    alpha_values = [float(x) for x in cfg_get(config, "preliminary.strength_grid", [5, 10, 15, 20, 25, 30, 35, 40])]
    topn_values = [int(x) for x in cfg_get(config, "preliminary.multi_topn_values", [2, 3, 4, 5])]

    run_one_condition(
        runner,
        data,
        probe_dir,
        generation_kwargs,
        "no_steering",
        [],
        None,
        None,
        condition_path(paths, "no_steering", [], None, None),
        args.force,
        modality,
    )
    for layer in range(layer_start, layer_end + 1):
        for direction in ["attack", "defense"]:
            run_one_condition(
                runner,
                data,
                probe_dir,
                generation_kwargs,
                "single_fixed",
                [layer],
                initial_alpha,
                direction,
                condition_path(paths, "single_fixed", [layer], initial_alpha, direction),
                args.force,
                modality,
            )

    fixed_selection = build_fixed_selection(config)
    best_single_layer = int(fixed_selection["best_single_layer"])
    for alpha in alpha_values:
        run_one_condition(
            runner,
            data,
            probe_dir,
            generation_kwargs,
            "single_sensitivity",
            [best_single_layer],
            alpha,
            "defense",
            condition_path(paths, "single_sensitivity", [best_single_layer], alpha, "defense"),
            args.force,
            modality,
        )

    ranked_layers = [int(layer) for layer in fixed_selection["ranked_layers"]]
    for top_n in topn_values:
        layers = ranked_layers[:top_n]
        for alpha in multi_strength_grid(config, top_n, initial_alpha):
            run_one_condition(
                runner,
                data,
                probe_dir,
                generation_kwargs,
                "multi_topn",
                layers,
                alpha,
                "defense",
                condition_path(paths, "multi_topn", layers, alpha, "defense"),
                args.force,
                modality,
            )

    build_summary(config, force=True)


if __name__ == "__main__":
    main()
