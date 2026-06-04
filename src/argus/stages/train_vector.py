from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from argus.activations import choose_augmented_answer
from argus.config import cfg_get, get_paths, load_config, resolve_input_path
from argus.eval_metrics import score_response, summarize_scores
from argus.io_utils import load_json, save_json, seed_everything
from argus.media import ensure_image_argus_stage, media_block, poison_media, split_json_path
from argus.modeling import get_language_layers, set_cuda_device, silence_greedy_sampling_warnings
from argus.vector_utils import LayerMixtureEditor, load_basis_vectors


class TrainDataset(Dataset):
    def __init__(self, data_path: Path, processor: Any):
        from qwen_vl_utils import process_vision_info

        self.items = load_json(data_path)
        self.processor = processor
        self.process_vision_info = process_vision_info

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.items[idx]
        block = media_block(poison_media(item), "image")
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": item["clean_prompt"]},
                    block,
                ],
            }
        ]
        prompt = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = self.process_vision_info(conversation)
        return {
            "prompt": prompt,
            "completion": choose_augmented_answer(item, "first_answer"),
            "image_input": image_inputs,
        }


def make_collate(processor: Any):
    def collate(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        prompts = [item["prompt"] for item in batch]
        completions = [item["completion"] for item in batch]
        full_texts = [p + c for p, c in zip(prompts, completions)]
        images = [media for item in batch for media in (item["image_input"] or [])]
        images = images or None
        inputs = processor(text=full_texts, images=images, padding=True, return_tensors="pt")
        prompt_inputs = processor(text=prompts, images=images, padding=True, return_tensors="pt")
        prompt_lengths = [int(torch.sum(row != processor.tokenizer.pad_token_id)) for row in prompt_inputs["input_ids"]]
        labels = inputs.input_ids.clone()
        edit_mask = torch.zeros_like(inputs.input_ids, dtype=torch.bool)
        seq_len = inputs.input_ids.shape[1]
        for i, prompt_len in enumerate(prompt_lengths):
            # Train loss starts at the completion; steering starts at answer time.
            total_len = int(inputs.attention_mask[i].sum())
            padding_len = seq_len - total_len
            completion_start = padding_len + prompt_len
            labels[i, :completion_start] = -100
            edit_start = max(padding_len + prompt_len - 1, 0)
            edit_mask[i, edit_start : padding_len + total_len] = True
        return {"input_ids": inputs.input_ids, "attention_mask": inputs.attention_mask, "labels": labels, "edit_mask": edit_mask}

    return collate


class MultiLayerEditor(nn.Module):
    def __init__(self, layers: list[int], editors: list[LayerMixtureEditor]):
        super().__init__()
        self.layers = layers
        self.editors = nn.ModuleList(editors)
        self.edit_mask: torch.Tensor | None = None
        self.generation_enabled = False

    def hook(self, editor_index: int):
        def hook_fn(module, inputs, output):
            hidden = output[0]
            vector = self.editors[editor_index].vector(hidden.device, hidden.dtype)
            if self.edit_mask is not None:
                mask = self.edit_mask.to(hidden.device).unsqueeze(-1).to(hidden.dtype)
                hidden = hidden + vector * mask
            elif self.generation_enabled:
                # Early-stop generation uses the inference-time last-token hook.
                hidden[:, -1, :] = hidden[:, -1, :] + vector
            return (hidden,) + output[1:]

        return hook_fn


def capture_editor_state(multi: MultiLayerEditor, layers: list[int], strength: float) -> dict[str, Any]:
    state: dict[str, Any] = {"layers": layers, "strength": strength}
    for idx, editor in enumerate(multi.editors):
        state[f"editors.{idx}.a"] = editor.a.detach().cpu().clone()
    return state


def restore_editor_state(multi: MultiLayerEditor, state: dict[str, Any]) -> None:
    for idx, editor in enumerate(multi.editors):
        editor.a.data.copy_(state[f"editors.{idx}.a"].to(device=editor.a.device, dtype=editor.a.dtype))


def save_editor_checkpoint(save_dir: Path, final_path: Path, state: dict[str, Any]) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(state, final_path)
    with (save_dir / "layer_index.json").open("w", encoding="utf-8") as f:
        import json

        json.dump(state["layers"], f)


def _same_preliminary_condition(row: dict[str, Any], layers: list[int], strength: float) -> bool:
    row_layers = [int(layer) for layer in row.get("layers", [])]
    row_alpha = row.get("alpha")
    if row_alpha is None:
        return False
    return row_layers == layers and abs(float(row_alpha) - float(strength)) < 1e-6


def select_hard_validation_items(
    prediction_path: Path,
    val_data: list[dict[str, Any]],
    sample_count: int,
    best_layers: list[int],
    best_strength: float,
) -> list[dict[str, Any]]:

    predictions = load_json(prediction_path)
    by_id = {item["id"]: item for item in val_data}
    attacked_without_defense = {
        row["id"]
        for row in predictions.get("no_steering", [])
        if row.get("id") in by_id and bool(row.get("second_success"))
    }
    rows = (
        predictions.get("fixed_layer_results", [])
        + predictions.get("sensitivity_results", [])
        + predictions.get("multi_layer_results", [])
    )
    best_failure_ids = {
        row.get("id")
        for row in rows
        if row.get("direction") == "defense"
        and row.get("id") in attacked_without_defense
        and _same_preliminary_condition(row, best_layers, best_strength)
        and not bool(row.get("second_success"))
        and not bool(row.get("first_success"))
    }
    recovery_stats: dict[Any, dict[str, int]] = {
        item_id: {"other_defense_uia": 0, "other_defense_count": 0, "other_defense_aia": 0}
        for item_id in best_failure_ids
    }
    for row in rows:
        item_id = row.get("id")
        if row.get("direction") != "defense" or item_id not in recovery_stats:
            continue
        if _same_preliminary_condition(row, best_layers, best_strength):
            continue
        recovery_stats[item_id]["other_defense_count"] += 1
        recovery_stats[item_id]["other_defense_uia"] += int(bool(row.get("first_success")))
        recovery_stats[item_id]["other_defense_aia"] += int(bool(row.get("second_success")))
    ranked = sorted(
        recovery_stats.items(),
        key=lambda pair: (
            pair[1]["other_defense_uia"],
            pair[1]["other_defense_count"],
            -pair[1]["other_defense_aia"],
        ),
        reverse=True,
    )
    selected = [by_id[item_id] for item_id, _ in ranked[:sample_count] if item_id in by_id]
    if len(selected) < sample_count:
        print(
            f"warning: selected only {len(selected)} early-stop samples matching "
            "no-defense attack success, selected-defense attack failure, and selected-defense user-answer failure"
        )
    return selected


@torch.no_grad()
def generate_with_editor(
    model: Any,
    processor: Any,
    process_vision_info: Any,
    multi: MultiLayerEditor,
    item: dict[str, Any],
    generation_kwargs: dict[str, Any],
) -> str:

    block = media_block(poison_media(item), "image")
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": item["clean_prompt"]},
                block,
            ],
        }
    ]
    prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(conversation)
    inputs = processor(text=[prompt], images=image_inputs, padding=True, return_tensors="pt")
    target = "cuda" if torch.cuda.is_available() else next(model.parameters()).device
    inputs = inputs.to(target)
    multi.generation_enabled = True
    try:
        output = model.generate(**inputs, **generation_kwargs)
    finally:
        multi.generation_enabled = False
    trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, output)]
    return processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()


def evaluate_editor_checkpoint(
    model: Any,
    processor: Any,
    process_vision_info: Any,
    multi: MultiLayerEditor,
    items: list[dict[str, Any]],
    generation_kwargs: dict[str, Any],
) -> tuple[float, dict[str, float], list[dict[str, Any]]]:

    was_training = model.training
    model.eval()
    rows = []
    try:
        for item in tqdm(items, desc="vector early-stop eval", leave=False):
            completion = generate_with_editor(
                model,
                processor,
                process_vision_info,
                multi,
                item,
                generation_kwargs,
            )
            scores = score_response(item, completion)
            rows.append(
                {
                    "id": item["id"],
                    "prompt": item["clean_prompt"],
                    "poison_media": poison_media(item),
                    "first_answer": item["first_answer"],
                    "second_answer": item["second_answer"],
                    "second_task": item["second_task"],
                    "response": completion,
                    **scores,
                }
            )
    finally:
        multi.generation_enabled = False
        if was_training:
            model.train()
    metrics = summarize_scores(rows)
    objective = round(metrics["UIA"] - metrics["AIA"], 6)
    return objective, metrics, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config, args.output_dir)
    seed_everything(int(cfg_get(config, "run.seed", 42)))
    ensure_image_argus_stage(config, "train_vector")
    set_cuda_device(cfg_get(config, "run.device"))
    paths = get_paths(config)
    save_dir = paths.vector_dir
    final_path = save_dir / "activation_editor.pth"
    if final_path.exists() and not args.force:
        print(f"skip existing {final_path}")
        return

    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        str(resolve_input_path(config, "model.path")),
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    silence_greedy_sampling_warnings(model)
    processor = AutoProcessor.from_pretrained(str(resolve_input_path(config, "model.path")))
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    for param in model.parameters():
        param.requires_grad = False

    preliminary_path = paths.result_dir / "preliminary_experiments.json"
    if not preliminary_path.exists():
        raise FileNotFoundError(
            f"{preliminary_path} is required. Run the preliminary_experiments stage before train_vector."
        )
    preliminary = load_json(preliminary_path)
    prediction_path = Path(preliminary.get("prediction_path", paths.result_dir / "preliminary_experiments_predictions.json"))
    if not prediction_path.exists():
        raise FileNotFoundError(
            f"{prediction_path} is required. Re-run preliminary_experiments so train_vector can select hard validation samples."
        )
    layers = [int(layer) for layer in preliminary.get("best_layers", [preliminary["best_layer"]])]
    strength = float(preliminary["best_strength"])
    aux_count = int(cfg_get(config, "paper.auxiliary_probe_count", 3))
    editors = [
        LayerMixtureEditor(load_basis_vectors(layer, paths.probe_dir / "edit", paths.probe_dir / "auxiliary", aux_count), strength)
        for layer in layers
    ]
    multi = MultiLayerEditor(layers, editors).to(next(model.parameters()).device)
    model.multi_editor = multi
    handles = []
    model_layers = get_language_layers(model)
    for idx, layer in enumerate(layers):
        # Section 4 search supplies the intervention layers and strength.
        target = model_layers[layer]
        handles.append(target.register_forward_hook(multi.hook(idx)))

    dataset = TrainDataset(split_json_path(paths, config, "train", augmented=True), processor)
    loader = DataLoader(
        dataset,
        batch_size=int(cfg_get(config, "paper.vector_batch_size", 1)),
        shuffle=True,
        collate_fn=make_collate(processor),
        num_workers=int(cfg_get(config, "run.num_workers", 2)),
    )
    optimizer = torch.optim.AdamW(multi.parameters(), lr=float(cfg_get(config, "paper.vector_learning_rate", 0.01)))
    val_data = load_json(split_json_path(paths, config, "val"))
    early_stop_count = int(cfg_get(config, "paper.vector_early_stop_samples", 20))
    early_stop_items = select_hard_validation_items(prediction_path, val_data, early_stop_count, layers, strength)
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": int(cfg_get(config, "generation.max_new_tokens", 128)),
        "do_sample": bool(cfg_get(config, "generation.do_sample", False)),
    }
    if generation_kwargs["do_sample"]:
        generation_kwargs["temperature"] = float(cfg_get(config, "generation.temperature", 0.0))
    eval_interval = max(1, int(cfg_get(config, "paper.vector_eval_interval_steps", 100)))
    patience = int(cfg_get(config, "paper.vector_early_stop_patience", 2))
    best_objective = float("-inf")
    best_state: dict[str, Any] | None = None
    stale_evals = 0
    global_step = 0
    last_eval_step = 0
    early_stop_history: list[dict[str, Any]] = []

    def run_early_stop_eval(step: int) -> bool:
        nonlocal best_objective, best_state, stale_evals, last_eval_step
        last_eval_step = step
        objective, metrics, rows = evaluate_editor_checkpoint(
            model,
            processor,
            process_vision_info,
            multi,
            early_stop_items,
            generation_kwargs,
        )
        improved = objective > best_objective
        if improved:
            best_objective = objective
            best_state = capture_editor_state(multi, layers, strength)
            stale_evals = 0
        else:
            stale_evals += 1
        early_stop_history.append(
            {
                "step": step,
                "objective": objective,
                "metrics": metrics,
                "improved": improved,
                "stale_evals": stale_evals,
                "results": rows,
            }
        )
        print({"step": step, "objective": objective, **metrics, "improved": improved, "stale_evals": stale_evals})
        return stale_evals >= patience

    model.train()
    epochs = int(cfg_get(config, "paper.vector_epochs", 2))
    should_stop = False
    for epoch in range(epochs):
        for batch in tqdm(loader, desc=f"vector epoch {epoch + 1}/{epochs}"):
            optimizer.zero_grad()
            multi.edit_mask = batch["edit_mask"].to(next(model.parameters()).device)
            outputs = model(
                input_ids=batch["input_ids"].to(next(model.parameters()).device),
                attention_mask=batch["attention_mask"].to(next(model.parameters()).device),
                labels=batch["labels"].to(next(model.parameters()).device),
            )
            loss = outputs.loss
            multi.edit_mask = None
            loss.backward()
            optimizer.step()
            global_step += 1
            if global_step % eval_interval == 0:
                should_stop = run_early_stop_eval(global_step)
                if should_stop:
                    break
        if should_stop:
            break

    if global_step != last_eval_step:
        run_early_stop_eval(global_step)
    if best_state is None:
        best_state = capture_editor_state(multi, layers, strength)
    restore_editor_state(multi, best_state)

    save_editor_checkpoint(save_dir, final_path, best_state)
    save_json(
        {
            "selection_rule": "Select samples where no-defense attack succeeds and the selected best defense blocks the attack but misses the user answer; rank by how often other defense settings preserve the user answer.",
            "best_layers": layers,
            "best_strength": strength,
            "sample_count": len(early_stop_items),
            "sample_ids": [item["id"] for item in early_stop_items],
            "eval_interval_steps": eval_interval,
            "patience": patience,
            "best_objective": best_objective,
            "history": early_stop_history,
        },
        save_dir / "vector_early_stop.json",
    )
    for handle in handles:
        handle.remove()
    print(final_path)


if __name__ == "__main__":
    main()
