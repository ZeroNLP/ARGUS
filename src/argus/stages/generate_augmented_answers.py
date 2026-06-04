from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from argus.config import cfg_get, get_paths, load_config, resolve_input_path
from argus.io_utils import load_json, save_json, seed_everything
from argus.media import clean_media, ensure_image_argus_stage, poison_media, split_json_path
from argus.modeling import VisionLanguageRunner, parse_json_object_detailed
from argus.prompts import AUGMENTATION_PROMPT


REPAIR_PROMPT = """You are a strict JSON repair tool.

The previous model output below was intended to be an answer-augmentation JSON object, but it may be malformed, truncated, wrapped in markdown, or repetitive.

Task:
1. Rewrite it into one valid JSON object with exactly these keys:
   - "short_response"
   - "medium_response"
   - "long_response"
2. Keep the content factually consistent with the reference answer.
3. If the malformed output is missing useful content, use the reference answer to fill the field.
4. The short response should be 1-5 words.
5. The medium response should be 1-2 sentences.
6. The long response should be 3-4 sentences.

Return only valid JSON. Do not use markdown fences.

Question:
{question}

Reference answer:
{reference_answer}

Malformed output:
{raw_output}
"""


def fallback_augmented_answer(reference_answer: str) -> dict[str, str]:
    answer = str(reference_answer).strip()
    words = answer.split()
    short = " ".join(words[:5]) if words else answer
    medium = answer
    long = f"{answer} This response keeps the same factual content as the reference answer. It is used only when automatic JSON parsing fails."
    return {"short_response": short, "medium_response": medium, "long_response": long}


def incomplete_meta(raw_output: str, reason: str, parse_mode: str = "fallback") -> dict[str, object]:
    return {
        "complete_json": False,
        "parse_mode": parse_mode,
        "needs_repair": True,
        "repaired": False,
        "repair_failed": False,
        "failure_reason": reason,
        "raw_output": raw_output,
    }


def generate_augmented_answer(
    runner: VisionLanguageRunner,
    prompt: str,
    media_path: str,
    media_type: str,
    reference_answer: str,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    retries: int,
) -> tuple[dict[str, str], dict[str, object]]:

    strict_prompt = prompt + "\nReturn only the JSON object. Do not wrap it in markdown code fences."
    last_output = ""
    last_error = ""
    for attempt in range(retries + 1):
        budget = max_new_tokens * (2**attempt)
        last_output = runner.generate(strict_prompt, media_path, budget, do_sample, temperature, media_type=media_type)
        try:
            parsed, meta = parse_json_object_detailed(last_output)
            return parsed, {
                **meta,
                "needs_repair": not bool(meta["complete_json"]),
                "repaired": False,
                "repair_failed": False,
                "raw_output": last_output if not bool(meta["complete_json"]) else "",
            }
        except (ValueError, OSError) as exc:
            last_error = str(exc)
            print(f"augmentation parse failed on attempt {attempt + 1}/{retries + 1}: {exc}")
    print(f"using fallback augmentation for reference answer: {reference_answer!r}; last model output: {last_output!r}")
    return fallback_augmented_answer(reference_answer), incomplete_meta(last_output, last_error)


def repair_augmented_answer(
    runner: VisionLanguageRunner,
    question: str,
    reference_answer: str,
    media_path: str,
    media_type: str,
    raw_output: str,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
) -> dict[str, str] | None:

    prompt = REPAIR_PROMPT.format(question=question, reference_answer=reference_answer, raw_output=raw_output)
    output = runner.generate(prompt, media_path, max_new_tokens, do_sample, temperature, media_type=media_type)
    try:
        parsed, meta = parse_json_object_detailed(output)
    except ValueError as exc:
        print(f"augmentation repair failed: {exc}")
        return None
    if not meta["complete_json"] or meta["parse_mode"] != "strict":
        print(f"augmentation repair rejected because output was not strict JSON: {output!r}")
        return None
    return parsed


def repair_incomplete_items(
    runner: VisionLanguageRunner,
    data: list[dict],
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    default_media_type: str,
) -> None:

    pending: list[tuple[dict, str]] = []
    for item in data:
        for field in ["first_answer", "second_answer"]:
            meta = item.get(f"{field}_augmented_meta", {})
            if meta.get("needs_repair"):
                pending.append((item, field))
    if not pending:
        return
    print(f"repairing {len(pending)} incomplete augmentation outputs")
    for item, field in tqdm(pending, desc="repair augmentation JSON"):
        meta_key = f"{field}_augmented_meta"
        answer_key = f"{field}_augmented"
        meta = item[meta_key]
        repaired = repair_augmented_answer(
            runner,
            item["first_instruction"] if field == "first_answer" else item["second_instruction"],
            item[field],
            clean_media(item) if field == "first_answer" else poison_media(item),
            "image",
            str(meta.get("raw_output", "")),
            max_new_tokens,
            do_sample,
            temperature,
        )
        if repaired is None:
            meta["repair_failed"] = True
            continue
        item[answer_key] = repaired
        meta["complete_json"] = True
        meta["parse_mode"] = "strict"
        meta["needs_repair"] = False
        meta["repaired"] = True
        meta["repair_failed"] = False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir")
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
    ensure_image_argus_stage(config, "generate_augmented_answers")
    paths = get_paths(config)
    input_path = split_json_path(paths, config, "train")
    output_path = split_json_path(paths, config, "train", augmented=True)
    if args.num_shards > 1:
        output_path = Path(args.shard_output) if args.shard_output else paths.data_dir / "train" / "augmentation_shards" / f"shard_{args.shard_index}.json"
    if output_path.exists() and not args.force:
        print(f"skip existing {output_path}")
        return

    runner = VisionLanguageRunner(
        resolve_input_path(config, "model.path"),
        device=cfg_get(config, "run.device"),
    )
    max_new_tokens = int(cfg_get(config, "augmentation.max_new_tokens", max(512, int(cfg_get(config, "generation.max_new_tokens", 128)))))
    repair_max_new_tokens = int(cfg_get(config, "augmentation.repair_max_new_tokens", max_new_tokens))
    retries = int(cfg_get(config, "augmentation.retries", 2))
    do_sample = bool(cfg_get(config, "generation.do_sample", False))
    temperature = float(cfg_get(config, "generation.temperature", 0.0))
    all_data = load_json(input_path)
    if args.num_shards > 1:
        data = []
        for original_index, item in enumerate(all_data):
            if original_index % args.num_shards == args.shard_index:
                item["_augmentation_index"] = original_index
                data.append(item)
        print(f"augment shard {args.shard_index + 1}/{args.num_shards}: {len(data)} samples")
    else:
        data = all_data
    for item in tqdm(data, desc="augment train answers"):
        # Use the target MLLM so augmentation matches its response style.
        first_prompt = AUGMENTATION_PROMPT.format(item["first_instruction"], item["first_answer"])
        second_prompt = AUGMENTATION_PROMPT.format(item["second_instruction"], item["second_answer"])
        item["first_answer_augmented"], item["first_answer_augmented_meta"] = generate_augmented_answer(
            runner,
            first_prompt,
            clean_media(item),
            "image",
            item["first_answer"],
            max_new_tokens,
            do_sample,
            temperature,
            retries,
        )
        item["second_answer_augmented"], item["second_answer_augmented_meta"] = generate_augmented_answer(
            runner,
            second_prompt,
            poison_media(item),
            "image",
            item["second_answer"],
            max_new_tokens,
            do_sample,
            temperature,
            retries,
        )
    repair_incomplete_items(runner, data, repair_max_new_tokens, do_sample, temperature, "image")
    save_json(data, output_path)
    print(output_path)


if __name__ == "__main__":
    main()
