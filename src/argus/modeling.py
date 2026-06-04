from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from .media import media_block


def set_cuda_device(device: str | None) -> None:
    if device is not None and "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device)


def _resolve_attr(root: Any, dotted_path: str) -> Any | None:
    value = root
    for part in dotted_path.split("."):
        if not hasattr(value, part):
            return None
        value = getattr(value, part)
    return value


def get_language_layers(model: Any) -> Any:
    candidates = [
        "language_model.base_model.layers",
        "language_model.model.layers",
        "language_model.layers",
        "model.language_model.base_model.layers",
        "model.language_model.model.layers",
        "model.language_model.layers",
        "model.model.layers",
        "model.layers",
        "model.decoder.layers",
        "model.base_model.layers",
        "base_model.layers",
        "transformer.layers",
    ]
    for path in candidates:
        layers = _resolve_attr(model, path)
        if layers is not None:
            return layers
    for name, module in model.named_modules():
        if "visual" in name or "vision" in name:
            continue
        if not name.endswith("layers") or not isinstance(module, torch.nn.ModuleList) or len(module) == 0:
            continue
        first_layer = module[0]
        if hasattr(first_layer, "self_attn") or hasattr(first_layer, "mlp"):
            return module
    available = [name for name, _ in model.named_children()]
    layer_like = [
        name
        for name, module in model.named_modules()
        if name.endswith("layers") and isinstance(module, torch.nn.ModuleList)
    ]
    raise AttributeError(f"Cannot locate language-model layers. Top-level children: {available}; layer-like modules: {layer_like[:20]}")


def silence_greedy_sampling_warnings(model: Any) -> None:
    generation_config = getattr(model, "generation_config", None)
    if generation_config is None:
        return
    if getattr(generation_config, "do_sample", False) is False and getattr(generation_config, "top_k", None) == 1:
        generation_config.top_k = None


class VisionLanguageRunner:
    def __init__(
        self,
        model_path: str | Path,
        device: str | None = None,
        dtype: torch.dtype = torch.bfloat16,
        video_options: dict[str, Any] | None = None,
    ):
        set_cuda_device(device)
        from qwen_vl_utils import process_vision_info
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        self.process_vision_info = process_vision_info
        self.video_options = video_options or {}
        self.processor = AutoProcessor.from_pretrained(str(model_path))
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            str(model_path),
            device_map="auto",
            torch_dtype=dtype,
        ).eval()
        silence_greedy_sampling_warnings(self.model)
        if getattr(self.processor, "tokenizer", None) is not None:
            self.processor.tokenizer.padding_side = "left"
            if self.processor.tokenizer.pad_token is None:
                self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
                self.model.config.pad_token_id = self.model.config.eos_token_id

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _conversation(self, text: str, media: str | Path, media_type: str = "image") -> list[dict[str, Any]]:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    media_block(media, media_type, self.video_options),
                ],
            }
        ]

    def _inputs(self, text: str, media: str | Path, prefix: str = "", media_type: str = "image"):
        conversation = self._conversation(text, media, media_type)
        prompt = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        prompt = prompt + prefix
        image_inputs, video_inputs = self.process_vision_info(conversation)
        inputs = self.processor(
            text=[prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        target = "cuda" if torch.cuda.is_available() else self.device
        return inputs.to(target)

    def _conversation_inputs(self, conversation: list[dict[str, Any]]):
        prompt = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = self.process_vision_info(conversation)
        inputs = self.processor(
            text=[prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        target = "cuda" if torch.cuda.is_available() else self.device
        return inputs.to(target)

    @torch.no_grad()
    def generate_conversation(
        self,
        conversation: list[dict[str, Any]],
        max_new_tokens: int = 128,
        do_sample: bool = False,
        temperature: float = 0.0,
    ) -> str:

        inputs = self._conversation_inputs(conversation)
        kwargs: dict[str, Any] = {"max_new_tokens": max_new_tokens, "do_sample": do_sample}
        if do_sample:
            kwargs["temperature"] = temperature
        output = self.model.generate(**inputs, **kwargs)
        trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, output)]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

    @torch.no_grad()
    def generate(
        self,
        text: str,
        media: str | Path,
        max_new_tokens: int = 128,
        do_sample: bool = False,
        temperature: float = 0.0,
        media_type: str = "image",
    ) -> str:

        inputs = self._inputs(text, media, media_type=media_type)
        kwargs: dict[str, Any] = {"max_new_tokens": max_new_tokens, "do_sample": do_sample}
        if do_sample:
            kwargs["temperature"] = temperature
        output = self.model.generate(**inputs, **kwargs)
        trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, output)]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

    @torch.no_grad()
    def activation(self, text: str, media: str | Path, prefix: str = "", token: int = -1, media_type: str = "image") -> torch.Tensor:
        inputs = self._inputs(text, media, prefix=prefix, media_type=media_type)
        outputs = self.model(**inputs, output_hidden_states=True)
        # Skip the embedding state so index 0 corresponds to transformer layer 0.
        states = [hidden[:, token].detach().cpu() for hidden in outputs.hidden_states[1:]]
        return torch.stack(states).squeeze(1)


def _clean_json_text(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    return re.sub(r"\s*```$", "", cleaned)


def parse_json_object_detailed(text: str) -> tuple[dict[str, str], dict[str, Any]]:
    required = ["short_response", "medium_response", "long_response"]
    cleaned = _clean_json_text(text)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(cleaned[start : end + 1])
            missing = [key for key in required if key not in parsed]
            if missing:
                raise ValueError(f"Model output missing keys {missing}: {text}")
            return {key: str(parsed[key]).strip() for key in required}, {"complete_json": True, "parse_mode": "strict"}
        except json.JSONDecodeError:
            pass

    parsed: dict[str, str] = {}
    for index, key in enumerate(required):
        next_keys = "|".join(re.escape(k) for k in required[index + 1 :])
        if next_keys:
            pattern = rf'"{key}"\s*:\s*"(.*?)(?="\s*,\s*"(?:{next_keys})"\s*:|"\s*\}}|$)'
        else:
            pattern = rf'"{key}"\s*:\s*"(.*?)(?="\s*\}}|$)'
        match = re.search(pattern, cleaned, flags=re.DOTALL)
        if match:
            value = match.group(1).strip().rstrip(",").strip()
            value = re.sub(r"\s*```$", "", value).strip()
            try:
                value = json.loads(f'"{value}"')
            except json.JSONDecodeError:
                value = value.replace('\\"', '"').replace("\\n", "\n")
            parsed[key] = str(value).strip()

    missing = [key for key in required if key not in parsed]
    if missing:
        raise ValueError(f"Model output missing keys {missing}: {text}")
    return {key: str(parsed[key]).strip() for key in required}, {"complete_json": False, "parse_mode": "tolerant"}


def parse_json_object(text: str) -> dict[str, str]:
    parsed, _ = parse_json_object_detailed(text)
    return parsed


class AudioLanguageRunner:
    def __init__(
        self,
        model_path: str | Path,
        device: str | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        set_cuda_device(device)
        import librosa
        from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

        self.librosa = librosa
        self.processor = AutoProcessor.from_pretrained(str(model_path))
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            str(model_path),
            device_map="auto",
            torch_dtype=dtype,
        ).eval()
        silence_greedy_sampling_warnings(self.model)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _audio_path_from_conversation(self, conversation: list[dict[str, Any]]) -> str:
        for message in conversation:
            content = message.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if block.get("type") == "audio":
                        return str(block.get("audio_url") or block.get("audio"))
        raise ValueError("Audio conversation does not contain an audio block.")

    @torch.no_grad()
    def generate_conversation(
        self,
        conversation: list[dict[str, Any]],
        max_new_tokens: int = 128,
        do_sample: bool = False,
        temperature: float = 0.0,
    ) -> str:

        audio_path = self._audio_path_from_conversation(conversation)
        audio, _ = self.librosa.load(audio_path, sr=self.processor.feature_extractor.sampling_rate)
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(
            text=prompt,
            audios=[audio],
            return_tensors="pt",
            padding=True,
            sampling_rate=self.processor.feature_extractor.sampling_rate,
        )
        target = "cuda" if torch.cuda.is_available() else self.device
        inputs = inputs.to(target)
        kwargs: dict[str, Any] = {"max_new_tokens": max_new_tokens, "do_sample": do_sample}
        if do_sample:
            kwargs["temperature"] = temperature
        output = self.model.generate(**inputs, **kwargs)
        trimmed = output[:, inputs.input_ids.size(1) :]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
