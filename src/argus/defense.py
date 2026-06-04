from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from .io_utils import load_json
from .media import get_modality
from .probes import load_pickle
from .vector_utils import combined_vectors_from_checkpoint


def adaptive_alpha(hidden: torch.Tensor, w: torch.Tensor, b: torch.Tensor, epsilon: float) -> torch.Tensor:
    h = hidden.float()
    w = w.float().squeeze()
    b = b.float()
    logits = torch.einsum("nd,d->n", h, w) + b
    return torch.relu((logits + epsilon) / (torch.dot(w, w) + 1e-8))


class ImageDefenseController:
    def __init__(self, config: dict[str, Any], paths: Any):
        from .config import cfg_get

        self.detect_layer = int(cfg_get(config, "paper.detection_layer", 6))
        self.modality = get_modality(config)
        self.enable_post_filtering = bool(cfg_get(config, "defense.enable_post_filtering", True))
        self.post_filter_layer = int(cfg_get(config, "paper.post_filter_layer", 20))
        self.epsilon_key = str(cfg_get(config, "paper.epsilon_stat", "mean_score"))
        self.detector = load_pickle(paths.probe_dir / "detection" / f"mllm_layer{self.detect_layer}.pickle")
        # Optional rejection reuses the steering/edit probe.
        self.post_filter_layer_probe = (
            load_pickle(paths.probe_dir / "edit" / f"mllm_layer{self.post_filter_layer}.pickle")
            if self.enable_post_filtering
            else None
        )
        self.epsilon = load_json(paths.epsilon_dir / f"epsilon_{self.modality}.json")["stats"]
        self.final_steering_probes = {}
        self.vectors = combined_vectors_from_checkpoint(
            paths.vector_dir,
            paths.probe_dir / "edit",
            paths.probe_dir / "auxiliary",
            int(cfg_get(config, "paper.auxiliary_probe_count", 3)),
        )
        for layer in self.vectors:
            self.final_steering_probes[layer] = load_pickle(paths.probe_dir / "final_steering" / f"mllm_layer{layer}.pickle")
        self.reset()

    def reset(self) -> None:
        self.step = -1
        self.attack_detected = False
        self.last_post_filter_hidden: np.ndarray | None = None

    def hook(self, layer_idx: int):
        def hook_fn(module, inputs, output):
            hidden = output[0]
            if layer_idx == 0:
                self.step += 1
            last = hidden[:, -1, :]
            if layer_idx == self.detect_layer and self.step == 0:
                self.attack_detected = bool(self.detector.predict(last.detach().cpu().float().numpy())[0] == 1)
            if self.enable_post_filtering and layer_idx == self.post_filter_layer:
                self.last_post_filter_hidden = last.detach().cpu().float().numpy()
            if self.attack_detected and layer_idx in self.vectors:
                if self.step == 0:
                    vector = torch.tensor(self.vectors[layer_idx], device=hidden.device, dtype=hidden.dtype)
                    hidden[:, -1, :] = hidden[:, -1, :] + vector
                else:
                    probe = self.final_steering_probes[layer_idx]
                    w = torch.tensor(probe["w"], device=hidden.device, dtype=hidden.dtype)
                    b = torch.tensor(probe["b"], device=hidden.device, dtype=hidden.dtype)
                    epsilon = float(self.epsilon[str(layer_idx)][self.epsilon_key])
                    alpha = adaptive_alpha(last, w, b, epsilon).to(hidden.dtype)
                    direction = -w.squeeze() / (torch.norm(w.float()) + 1e-8)
                    hidden[:, -1, :] = hidden[:, -1, :] + alpha.view(-1, 1) * direction.to(hidden.dtype)
            return (hidden,) + output[1:]

        return hook_fn

    def post_filter_rejects(self) -> bool:
        if not self.enable_post_filtering or self.post_filter_layer_probe is None or self.last_post_filter_hidden is None:
            return False
        return bool(self.post_filter_layer_probe.predict(self.last_post_filter_hidden)[0] == 1)
