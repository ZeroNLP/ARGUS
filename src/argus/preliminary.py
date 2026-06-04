from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from .probes import load_pickle
from .modeling import get_language_layers


class SingleLayerSteeringController:
    def __init__(self, probe_path: Path, layer: int, alpha: float, direction: str):
        if direction not in {"attack", "defense"}:
            raise ValueError("direction must be 'attack' or 'defense'")
        self.layer = int(layer)
        self.alpha = float(alpha)
        self.direction = direction
        probe = load_pickle(probe_path)
        weight = np.asarray(probe.coef_).squeeze().astype(np.float32)
        weight = weight / (np.linalg.norm(weight) + 1e-8)
        if direction == "defense":
            weight = -weight
        self.vector = torch.tensor(weight, dtype=torch.float32)

    def hook(self, layer_idx: int):
        def hook_fn(module, inputs, output):
            if layer_idx != self.layer:
                return output
            hidden = output[0]
            vector = self.vector.to(device=hidden.device, dtype=hidden.dtype)
            hidden[:, -1, :] = hidden[:, -1, :] + self.alpha * vector
            return (hidden,) + output[1:]

        return hook_fn


class MultiLayerSteeringController:
    def __init__(self, probe_dir: Path, layers: list[int], alpha: float, direction: str):
        if direction not in {"attack", "defense"}:
            raise ValueError("direction must be 'attack' or 'defense'")
        if not layers:
            raise ValueError("at least one layer is required")
        self.layers = [int(layer) for layer in layers]
        self.alpha = float(alpha)
        self.direction = direction
        self.vectors: dict[int, torch.Tensor] = {}
        for layer in self.layers:
            probe = load_pickle(probe_dir / f"mllm_layer{layer}.pickle")
            weight = np.asarray(probe.coef_).squeeze().astype(np.float32)
            weight = weight / (np.linalg.norm(weight) + 1e-8)
            if direction == "defense":
                weight = -weight
            self.vectors[layer] = torch.tensor(weight, dtype=torch.float32)

    def hook(self, layer_idx: int):
        def hook_fn(module, inputs, output):
            hidden = output[0]
            vector = self.vectors[layer_idx].to(device=hidden.device, dtype=hidden.dtype)
            hidden[:, -1, :] = hidden[:, -1, :] + self.alpha * vector
            return (hidden,) + output[1:]

        return hook_fn


def install_single_layer_steering(runner: Any, controller: SingleLayerSteeringController):
    target = get_language_layers(runner.model)[controller.layer]
    return target.register_forward_hook(controller.hook(controller.layer))


def install_multi_layer_steering(runner: Any, controller: MultiLayerSteeringController):
    model_layers = get_language_layers(runner.model)
    return [model_layers[layer].register_forward_hook(controller.hook(layer)) for layer in controller.layers]
