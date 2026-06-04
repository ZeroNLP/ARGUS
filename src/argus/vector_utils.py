from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .probes import load_pickle, normalized_probe_vector


def load_basis_vectors(layer: int, edit_probe_dir: Path, auxiliary_dir: Path, auxiliary_count: int) -> torch.Tensor:
    vectors = [normalized_probe_vector(load_pickle(edit_probe_dir / f"mllm_layer{layer}.pickle"))]
    for i in range(1, auxiliary_count + 1):
        aux_path = auxiliary_dir / f"aux{i}" / f"mllm_layer{layer}.pickle"
        aux = load_pickle(aux_path)
        if isinstance(aux, dict):
            coef = np.asarray(aux["w"]).squeeze()
            coef = coef / np.linalg.norm(coef)
            vectors.append(-coef)
        else:
            vectors.append(normalized_probe_vector(aux))
    return torch.tensor(np.stack(vectors), dtype=torch.bfloat16)


class LayerMixtureEditor(nn.Module):
    def __init__(self, basis_vectors: torch.Tensor, strength: float):
        super().__init__()
        self.register_buffer("basis_vectors", basis_vectors)
        init = torch.zeros(basis_vectors.shape[0], dtype=torch.bfloat16)
        init[0] = 6
        self.a = nn.Parameter(init)
        self.strength = float(strength)

    def vector(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        weights = self.strength * F.softmax(self.a.float(), dim=0).to(device=device, dtype=dtype)
        vectors = self.basis_vectors.to(device=device, dtype=dtype)
        return (weights.unsqueeze(1) * vectors).sum(dim=0)


def load_trained_editor(vector_dir: Path) -> dict[str, Any]:
    layers = torch.load(vector_dir / "activation_editor.pth", map_location="cpu")
    return layers


def combined_vectors_from_checkpoint(vector_dir: Path, edit_probe_dir: Path, auxiliary_dir: Path, auxiliary_count: int) -> dict[int, np.ndarray]:
    state = torch.load(vector_dir / "activation_editor.pth", map_location="cpu")
    layers = state["layers"]
    strength = float(state["strength"])
    output: dict[int, np.ndarray] = {}
    for idx, layer in enumerate(layers):
        basis = load_basis_vectors(layer, edit_probe_dir, auxiliary_dir, auxiliary_count).float()
        a = state[f"editors.{idx}.a"].float()
        weights = strength * F.softmax(a, dim=0)
        output[int(layer)] = (weights.unsqueeze(1) * basis).sum(dim=0).numpy()
    return output
