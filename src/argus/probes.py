from __future__ import annotations

import json
import pickle
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.float().cpu().numpy()


def shuffle_xy(x: list[np.ndarray], y: list[int]) -> tuple[list[np.ndarray], list[int]]:
    combined = list(zip(x, y))
    random.shuffle(combined)
    xs, ys = zip(*combined)
    return list(xs), list(ys)


def load_detection_xy(path: Path, layer: int, ratio: float = 0.8):
    dataset = torch.load(path, map_location="cpu")
    clean = [_to_numpy(item["clean_activation"][layer]) for item in dataset]
    poison = [_to_numpy(item["image_inject_activation"][layer]) for item in dataset]
    return _split_binary(clean, poison, ratio)


def load_edit_xy(path: Path, layer: int, ratio: float = 0.8):
    dataset = torch.load(path, map_location="cpu")
    first = [_to_numpy(item["first_activation"][layer]) for item in dataset]
    second = [_to_numpy(item["second_activation"][layer]) for item in dataset]
    return _split_binary(first, second, ratio)


def load_edit_eval_xy(path: Path, layer: int):
    dataset = torch.load(path, map_location="cpu")
    first = [_to_numpy(item["first_activation"][layer]) for item in dataset]
    second = [_to_numpy(item["second_activation"][layer]) for item in dataset]
    x = first + second
    y = [0] * len(first) + [1] * len(second)
    return np.vstack(x), np.asarray(y)


def _split_binary(class0: list[np.ndarray], class1: list[np.ndarray], ratio: float):
    split0 = int(ratio * len(class0))
    split1 = int(ratio * len(class1))
    train_x = class0[:split0] + class1[:split1]
    train_y = [0] * split0 + [1] * split1
    val_x = class0[split0:] + class1[split1:]
    val_y = [0] * (len(class0) - split0) + [1] * (len(class1) - split1)
    train_x, train_y = shuffle_xy(train_x, train_y)
    return np.vstack(train_x), np.asarray(train_y), np.vstack(val_x), np.asarray(val_y)


def train_logistic(train_x: np.ndarray, train_y: np.ndarray) -> LogisticRegression:
    model = LogisticRegression(max_iter=2000)
    model.fit(train_x, train_y)
    return model


class OrthogonalLogisticRegression:
    def __init__(self, basis: np.ndarray, lr: float = 1e-2, epochs: int = 2000, reg_l2: float = 1e-5, seed: int = 42):
        self.basis = np.atleast_2d(basis).astype(float)
        self.lr = lr
        self.epochs = epochs
        self.reg_l2 = reg_l2
        self.seed = seed
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0

    def fit(self, x: np.ndarray, y: np.ndarray):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(self.seed)
        x_t = torch.tensor(x, dtype=torch.double, device=device)
        y_t = torch.tensor(y, dtype=torch.double, device=device).view(-1, 1)
        basis_t = torch.tensor(self.basis, dtype=torch.double, device=device)
        q, _ = torch.linalg.qr(basis_t.T)
        projection = torch.eye(x.shape[1], dtype=torch.double, device=device) - q @ q.T
        v = torch.randn((x.shape[1], 1), dtype=torch.double, device=device) * 0.01
        b = torch.zeros(1, dtype=torch.double, device=device)
        v.requires_grad = True
        b.requires_grad = True
        optimizer = torch.optim.SGD([v, b], lr=self.lr)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            w = projection @ v
            logits = x_t @ w + b
            loss = loss_fn(logits, y_t) + self.reg_l2 * torch.sum(w**2) / 2
            loss.backward()
            optimizer.step()
        self.coef_ = (projection @ v).detach().cpu().numpy().flatten()
        self.intercept_ = float(b.detach().cpu().item())
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        logits = x @ self.coef_ + self.intercept_
        probs = 1 / (1 + np.exp(-logits))
        return np.c_[1 - probs, probs]

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (self.predict_proba(x)[:, 1] >= 0.5).astype(int)

    def score(self, x: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(x) == y))


class ParallelLogisticRegression:
    def __init__(self, direction: np.ndarray, lr: float = 1e-2, epochs: int = 2000, reg_l2: float = 1e-5, seed: int = 42):
        self.direction = np.asarray(direction, dtype=float).flatten()
        self.lr = lr
        self.epochs = epochs
        self.reg_l2 = reg_l2
        self.seed = seed
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self.k_: float = 0.0

    def fit(self, x: np.ndarray, y: np.ndarray):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(self.seed)
        x_t = torch.tensor(x, dtype=torch.double, device=device)
        y_t = torch.tensor(y, dtype=torch.double, device=device).view(-1, 1)
        direction = torch.tensor(self.direction, dtype=torch.double, device=device).view(-1, 1)
        k = torch.zeros(1, dtype=torch.double, device=device, requires_grad=True)
        b = torch.zeros(1, dtype=torch.double, device=device, requires_grad=True)
        optimizer = torch.optim.SGD([k, b], lr=self.lr)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            w = k * direction
            logits = x_t @ w + b
            loss = loss_fn(logits, y_t) + self.reg_l2 * torch.sum(w**2) / 2
            loss.backward()
            optimizer.step()
        self.k_ = float(k.detach().cpu().item())
        self.coef_ = (k * direction).detach().cpu().numpy().flatten()
        self.intercept_ = float(b.detach().cpu().item())
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        logits = x @ self.coef_ + self.intercept_
        probs = 1 / (1 + np.exp(-logits))
        return np.c_[1 - probs, probs]

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (self.predict_proba(x)[:, 1] >= 0.5).astype(int)

    def score(self, x: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(x) == y))


def save_pickle(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def save_metrics(metrics: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def normalized_probe_vector(model: Any) -> np.ndarray:
    coef = np.asarray(model.coef_).squeeze().astype(float)
    norm = np.linalg.norm(coef)
    if norm == 0:
        raise ValueError("Probe vector norm is zero.")
    # Defense direction points from attacker-following back to user-following.
    return -coef / norm
