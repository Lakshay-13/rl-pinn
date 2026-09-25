from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Iterable, Optional, Sequence, Tuple

import itertools
import numpy as np
import torch


@dataclass(frozen=True)
class HyperparamActionSpec:
    """Shared action-space definition for the TorchHoloEnv strategies.

    The environment accepts a dictionary with:
    - optimizer: "adam" (LBFGS decoding is currently disabled for stability)
    - lr: scalar learning rate
    - betas: Adam beta tuple
    - s_v: 7-element scale vector
    - epochs: integer inner-loop epochs

    The strategies work with a normalized 10D parameter vector:
    [lr, beta1, beta2, s_v[0], ..., s_v[6]]
    """

    num_actions: int = 2
    param_dim: int = 10
    epochs_per_step: int = 50
    lr_bounds: Tuple[float, float] = (1e-9, 1e-2)
    beta_bounds: Tuple[float, float] = (0.8, 0.999)
    scale_bounds: Tuple[float, float] = (0.0, 1.0)

    @property
    def action_names(self) -> Tuple[str, str]:
        # Keep two discrete actions for compatibility with existing checkpoints,
        # but both decode to Adam while LBFGS is disabled.
        return ("adam", "adam")

    def mask(self, action_id: int) -> np.ndarray:
        mask = np.zeros(self.param_dim, dtype=np.float32)
        mask[0] = 1.0
        mask[3:] = 1.0
        if int(action_id) == 0:
            mask[1:3] = 1.0
        return mask

    def sanitize_params(self, action_id: int, params: Any) -> np.ndarray:
        arr = _to_numpy(params).reshape(-1).astype(np.float32)
        if arr.size < self.param_dim:
            arr = np.pad(arr, (0, self.param_dim - arr.size))
        elif arr.size > self.param_dim:
            arr = arr[: self.param_dim]
        # Force non-finite network outputs into valid normalized bounds.
        arr = np.nan_to_num(arr, nan=0.5, posinf=1.0, neginf=0.5)
        arr = np.clip(arr, 0.0, 1.0)
        return arr * self.mask(action_id)

    def sample_params(self, action_id: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        return rng.uniform(0.0, 1.0, size=self.param_dim).astype(np.float32) * self.mask(action_id)

    def decode(self, action_id: int, params: Any, epochs: Optional[int] = None) -> Dict[str, Any]:
        p = self.sanitize_params(action_id, params)
        lr = max(_linear_map(p[0], self.lr_bounds), self.lr_bounds[0])
        scales = np.clip(p[3:10], *self.scale_bounds).astype(np.float32)

        if int(action_id) == 0:
            beta1 = _linear_map(p[1], self.beta_bounds)
            beta2 = _linear_map(p[2], self.beta_bounds)
            beta1, beta2 = sorted((float(beta1), float(beta2)))
            betas = (beta1, beta2)
        else:
            betas = (0.9, 0.999)
        optimizer = "adam"

        return {
            "optimizer": optimizer,
            "lr": float(lr),
            "betas": betas,
            "s_v": scales,
            "epochs": int(self.epochs_per_step if epochs is None else epochs),
        }

    def encode(self, action_dict: Dict[str, Any]) -> Tuple[int, np.ndarray]:
        optimizer = str(action_dict.get("optimizer", "adam")).lower()
        action_id = 0 if optimizer == "adam" else 1
        params = np.zeros(self.param_dim, dtype=np.float32)
        params[0] = _inverse_linear_map(float(action_dict.get("lr", self.lr_bounds[0])), self.lr_bounds)

        betas = action_dict.get("betas", (0.9, 0.999))
        if isinstance(betas, torch.Tensor):
            betas = betas.detach().cpu().numpy()
        betas = np.asarray(betas, dtype=np.float32).reshape(-1)
        if betas.size >= 2:
            params[1] = _inverse_linear_map(float(np.clip(betas[0], *self.beta_bounds)), self.beta_bounds)
            params[2] = _inverse_linear_map(float(np.clip(betas[1], *self.beta_bounds)), self.beta_bounds)

        scales = action_dict.get("s_v", np.ones(7, dtype=np.float32))
        params[3:10] = np.clip(_to_numpy(scales).reshape(-1)[:7], *self.scale_bounds)
        return action_id, self.sanitize_params(action_id, params)

    def random_action(self, rng: Optional[np.random.Generator] = None) -> Tuple[int, np.ndarray, Dict[str, Any]]:
        rng = rng or np.random.default_rng()
        action_id = int(rng.integers(0, self.num_actions))
        params = self.sample_params(action_id, rng)
        return action_id, params, self.decode(action_id, params)


class TransitionBuffer:
    """Simple replay buffer that handles numpy arrays and tensors."""

    def __init__(self, capacity: int = 100000):
        # Use unlimited buffer (no maxlen) - relies on swap for memory
        self.buffer: Deque[Tuple[Any, ...]] = deque()

    def push(self, *transition: Any) -> None:
        self.buffer.append(tuple(transition))

    def clear(self) -> None:
        self.buffer.clear()

    def __len__(self) -> int:
        return len(self.buffer)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        if batch_size > len(self.buffer):
            raise ValueError(f"Cannot sample {batch_size} transitions from buffer of size {len(self.buffer)}")
        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        batch = [self.buffer[idx] for idx in indices]
        columns = list(zip(*batch))
        return tuple(_stack_column(column) for column in columns)

    def save_to_disk(self, filepath: str) -> None:
        import pickle

        with open(filepath, "wb") as handle:
            pickle.dump(list(self.buffer), handle)

    def load_from_disk(self, filepath: str) -> None:
        import pickle
        from pathlib import Path

        if not Path(filepath).exists():
            return
        with open(filepath, "rb") as handle:
            memories = pickle.load(handle)
            self.buffer.extend(memories)


def build_mlp(
    input_dim: int,
    hidden_dims: Iterable[int],
    output_dim: int,
    *,
    final_activation: Optional[torch.nn.Module] = None,
) -> torch.nn.Sequential:
    layers = []
    dims = [input_dim, *hidden_dims, output_dim]
    for in_dim, out_dim in zip(dims[:-1], dims[1:]):
        layers.append(torch.nn.Linear(in_dim, out_dim))
        if out_dim != output_dim:
            layers.append(torch.nn.ReLU())
    if final_activation is not None:
        layers.append(final_activation)
    return torch.nn.Sequential(*layers)


def flatten_state(state: Any) -> np.ndarray:
    return _to_numpy(state).reshape(-1).astype(np.float32)


def batch_tensor(x: Any, device: torch.device, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.as_tensor(x, dtype=dtype, device=device)


def compute_reward(
    new_loss: Any,
    old_loss: Any,
    s_v: Any,
    step_count: int,
    *,
    gamma: float = 0.99,
    eps: float = 1e-10,
) -> float:
    s_v_arr = _to_numpy(s_v).reshape(-1)
    # Continuous penalty: 2.0 * sum(1.0 - s_v)
    scales_penalty = 2.0 * np.sum(1.0 - s_v_arr)
    log_reward = 50.0 * (np.log10(float(old_loss) + eps) - np.log10(float(new_loss) + eps))
    reward = log_reward - scales_penalty
    return float((gamma ** max(step_count - 1, 0)) * reward)


def soft_update(target: torch.nn.Module, source: torch.nn.Module, tau: float) -> None:
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.mul_(1.0 - tau).add_(source_param.data * tau)


def legacy_lr_values() -> np.ndarray:
    """Compatibility grid used by the historical PINN agents."""
    return np.concatenate((np.linspace(1e-4, 9e-4, 9), np.linspace(1e-3, 1e-2, 10))).astype(np.float32)


def legacy_beta_pairs() -> np.ndarray:
    beta_values = np.array([0.8, 0.88, 0.888, 0.9, 0.99, 0.999], dtype=np.float32)
    return np.array(list(itertools.combinations(beta_values, 2)), dtype=np.float32)


def legacy_scale_values(levels: int = 11) -> np.ndarray:
    return np.linspace(0.0, 1.0, levels).astype(np.float32)


def normalize_value(value: float, bounds: Tuple[float, float]) -> float:
    lo, hi = bounds
    if hi <= lo:
        return 0.0
    return float((value - lo) / (hi - lo))


def denormalize_value(value: float, bounds: Tuple[float, float]) -> float:
    lo, hi = bounds
    return float(lo + np.clip(value, 0.0, 1.0) * (hi - lo))


def _linear_map(value: float, bounds: Tuple[float, float]) -> float:
    lo, hi = bounds
    return lo + float(value) * (hi - lo)


def _inverse_linear_map(value: float, bounds: Tuple[float, float]) -> float:
    lo, hi = bounds
    if hi <= lo:
        return 0.0
    return float((value - lo) / (hi - lo))


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, (list, tuple, dict)):
        try:
            return np.asarray(x)
        except Exception:
            return np.array(x, dtype=object)
    try:
        return np.asarray(x)
    except Exception:
        return np.array(x, dtype=object)


def _stack_column(column: Tuple[Any, ...]) -> np.ndarray:
    values = [_to_numpy(item) for item in column]
    try:
        return np.stack(values)
    except ValueError:
        return np.asarray(values)
