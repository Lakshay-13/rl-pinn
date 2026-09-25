"""Sequential MADAC strategy for TorchHoloEnv."""

from __future__ import annotations

import copy
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import (
    HyperparamActionSpec,
    TransitionBuffer,
    build_mlp,
    flatten_state,
    legacy_beta_pairs,
    legacy_lr_values,
    legacy_scale_values,
    normalize_value,
    soft_update,
)

logger = logging.getLogger(__name__)

try:  # pragma: no cover - fallback only used outside the repo
    from core.device import get_device
except Exception:  # pragma: no cover

    def get_device() -> torch.device:
        return torch.device("cpu")


DEFAULT_STEP_SIZES = [2, len(legacy_lr_values()), len(legacy_beta_pairs())] + [len(legacy_scale_values())] * 7
DEFAULT_STEP_NAMES = ["optimizer", "lr", "betas"] + [f"scale_{idx}" for idx in range(7)]


def _to_state_tensor(state: Any, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(flatten_state(state), dtype=torch.float32, device=device).view(1, -1)


def _quantize_index(value: float, choices: np.ndarray) -> int:
    return int(np.abs(choices - float(value)).argmin())


def _trace_to_param_vector(trace: Sequence[int], active_steps: int, action_spec: HyperparamActionSpec) -> np.ndarray:
    params = np.zeros(action_spec.param_dim, dtype=np.float32)
    lr_values = legacy_lr_values()
    beta_pairs = legacy_beta_pairs()
    scale_values = legacy_scale_values()

    params[0] = normalize_value(1e-3, action_spec.lr_bounds)
    params[1] = normalize_value(0.9, action_spec.beta_bounds)
    params[2] = normalize_value(0.999, action_spec.beta_bounds)
    params[3:] = 1.0

    if active_steps > 1 and len(trace) > 1:
        lr_value = lr_values[int(trace[1]) % len(lr_values)]
        params[0] = normalize_value(lr_value, action_spec.lr_bounds)
    if active_steps > 2 and len(trace) > 2:
        beta_pair = beta_pairs[int(trace[2]) % len(beta_pairs)]
        params[1] = normalize_value(float(beta_pair[0]), action_spec.beta_bounds)
        params[2] = normalize_value(float(beta_pair[1]), action_spec.beta_bounds)
    for idx in range(3, min(active_steps, len(trace))):
        params[idx] = scale_values[int(trace[idx]) % len(scale_values)]
    return params


def _trace_from_action(action: Any, action_spec: HyperparamActionSpec, active_steps: int) -> np.ndarray:
    if isinstance(action, dict):
        action_id, params = action_spec.encode(action)
        params = np.asarray(params, dtype=np.float32)
        trace = np.zeros(active_steps, dtype=np.int64)
        trace[0] = int(action_id)
        if active_steps > 1:
            lr_values = legacy_lr_values()
            trace[1] = _quantize_index(params[0] * (action_spec.lr_bounds[1] - action_spec.lr_bounds[0]) + action_spec.lr_bounds[0], lr_values)
        if active_steps > 2:
            beta_pairs = legacy_beta_pairs()
            beta_1 = params[1] * (action_spec.beta_bounds[1] - action_spec.beta_bounds[0]) + action_spec.beta_bounds[0]
            beta_2 = params[2] * (action_spec.beta_bounds[1] - action_spec.beta_bounds[0]) + action_spec.beta_bounds[0]
            dists = np.abs(beta_pairs[:, 0] - beta_1) + np.abs(beta_pairs[:, 1] - beta_2)
            trace[2] = int(np.argmin(dists))
        for idx in range(3, active_steps):
            scale_value = params[idx] if idx < len(params) else 1.0
            trace[idx] = _quantize_index(scale_value, legacy_scale_values())
        return trace

    if isinstance(action, np.ndarray):
        arr = np.asarray(action)
        if arr.ndim > 1:
            arr = arr.reshape(-1)
        if arr.size >= active_steps:
            return np.asarray(arr[:active_steps], dtype=np.int64)
        if arr.size > 0:
            padded = np.zeros(active_steps, dtype=np.int64)
            padded[: arr.size] = np.asarray(arr, dtype=np.int64).reshape(-1)
            return padded

    if isinstance(action, (tuple, list)):
        if len(action) == 3 and isinstance(action[2], dict):
            action_id = int(action[0])
            params = np.asarray(action[1], dtype=np.float32)
            return _trace_from_action(action_spec.decode(action_id, params), action_spec, active_steps)
        if len(action) == 2 and np.isscalar(action[0]):
            action_id = int(action[0])
            params = np.asarray(action[1], dtype=np.float32)
            return _trace_from_action(action_spec.decode(action_id, params), action_spec, active_steps)
        arr = np.asarray(action)
        if arr.ndim == 1 and arr.size >= active_steps:
            return arr[:active_steps].astype(np.int64)

    if np.isscalar(action):
        trace = np.zeros(active_steps, dtype=np.int64)
        trace[0] = int(action)
        return trace

    raise ValueError(f"Unsupported action record: {type(action)!r}")


class SequentialPolicyNetwork(nn.Module):
    """Encodes state and prefix choices for sequential greedy decoding."""

    def __init__(
        self,
        state_dim: int,
        choice_sizes: Sequence[int],
        hidden_dim: int = 128,
        embed_dim: int = 64,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.choice_sizes = [int(size) for size in choice_sizes]
        self.hidden_dim = int(hidden_dim)
        self.embed_dim = int(embed_dim)

        self.state_encoder = build_mlp(self.state_dim, [hidden_dim * 2], hidden_dim)
        self.choice_embeddings = nn.ModuleList([nn.Embedding(size, embed_dim) for size in self.choice_sizes])
        self.prefix_cell = nn.GRUCell(embed_dim, hidden_dim)
        self.step_heads = nn.ModuleList(
            [build_mlp(hidden_dim * 2, [hidden_dim], size) for size in self.choice_sizes]
        )

    def encode_state(self, state: torch.Tensor) -> torch.Tensor:
        if state.dim() > 2:
            state = state.view(state.size(0), -1)
        return self.state_encoder(state)

    def initial_hidden(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype)

    def step_logits(self, state_feat: torch.Tensor, hidden: torch.Tensor, step_idx: int) -> torch.Tensor:
        step_input = torch.cat([state_feat, hidden], dim=-1)
        return self.step_heads[step_idx](step_input)

    def advance_hidden(self, hidden: torch.Tensor, choice: torch.Tensor, step_idx: int) -> torch.Tensor:
        embed = self.choice_embeddings[step_idx](choice.long())
        return self.prefix_cell(embed, hidden)


class GlobalValueNetwork(nn.Module):
    """Global value head over state and prefix context."""

    def __init__(self, state_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.hidden_dim = int(hidden_dim)
        self.value_net = build_mlp(self.state_dim + hidden_dim, [hidden_dim * 2, hidden_dim], 1)

    def forward(self, state_feat: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        return self.value_net(torch.cat([state_feat, hidden], dim=-1)).squeeze(-1)


class SequentialAdvantageDecomposition(nn.Module):
    """Factorized Q = V + A representation for sequential action selection."""

    def __init__(
        self,
        state_dim: int,
        choice_sizes: Sequence[int],
        hidden_dim: int = 128,
        embed_dim: int = 64,
    ) -> None:
        super().__init__()
        self.policy = SequentialPolicyNetwork(state_dim, choice_sizes, hidden_dim=hidden_dim, embed_dim=embed_dim)
        self.value = GlobalValueNetwork(hidden_dim, hidden_dim=hidden_dim)
        self.choice_sizes = list(choice_sizes)

    def encode_state(self, state: torch.Tensor) -> torch.Tensor:
        return self.policy.encode_state(state)

    def initial_hidden(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        return self.policy.initial_hidden(batch_size, device, dtype=dtype)

    def step_q_values(self, state_feat: torch.Tensor, hidden: torch.Tensor, step_idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.policy.step_logits(state_feat, hidden, step_idx)
        value = self.value(state_feat, hidden)
        advantage = logits - logits.mean(dim=-1, keepdim=True)
        q_values = value.unsqueeze(-1) + advantage
        return logits, q_values, value

    def advance_hidden(self, hidden: torch.Tensor, choice: torch.Tensor, step_idx: int) -> torch.Tensor:
        return self.policy.advance_hidden(hidden, choice, step_idx)


class SequentialActionValue(SequentialAdvantageDecomposition):
    """Compatibility alias for older imports."""


class SeqMADACAgent:
    """Sequential policy/value agent that decodes the env action one choice at a time."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        num_steps: int = 10,
        device: Optional[torch.device] = None,
        *,
        lr: float = 1e-3,
        gamma: float = 0.99,
        n_steps: int = 1,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.85,
        epsilon_min: float = 0.5,
        batch_size: int = 32,
        embed_dim: int = 64,
        tau: float = 0.01,
        buffer_capacity: int = 50000,
    ) -> None:
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_steps = int(min(num_steps, len(DEFAULT_STEP_SIZES)))
        self.active_choice_sizes = DEFAULT_STEP_SIZES[: self.num_steps]
        self.active_step_names = DEFAULT_STEP_NAMES[: self.num_steps]
        self.device = torch.device(device) if device is not None else get_device()
        self.gamma = float(gamma)
        self.n_steps = int(max(1, n_steps))
        self.epsilon = float(epsilon)
        self.epsilon_decay = float(epsilon_decay)
        self.epsilon_min = float(epsilon_min)
        self.batch_size = int(batch_size)
        self.tau = float(tau)
        self.action_spec = HyperparamActionSpec()
        self.replay = TransitionBuffer(capacity=buffer_capacity)

        self.model = SequentialAdvantageDecomposition(
            self.state_dim,
            self.active_choice_sizes,
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
        ).float().to(self.device)
        self.target_model = copy.deepcopy(self.model).float().to(self.device)
        self.target_model.eval()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()
        self._best_weights: Optional[Dict[str, torch.Tensor]] = None
        self._best_loss: float = float("inf")
        self._update_target_counter: int = 0
        self._target_update_interval: int = 30

    def _trace_from_action(self, action: Any) -> np.ndarray:
        return _trace_from_action(action, self.action_spec, self.num_steps)

    def _trace_to_params(self, trace: Sequence[int]) -> Tuple[int, np.ndarray, Dict[str, Any]]:
        params = _trace_to_param_vector(trace, self.num_steps, self.action_spec)
        action_id = int(trace[0])
        decoded = self.action_spec.decode(action_id, params)
        return action_id, params, decoded

    def decode_action(self, action: Any) -> Dict[str, Any]:
        if isinstance(action, dict):
            return action
        if isinstance(action, (tuple, list)):
            if len(action) == 3 and isinstance(action[2], dict):
                return action[2]
            if len(action) == 2 and np.isscalar(action[0]):
                return self.action_spec.decode(int(action[0]), action[1])
        if np.isscalar(action):
            return self.action_spec.decode(int(action), np.zeros(self.action_spec.param_dim, dtype=np.float32))
        raise ValueError(f"Cannot decode action of type {type(action)!r}")

    def _greedy_trace(self, state_feat: torch.Tensor, eval_mode: bool = False) -> np.ndarray:
        hidden = self.model.initial_hidden(state_feat.size(0), state_feat.device, dtype=state_feat.dtype)
        trace: List[int] = []
        for step_idx, choice_size in enumerate(self.active_choice_sizes):
            logits, q_values, _ = self.model.step_q_values(state_feat, hidden, step_idx)
            if (not eval_mode) and random.random() < self.epsilon:
                choice = int(torch.randint(0, choice_size, (1,), device=state_feat.device).item())
            else:
                choice = int(torch.argmax(q_values, dim=-1).item())
            trace.append(choice)
            hidden = self.model.advance_hidden(hidden, torch.tensor([choice], device=state_feat.device), step_idx)
        return np.asarray(trace, dtype=np.int64)

    def select_action(self, state: Any, eval_mode: bool = False) -> Tuple[int, np.ndarray, Dict[str, Any]]:
        state_t = _to_state_tensor(state, self.device)
        state_feat = self.model.encode_state(state_t)
        trace = self._greedy_trace(state_feat, eval_mode=eval_mode)
        action_id, params, decoded = self._trace_to_params(trace)
        self.last_trace = trace
        return action_id, params, decoded

    def act(self, state: Any, eval_mode: bool = False) -> Tuple[int, np.ndarray, Dict[str, Any]]:
        return self.select_action(state, eval_mode=eval_mode)

    def _trace_from_record(self, action: Any) -> np.ndarray:
        try:
            return self._trace_from_action(action)
        except Exception:
            if hasattr(self, "last_trace") and self.last_trace is not None:
                return np.asarray(self.last_trace, dtype=np.int64)
            raise

    def update(self, state: Any, action: Any, reward: float, next_state: Any, done: bool) -> None:
        trace = self._trace_from_record(action)
        self.replay.push(flatten_state(state), trace, float(reward), flatten_state(next_state), bool(done))
        if len(self.replay) >= 2:
            self.train_step(batch_size=min(self.batch_size, len(self.replay)))

    def remember(self, state: Any, action: Any, reward: float, next_state: Any, done: bool) -> None:
        self.update(state, action, reward, next_state, done)

    def train_step(self, batch_size: int = 32) -> Optional[float]:
        if len(self.replay) < batch_size:
            return None

        states, traces, rewards, next_states, dones = self.replay.sample(batch_size)
        states_t = torch.as_tensor(np.stack([flatten_state(s) for s in states]), dtype=torch.float32, device=self.device)
        next_states_t = torch.as_tensor(np.stack([flatten_state(s) for s in next_states]), dtype=torch.float32, device=self.device)
        normalized_traces = [self._trace_from_action(t) for t in traces]
        traces_t = torch.as_tensor(np.stack(normalized_traces), dtype=torch.long, device=self.device)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).view(-1)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device).view(-1)

        state_feat = self.model.encode_state(states_t).float()
        next_state_feat = self.target_model.encode_state(next_states_t).float()
        bootstrap = rewards_t + ((self.gamma ** self.n_steps) * (1.0 - dones_t) * self.target_model.value(
            next_state_feat,
            self.target_model.initial_hidden(
                next_state_feat.size(0),
                next_state_feat.device,
                dtype=next_state_feat.dtype,
            ),
        ))

        hidden = self.model.initial_hidden(state_feat.size(0), self.device, dtype=state_feat.dtype)
        policy_loss = torch.zeros((), device=self.device)
        value_loss = torch.zeros((), device=self.device)
        q_loss = torch.zeros((), device=self.device)

        for step_idx, choice_size in enumerate(self.active_choice_sizes):
            logits, q_values, value_pred = self.model.step_q_values(state_feat, hidden, step_idx)
            chosen = traces_t[:, step_idx].clamp(0, choice_size - 1)
            policy_loss = policy_loss + F.cross_entropy(logits, chosen)
            q_selected = q_values.gather(1, chosen.unsqueeze(1)).squeeze(1)
            q_loss = q_loss + self.loss_fn(q_selected, bootstrap)
            hidden = self.model.advance_hidden(hidden, chosen, step_idx)
            value_loss = value_loss + self.loss_fn(value_pred, bootstrap)

        total_loss = q_loss + 0.5 * policy_loss + 0.5 * value_loss
        loss_value = float(total_loss.detach().cpu().item())

        # Track best weights for restoration after training
        if loss_value < self._best_loss:
            self._best_loss = loss_value
            self._best_weights = {k: v.clone() for k, v in self.model.state_dict().items()}

        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
        self.optimizer.step()

        # Update target network periodically instead of every step
        self._update_target_counter += 1
        if self._update_target_counter >= self._target_update_interval:
            self._update_target_counter = 0
            self.update_target_network(self.tau)

        return loss_value

    def update_target_network(self, tau: float = 0.01) -> None:
        soft_update(self.target_model, self.model, tau)

    def update_target_net(self, tau: float = 0.01) -> None:
        self.update_target_network(tau=tau)

    def restore_best_weights(self) -> None:
        """Restore the network weights from the best loss point during training."""
        if self._best_weights is not None:
            self.model.load_state_dict(self._best_weights)

    def reset_training_state(self) -> None:
        """Reset best weights tracking for a new training cycle."""
        self._best_weights = None
        self._best_loss = float("inf")
        self._update_target_counter = 0

    def save(self, filepath: str, include_memory: bool = False) -> None:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": self.model.state_dict(),
                "target_model": self.target_model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "hidden_dim": self.hidden_dim,
                "num_steps": self.num_steps,
                "gamma": self.gamma,
                "n_steps": self.n_steps,
                "batch_size": self.batch_size,
                "best_weights": self._best_weights,
                "best_loss": self._best_loss,
            },
            filepath,
        )

    def load(self, filepath: str, load_memory: bool = False) -> None:
        state = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(state["model"])
        self.target_model.load_state_dict(state["target_model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.epsilon = float(state["epsilon"])
        self.gamma = float(state["gamma"])
        self.n_steps = int(state.get("n_steps", self.n_steps))
        self.batch_size = int(state["batch_size"])
        self._best_weights = state.get("best_weights")
        self._best_loss = float(state.get("best_loss", float("inf")))
        self.target_model.eval()

    def to_device(self, device: torch.device) -> None:
        self.device = torch.device(device)
        self.model.to(self.device)
        self.target_model.to(self.device)
        for state in self.optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(self.device)


def test_seq_madac() -> None:
    state_dim = 7 * 16 * 41
    agent = SeqMADACAgent(state_dim, 10, num_steps=10, device=torch.device("cpu"), batch_size=1)
    state = np.random.rand(7, 16, 41).astype(np.float32)
    action = agent.select_action(state)
    agent.update(state, action, 1.0, state, False)
    loss = agent.train_step(batch_size=1)
    assert loss is not None and np.isfinite(loss)
    print("Seq-MADAC smoke test passed")


__all__ = [
    "SequentialPolicyNetwork",
    "SequentialAdvantageDecomposition",
    "SequentialActionValue",
    "GlobalValueNetwork",
    "SeqMADACAgent",
]


if __name__ == "__main__":  # pragma: no cover
    test_seq_madac()
