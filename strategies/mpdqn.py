"""Parameterised-action MPDQN for TorchHoloEnv."""

from __future__ import annotations

import copy
import itertools
import logging
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import HyperparamActionSpec, TransitionBuffer, build_mlp, flatten_state, soft_update

logger = logging.getLogger(__name__)

try:  # pragma: no cover - fallback only used outside the repo
    from core.device import get_device
except Exception:  # pragma: no cover

    def get_device() -> torch.device:
        return torch.device("cpu")


def _infer_state_dim(state_shape: Sequence[int] | int) -> int:
    if isinstance(state_shape, int):
        return int(state_shape)
    return int(np.prod(tuple(state_shape)))


def _to_state_tensor(state: Any, device: torch.device) -> torch.Tensor:
    flat = flatten_state(state)
    return torch.as_tensor(flat, dtype=torch.float32, device=device).view(1, -1)


def _looks_like_legacy_action(action: Any) -> bool:
    if isinstance(action, (list, tuple, np.ndarray)):
        try:
            arr = np.asarray(action)
        except Exception:
            return False
        return arr.ndim == 1 and arr.size >= 3 and np.issubdtype(arr.dtype, np.integer)
    return False


class MPDQNReplayBuffer(TransitionBuffer):
    """Alias kept for compatibility."""


class MPDQNQNetwork(nn.Module):
    """Online model used by MPDQN.

    The module contains a shared encoder, action-specific parameter heads, and a
    critic that scores masked parameter vectors for each discrete optimizer.
    """

    def __init__(
        self,
        state_shape: Sequence[int] | int,
        num_actions: int,
        param_dim: int,
        *,
        hidden_size: int = 128,
    ) -> None:
        super().__init__()
        self.state_shape = tuple(state_shape) if not isinstance(state_shape, int) else (int(state_shape),)
        self.state_dim = _infer_state_dim(state_shape)
        self.num_actions = int(num_actions)
        self.param_dim = int(param_dim)
        self.hidden_size = int(hidden_size)

        self.encoder = build_mlp(self.state_dim, [hidden_size * 2, hidden_size], hidden_size)
        self.actor_heads = nn.ModuleList(
            [build_mlp(hidden_size, [hidden_size], self.param_dim, final_activation=nn.Sigmoid()) for _ in range(self.num_actions)]
        )
        self.critic = build_mlp(hidden_size + self.num_actions + self.param_dim, [hidden_size, hidden_size], 1)
        self.register_buffer(
            "action_masks",
            torch.as_tensor(
                np.stack([HyperparamActionSpec().mask(action_id) for action_id in range(self.num_actions)]),
                dtype=torch.float32,
            ),
        )

    def encode(self, state: torch.Tensor) -> torch.Tensor:
        if state.dim() > 2:
            state = state.view(state.size(0), -1)
        return self.encoder(state)

    def _masked_params(self, params: torch.Tensor, action_id: int) -> torch.Tensor:
        mask = self.action_masks[action_id].to(params.device)
        return params * mask

    def actor_params(self, encoded_state: torch.Tensor, action_id: int) -> torch.Tensor:
        params = self.actor_heads[action_id](encoded_state)
        return self._masked_params(params, action_id)

    def q_value_from_encoded(self, encoded_state: torch.Tensor, action_id: int, params: torch.Tensor) -> torch.Tensor:
        if encoded_state.dim() == 1:
            encoded_state = encoded_state.unsqueeze(0)
        if params.dim() == 1:
            params = params.unsqueeze(0)
        action_idx = torch.full((encoded_state.size(0),), int(action_id), device=encoded_state.device, dtype=torch.long)
        action_one_hot = F.one_hot(action_idx, num_classes=self.num_actions).to(encoded_state.dtype)
        masked_params = self._masked_params(params, action_id)
        critic_input = torch.cat([encoded_state, action_one_hot, masked_params], dim=-1)
        return self.critic(critic_input).squeeze(-1)

    def q_value_batch(
        self, encoded_state: torch.Tensor, action_ids: torch.Tensor, params: torch.Tensor
    ) -> torch.Tensor:
        if encoded_state.dim() == 1:
            encoded_state = encoded_state.unsqueeze(0)
        if params.dim() == 1:
            params = params.unsqueeze(0)
        action_ids = action_ids.view(-1).long()
        action_one_hot = F.one_hot(action_ids, num_classes=self.num_actions).to(encoded_state.dtype)
        masked_params = params * self.action_masks[action_ids].to(params.device)
        critic_input = torch.cat([encoded_state, action_one_hot, masked_params], dim=-1)
        return self.critic(critic_input).squeeze(-1)

    def q_values_for_all_actions(
        self, state: Optional[torch.Tensor] = None, encoded_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        if encoded_state is None:
            if state is None:
                raise ValueError("Either state or encoded_state must be provided")
            encoded_state = self.encode(state)
        q_values: List[torch.Tensor] = []
        params_by_action: List[torch.Tensor] = []
        for action_id in range(self.num_actions):
            params = self.actor_params(encoded_state, action_id)
            q_values.append(self.q_value_from_encoded(encoded_state, action_id, params))
            params_by_action.append(params)
        return torch.stack(q_values, dim=-1), params_by_action

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        return self.q_values_for_all_actions(state=state)


class MPDQNAgent:
    """A parameterised-action MPDQN for the TorchHoloEnv action contract."""

    def __init__(
        self,
        state_shape: Sequence[int] | int,
        action_dims: Optional[Sequence[int]] = None,
        device: Optional[torch.device] = None,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.85,
        epsilon_min: float = 0.5,
        n_steps: int = 1,
        batch_size: int = 64,
        *,
        hidden_size: int = 128,
        tau: float = 0.01,
        buffer_capacity: int = 50000,
    ) -> None:
        self.device = torch.device(device) if device is not None else get_device()
        self.state_shape = tuple(state_shape) if not isinstance(state_shape, int) else (int(state_shape),)
        self.state_dim = _infer_state_dim(state_shape)
        self.legacy_action_dims = list(action_dims) if action_dims is not None else [2, HyperparamActionSpec().param_dim]
        self.action_spec = HyperparamActionSpec()
        self.num_actions = int(self.legacy_action_dims[2]) if len(self.legacy_action_dims) >= 3 else self.action_spec.num_actions
        self.num_actions = max(2, self.num_actions)
        self.param_dim = self.action_spec.param_dim
        self.hidden_size = int(hidden_size)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.epsilon_decay = float(epsilon_decay)
        self.epsilon_min = float(epsilon_min)
        self.n_steps = int(n_steps)
        self.batch_size = int(batch_size)
        self.tau = float(tau)

        self.q_net = MPDQNQNetwork(self.state_shape, self.num_actions, self.param_dim, hidden_size=hidden_size).float().to(self.device)
        self.target_net = copy.deepcopy(self.q_net).float().to(self.device)
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()
        # Unlimited buffer (capacity param ignored for consistency)
        self.memory = MPDQNReplayBuffer(capacity=100000)
        self.rng = np.random.default_rng()
        self._best_weights: Optional[Dict[str, torch.Tensor]] = None
        self._best_loss: float = float("inf")
        self._update_target_counter: int = 0
        self._target_update_interval: int = 30

    def _parse_action_record(self, action: Any) -> Tuple[int, np.ndarray]:
        if isinstance(action, dict):
            action_id, params = self.action_spec.encode(action)
            return action_id, params

        if isinstance(action, np.ndarray):
            arr = np.asarray(action, dtype=object)
            if arr.ndim == 1 and arr.size == 3 and isinstance(arr[2], dict):
                action_id = int(arr[0])
                params = self.action_spec.sanitize_params(action_id, arr[1])
                return action_id, params
            if arr.ndim == 1 and arr.size == 2 and np.isscalar(arr[0]):
                action_id = int(arr[0])
                params = self.action_spec.sanitize_params(action_id, arr[1])
                return action_id, params
            if _looks_like_legacy_action(arr):
                return self._legacy_indices_to_action(arr)

        if isinstance(action, (tuple, list)):
            if len(action) == 3 and isinstance(action[2], dict):
                action_id = int(action[0])
                params = self.action_spec.sanitize_params(action_id, action[1])
                return action_id, params
            if len(action) == 2 and np.isscalar(action[0]):
                action_id = int(action[0])
                params = self.action_spec.sanitize_params(action_id, action[1])
                return action_id, params
            if _looks_like_legacy_action(action):
                return self._legacy_indices_to_action(action)

        arr = np.asarray(action)
        if arr.ndim == 0:
            action_id = int(arr.item())
            return action_id, self.action_spec.sample_params(action_id, self.rng)
        if _looks_like_legacy_action(arr):
            return self._legacy_indices_to_action(arr)
        raise ValueError(f"Unsupported action record: {type(action)!r}")

    def _legacy_indices_to_action(self, indices: Sequence[int]) -> Tuple[int, np.ndarray]:
        indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        lr_values = np.concatenate((np.linspace(1e-4, 9e-4, 9), np.linspace(1e-3, 1e-2, 10))).astype(np.float32)
        beta_pairs = np.array(
            list(itertools.combinations([0.8, 0.88, 0.888, 0.9, 0.99, 0.999], 2)), dtype=np.float32
        )
        scale_values = np.linspace(0.0, 1.0, 11).astype(np.float32)
        params = np.zeros(self.param_dim, dtype=np.float32)
        if indices.size >= 1:
            params[0] = (lr_values[int(indices[0]) % len(lr_values)] - self.action_spec.lr_bounds[0]) / (
                self.action_spec.lr_bounds[1] - self.action_spec.lr_bounds[0]
            )
        if indices.size >= 2:
            beta_pair = beta_pairs[int(indices[1]) % len(beta_pairs)]
            params[1] = (float(beta_pair[0]) - self.action_spec.beta_bounds[0]) / (
                self.action_spec.beta_bounds[1] - self.action_spec.beta_bounds[0]
            )
            params[2] = (float(beta_pair[1]) - self.action_spec.beta_bounds[0]) / (
                self.action_spec.beta_bounds[1] - self.action_spec.beta_bounds[0]
            )
        if indices.size >= 4:
            scale_indices = indices[3 : 3 + 7]
            scaled = np.ones(7, dtype=np.float32)
            scaled[: min(scale_indices.size, 7)] = scale_values[scale_indices[:7] % len(scale_values)]
            params[3:10] = scaled
        action_id = 0 if int(indices[2]) == 1 else 1 if indices.size >= 3 else 0
        return action_id, self.action_spec.sanitize_params(action_id, params)

    def decode_action(self, action: Any) -> Dict[str, Any]:
        if isinstance(action, dict):
            return action
        if isinstance(action, (tuple, list)):
            if len(action) == 3 and isinstance(action[2], dict):
                return action[2]
            if len(action) == 2 and np.isscalar(action[0]):
                return self.action_spec.decode(int(action[0]), action[1])
            if _looks_like_legacy_action(action):
                action_id, params = self._legacy_indices_to_action(action)
                return self.action_spec.decode(action_id, params)
        if isinstance(action, np.ndarray) and _looks_like_legacy_action(action):
            action_id, params = self._legacy_indices_to_action(action)
            return self.action_spec.decode(action_id, params)
        if np.isscalar(action):
            return self.action_spec.decode(int(action), np.zeros(self.param_dim, dtype=np.float32))
        raise ValueError(f"Cannot decode action of type {type(action)!r}")

    def select_action(self, state: Any, eval_mode: bool = False) -> Tuple[int, np.ndarray, Dict[str, Any]]:
        state_t = _to_state_tensor(state, self.device)
        if (not eval_mode) and random.random() < self.epsilon:
            action_id, params, decoded = self.action_spec.random_action(self.rng)
            return action_id, params, decoded

        with torch.no_grad():
            encoded = self.q_net.encode(state_t)
            q_values, params_by_action = self.q_net.q_values_for_all_actions(encoded_state=encoded)
            action_id = int(torch.argmax(q_values, dim=-1).item())
            params = params_by_action[action_id].detach().cpu().numpy().reshape(-1)
            params = self.action_spec.sanitize_params(action_id, params)
            decoded = self.action_spec.decode(action_id, params)
            return action_id, params, decoded

    def act(self, state: Any, eval_mode: bool = False) -> Tuple[int, np.ndarray, Dict[str, Any]]:
        return self.select_action(state, eval_mode=eval_mode)

    def remember(self, state: Any, action: Any, reward: float, next_state: Any, done: bool, multi_step_reward: Any = None) -> None:
        self.memory.push(state, action, reward, next_state, done, multi_step_reward)

    def _prepare_action_batch(self, actions: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        action_ids: List[int] = []
        params: List[np.ndarray] = []
        for action in actions:
            action_id, param = self._parse_action_record(action)
            action_ids.append(action_id)
            params.append(param)
        action_ids_t = torch.as_tensor(action_ids, dtype=torch.long, device=self.device)
        params_t = torch.as_tensor(np.stack(params), dtype=torch.float32, device=self.device)
        return action_ids_t, params_t

    def train_step(self, batch_size: Optional[int] = None) -> Optional[float]:
        batch_size = int(batch_size or self.batch_size)
        if len(self.memory) < batch_size:
            return None

        sample = self.memory.sample(batch_size)
        if len(sample) == 5:
            states, actions, rewards, next_states, dones = sample
            multi_step_rewards = None
        else:
            states, actions, rewards, next_states, dones, multi_step_rewards = sample

        states_t = torch.as_tensor(np.stack([flatten_state(s) for s in states]), dtype=torch.float32, device=self.device)
        next_states_t = torch.as_tensor(np.stack([flatten_state(s) for s in next_states]), dtype=torch.float32, device=self.device)
        action_ids_t, params_t = self._prepare_action_batch(actions)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).view(-1)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device).view(-1)

        if multi_step_rewards is not None:
            try:
                multi_step_arr = np.asarray(multi_step_rewards, dtype=np.float32)
                if multi_step_arr.ndim >= 1:
                    rewards_t = torch.as_tensor(multi_step_arr[..., 0], dtype=torch.float32, device=self.device).view(-1)
            except Exception:
                pass

        encoded = self.q_net.encode(states_t)
        q_pred = self.q_net.q_value_batch(encoded, action_ids_t, params_t)

        with torch.no_grad():
            next_encoded = self.target_net.encode(next_states_t)
            next_q_candidates = []
            for action_id in range(self.num_actions):
                next_params = self.target_net.actor_params(next_encoded, action_id)
                next_q_candidates.append(self.target_net.q_value_from_encoded(next_encoded, action_id, next_params))
            next_q = torch.stack(next_q_candidates, dim=-1).max(dim=-1).values
            target_q = rewards_t + (self.gamma**self.n_steps) * (1.0 - dones_t) * next_q

        critic_loss = self.loss_fn(q_pred, target_q)

        current_q_candidates = []
        for action_id in range(self.num_actions):
            actor_params = self.q_net.actor_params(encoded, action_id)
            current_q_candidates.append(self.q_net.q_value_from_encoded(encoded, action_id, actor_params))
        current_q = torch.stack(current_q_candidates, dim=-1)
        actor_loss = -current_q.mean()

        loss = critic_loss + 0.05 * actor_loss
        if not torch.isfinite(loss):
            return 0.0
        loss = torch.nan_to_num(loss, nan=0.0, posinf=1e3, neginf=-1e3)
        loss_value = float(loss.detach().cpu().item())

        # Track best weights for restoration after training
        if loss_value < self._best_loss:
            self._best_loss = loss_value
            self._best_weights = {k: v.clone() for k, v in self.q_net.state_dict().items()}

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 5.0)
        self.optimizer.step()

        # Update target network periodically instead of every step
        self._update_target_counter += 1
        if self._update_target_counter >= self._target_update_interval:
            self._update_target_counter = 0
            self.update_target_network(self.tau)

        return loss_value

    def update_epsilon(self) -> None:
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def update_target_network(self, tau: float = 0.01) -> None:
        soft_update(self.target_net, self.q_net, tau)

    def update_target_net(self, tau: float = 0.01) -> None:
        self.update_target_network(tau=tau)

    def restore_best_weights(self) -> None:
        """Restore the network weights from the best loss point during training."""
        if self._best_weights is not None:
            self.q_net.load_state_dict(self._best_weights)

    def reset_training_state(self) -> None:
        """Reset best weights tracking for a new training cycle."""
        self._best_weights = None
        self._best_loss = float("inf")
        self._update_target_counter = 0

    def save(self, filepath: str, include_memory: bool = False) -> None:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        state = {
            "q_net_state_dict": self.q_net.state_dict(),
            "target_net_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "state_shape": self.state_shape,
            "legacy_action_dims": self.legacy_action_dims,
            "hidden_size": self.hidden_size,
            "gamma": self.gamma,
            "epsilon_decay": self.epsilon_decay,
            "epsilon_min": self.epsilon_min,
            "n_steps": self.n_steps,
            "batch_size": self.batch_size,
            "tau": self.tau,
            "best_weights": self._best_weights,
            "best_loss": self._best_loss,
        }
        if include_memory and hasattr(self, "memory"):
            state["memory"] = list(self.memory.buffer)
        torch.save(state, filepath)

    def load(self, filepath: str, load_memory: bool = False) -> None:
        state = torch.load(filepath, map_location=self.device)
        self.q_net.load_state_dict(state["q_net_state_dict"])
        self.target_net.load_state_dict(state["target_net_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        for opt_state in self.optimizer.state.values():
            for key, value in opt_state.items():
                if isinstance(value, torch.Tensor):
                    opt_state[key] = value.to(self.device)
        self.epsilon = float(state["epsilon"])
        self.state_shape = tuple(state["state_shape"])
        self.legacy_action_dims = list(state.get("legacy_action_dims", self.legacy_action_dims))
        self.hidden_size = int(state.get("hidden_size", self.hidden_size))
        self.gamma = float(state["gamma"])
        self.epsilon_decay = float(state["epsilon_decay"])
        self.epsilon_min = float(state["epsilon_min"])
        self.n_steps = int(state["n_steps"])
        self.batch_size = int(state["batch_size"])
        self.tau = float(state.get("tau", self.tau))
        self._best_weights = state.get("best_weights")
        self._best_loss = float(state.get("best_loss", float("inf")))
        if load_memory and "memory" in state and hasattr(self, "memory"):
            self.memory.buffer.clear()
            self.memory.buffer.extend(state["memory"])
        self.target_net.eval()

    def to_device(self, device: torch.device) -> None:
        self.device = torch.device(device)
        self.q_net.to(self.device)
        self.target_net.to(self.device)
        for state in self.optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(self.device)


def test_mpdqn() -> None:
    """Small smoke test for direct execution."""
    state_shape = (7, 16, 41)
    agent = MPDQNAgent(state_shape, [19, 15, 2, 11, 11, 11, 11, 11, 11], hidden_size=64, batch_size=1)
    test_state = np.random.rand(*state_shape).astype(np.float32)
    action = agent.select_action(test_state)
    agent.memory.push(test_state, action, 1.0, test_state, False)
    loss = agent.train_step(batch_size=1)
    assert loss is not None and np.isfinite(loss)
    print("MPDQN smoke test passed")


__all__ = [
    "MPDQNAgent",
    "MPDQNQNetwork",
    "MPDQNReplayBuffer",
]


if __name__ == "__main__":  # pragma: no cover
    test_mpdqn()
