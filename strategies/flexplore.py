"""FLEXplore strategy for TorchHoloEnv."""

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

from .common import HyperparamActionSpec, TransitionBuffer, build_mlp, flatten_state, soft_update

logger = logging.getLogger(__name__)

try:  # pragma: no cover - fallback only used outside the repo
    from core.device import get_device
except Exception:  # pragma: no cover

    def get_device() -> torch.device:
        return torch.device("cpu")


def _to_state_tensor(state: Any, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(flatten_state(state), dtype=torch.float32, device=device).view(1, -1)


def _action_to_record(action: Any, spec: HyperparamActionSpec) -> Tuple[int, np.ndarray]:
    if isinstance(action, dict):
        return spec.encode(action)
    if isinstance(action, np.ndarray):
        arr = np.asarray(action, dtype=object)
        if arr.ndim == 1 and arr.size == 3 and isinstance(arr[2], dict):
            action_id = int(arr[0])
            return action_id, spec.sanitize_params(action_id, arr[1])
        if arr.ndim == 1 and arr.size == 2 and np.isscalar(arr[0]):
            action_id = int(arr[0])
            return action_id, spec.sanitize_params(action_id, arr[1])
    if isinstance(action, (tuple, list)):
        if len(action) == 3 and isinstance(action[2], dict):
            action_id = int(action[0])
            return action_id, spec.sanitize_params(action_id, action[1])
        if len(action) == 2 and np.isscalar(action[0]):
            action_id = int(action[0])
            return action_id, spec.sanitize_params(action_id, action[1])
    if np.isscalar(action):
        action_id = int(action)
        return action_id, spec.sample_params(action_id)
    arr = np.asarray(action)
    if arr.ndim == 1 and arr.size >= 2:
        action_id = int(arr[0])
        return action_id, spec.sanitize_params(action_id, arr[1:])
    raise ValueError(f"Unsupported action record: {type(action)!r}")


def _pack_action_features(action_ids: torch.Tensor, params: torch.Tensor, spec: HyperparamActionSpec) -> torch.Tensor:
    if action_ids.dim() == 0:
        action_ids = action_ids.view(1)
    if params.dim() == 1:
        params = params.view(1, -1)
    action_one_hot = F.one_hot(action_ids.long(), num_classes=spec.num_actions).to(params.dtype)
    masked_params = params * torch.as_tensor(
        np.stack([spec.mask(int(a)) for a in action_ids.detach().cpu().numpy()]),
        dtype=params.dtype,
        device=params.device,
    )
    return torch.cat([action_one_hot, masked_params], dim=-1)


class DynamicsModel(nn.Module):
    """Predicts a state delta from encoded state and hybrid action features."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.state_encoder = build_mlp(self.state_dim, [hidden_dim * 2], hidden_dim)
        self.action_encoder = build_mlp(self.action_dim, [hidden_dim], hidden_dim)
        self.transition_head = build_mlp(hidden_dim * 2, [hidden_dim, hidden_dim], self.state_dim)

    def encode_state(self, state: torch.Tensor) -> torch.Tensor:
        if state.dim() > 2:
            state = state.view(state.size(0), -1)
        return self.state_encoder(state)

    def forward(self, state: torch.Tensor, action_features: torch.Tensor) -> torch.Tensor:
        encoded_state = self.encode_state(state)
        encoded_action = self.action_encoder(action_features)
        transition_input = torch.cat([encoded_state, encoded_action], dim=-1)
        return self.transition_head(transition_input)


class RewardModel(nn.Module):
    """Predicts reward from state, action, and next-state features."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.state_encoder = build_mlp(self.state_dim, [hidden_dim * 2], hidden_dim)
        self.next_state_encoder = build_mlp(self.state_dim, [hidden_dim], hidden_dim)
        self.action_encoder = build_mlp(self.action_dim, [hidden_dim], hidden_dim)
        self.reward_head = build_mlp(hidden_dim * 3, [hidden_dim], 1)

    def encode_state(self, state: torch.Tensor) -> torch.Tensor:
        if state.dim() > 2:
            state = state.view(state.size(0), -1)
        return self.state_encoder(state)

    def encode_next_state(self, state: torch.Tensor) -> torch.Tensor:
        if state.dim() > 2:
            state = state.view(state.size(0), -1)
        return self.next_state_encoder(state)

    def forward(self, state: torch.Tensor, action_features: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor:
        encoded_state = self.encode_state(state)
        encoded_next_state = self.encode_next_state(next_state)
        encoded_action = self.action_encoder(action_features)
        reward_input = torch.cat([encoded_state, encoded_action, encoded_next_state], dim=-1)
        return self.reward_head(reward_input).squeeze(-1)


class RewardSmoothing:
    """FGSM-style reward smoothing for high-return regions."""

    def __init__(self, threshold: float = 0.0, epsilon: float = 0.05, blend: float = 0.5):
        self.threshold = float(threshold)
        self.epsilon = float(epsilon)
        self.blend = float(blend)

    def smooth(
        self,
        reward_model: RewardModel,
        state: torch.Tensor,
        action_features: torch.Tensor,
        next_state: torch.Tensor,
        reward: torch.Tensor,
    ) -> torch.Tensor:
        if reward.detach().mean().item() <= self.threshold:
            return reward

        with torch.enable_grad():
            action_var = action_features.detach().clone().requires_grad_(True)
            pred_reward = reward_model(state, action_var, next_state)
            grad = torch.autograd.grad(pred_reward.sum(), action_var, retain_graph=False, create_graph=False)[0]
            perturbed = action_features.detach().clone()
            if perturbed.size(-1) > 2:
                perturbed[:, 2:] = torch.clamp(
                    perturbed[:, 2:] + self.epsilon * grad[:, 2:].sign(),
                    0.0,
                    1.0,
                )
            smoothed = reward_model(state, perturbed, next_state).detach()
            return (1.0 - self.blend) * reward + self.blend * smoothed


class MPPIPlanner:
    """Short-horizon CEM planner with MPPI-style scoring."""

    def __init__(
        self,
        dynamics_model: DynamicsModel,
        reward_model: RewardModel,
        action_spec: HyperparamActionSpec,
        horizon: int = 3,
        num_samples: int = 64,
        elite_frac: float = 0.2,
        iterations: int = 3,
        temperature: float = 1.0,
        exploration_bonus: float = 0.01,
        discount: float = 0.97,
        device: Optional[torch.device] = None,
        use_reward_smoothing: bool = True,
    ) -> None:
        self.dynamics_model = dynamics_model
        self.reward_model = reward_model
        self.action_spec = action_spec
        self.horizon = int(horizon)
        self.num_samples = int(num_samples)
        self.elite_frac = float(elite_frac)
        self.iterations = int(iterations)
        self.temperature = float(temperature)
        self.exploration_bonus = float(exploration_bonus)
        self.discount = float(discount)
        self.device = device or torch.device("cpu")
        self.use_reward_smoothing = bool(use_reward_smoothing)
        self.latent_dim = 1 + self.action_spec.param_dim
        self.reward_smoother = RewardSmoothing() if use_reward_smoothing else None

    def _decode_latent(self, latent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        action_id = (torch.sigmoid(latent[..., :1]) > 0.5).long().squeeze(-1)
        params = torch.sigmoid(latent[..., 1:])
        params = torch.stack(
            [
                torch.as_tensor(self.action_spec.sanitize_params(int(a), p), dtype=latent.dtype, device=latent.device)
                for a, p in zip(action_id.detach().cpu().numpy(), params)
            ],
            dim=0,
        )
        return action_id, params

    def _score_sequence(self, state: torch.Tensor, sequence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[Dict[str, Any]]]:
        batch_size = sequence.size(0)
        current_state = state.expand(batch_size, -1)
        total_score = torch.zeros(batch_size, device=state.device)
        first_action_ids = None
        first_action_params = None
        decoded_sequences: List[List[Dict[str, Any]]] = [[] for _ in range(batch_size)]

        for t in range(self.horizon):
            latent = sequence[:, t, :]
            action_id, params = self._decode_latent(latent)
            action_features = _pack_action_features(action_id, params, self.action_spec)
            next_state = current_state + self.dynamics_model(current_state, action_features)
            reward = self.reward_model(current_state, action_features, next_state)

            if self.reward_smoother is not None:
                reward = self.reward_smoother.smooth(self.reward_model, current_state, action_features, next_state, reward)

            action_prob = torch.sigmoid(latent[:, 0])
            entropy = -(action_prob * torch.log(action_prob.clamp_min(1e-6)) + (1.0 - action_prob) * torch.log((1.0 - action_prob).clamp_min(1e-6)))
            novelty = torch.mean(torch.abs(next_state - current_state), dim=-1)
            total_score = total_score + (self.discount**t) * (reward + self.exploration_bonus * (entropy + novelty))

            if t == 0:
                first_action_ids = action_id
                first_action_params = params
            current_state = next_state
            for batch_idx in range(batch_size):
                decoded_sequences[batch_idx].append(
                    {
                        "action_id": int(action_id[batch_idx].item()),
                        "params": params[batch_idx].detach().cpu().numpy().astype(np.float32),
                    }
                )

        assert first_action_ids is not None and first_action_params is not None
        return total_score, first_action_ids, first_action_params, decoded_sequences

    def plan(self, state: torch.Tensor, generator: Optional[torch.Generator] = None) -> Dict[str, Any]:
        state = state.to(self.device)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if state.dim() > 2:
            state = state.view(state.size(0), -1)

        mean = torch.zeros(self.horizon, self.latent_dim, device=self.device)
        std = torch.ones_like(mean) * 0.65
        elite_count = max(1, int(self.num_samples * self.elite_frac))
        best: Optional[Dict[str, Any]] = None

        for _ in range(self.iterations):
            noise = torch.randn(
                self.num_samples,
                self.horizon,
                self.latent_dim,
                device=self.device,
                generator=generator,
            )
            candidates = mean.unsqueeze(0) + std.unsqueeze(0) * noise
            scores, first_ids, first_params, sequences = self._score_sequence(state, candidates)
            elite_idx = torch.topk(scores, elite_count, dim=0).indices
            elites = candidates[elite_idx]
            mean = elites.mean(dim=0)
            std = elites.std(dim=0, unbiased=False).clamp_min(1e-3)

            best_idx = int(torch.argmax(scores).item())
            best = {
                "action_id": int(first_ids[best_idx].item()),
                "params": first_params[best_idx].detach().cpu().numpy().astype(np.float32),
                "score": float(scores[best_idx].item()),
                "sequence": sequences[best_idx],
            }

        assert best is not None
        return best


class FLEXploreAgent:
    """Model-based strategy for TorchHoloEnv."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        horizon: int = 5,
        device: Optional[torch.device] = None,
        num_samples: int = 64,
        temperature: float = 1.0,
        use_reward_smoothing: bool = True,
        use_exploration_bonus: bool = True,
        *,
        lr: float = 1e-3,
        planning_iters: int = 3,
        elite_frac: float = 0.2,
        discount: float = 0.97,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.85,
        epsilon_min: float = 0.5,
        buffer_capacity: int = 50000,
    ) -> None:
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.horizon = int(horizon)
        self.device = torch.device(device) if device is not None else get_device()
        self.num_samples = int(num_samples)
        self.temperature = float(temperature)
        self.use_reward_smoothing = bool(use_reward_smoothing)
        self.use_exploration_bonus = bool(use_exploration_bonus)
        self.discount = float(discount)
        self.epsilon = float(epsilon)
        self.epsilon_decay = float(epsilon_decay)
        self.epsilon_min = float(epsilon_min)
        self.action_spec = HyperparamActionSpec()
        self.replay = TransitionBuffer(capacity=buffer_capacity)

        self.dynamics_model = DynamicsModel(self.state_dim, self.action_spec.num_actions + self.action_spec.param_dim, hidden_dim).float().to(self.device)
        self.reward_model = RewardModel(self.state_dim, self.action_spec.num_actions + self.action_spec.param_dim, hidden_dim).float().to(self.device)
        self.planner = MPPIPlanner(
            self.dynamics_model,
            self.reward_model,
            self.action_spec,
            horizon=self.horizon,
            num_samples=self.num_samples,
            elite_frac=elite_frac,
            iterations=planning_iters,
            temperature=temperature,
            exploration_bonus=0.01 if use_exploration_bonus else 0.0,
            discount=discount,
            device=self.device,
            use_reward_smoothing=use_reward_smoothing,
        )

        self.optimizer = torch.optim.Adam(
            list(self.dynamics_model.parameters()) + list(self.reward_model.parameters()),
            lr=lr,
        )
        self.loss_fn = nn.SmoothL1Loss()
        self._best_weights: Optional[Dict[str, torch.Tensor]] = None
        self._best_loss: float = float("inf")

    def _encode_action(self, action: Any) -> Tuple[int, np.ndarray]:
        return _action_to_record(action, self.action_spec)

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

    def plan(self, state: Any, *, deterministic: bool = False) -> Dict[str, Any]:
        state_t = _to_state_tensor(state, self.device)
        generator = None
        if deterministic:
            generator_device = self.device.type if self.device.type in {"cpu", "cuda"} else "cpu"
            generator = torch.Generator(device=generator_device)
            generator.manual_seed(0)
        return self.planner.plan(state_t, generator=generator)

    def act(self, state: Any, eval_mode: bool = False) -> Tuple[int, np.ndarray, Dict[str, Any]]:
        if (not eval_mode) and random.random() < self.epsilon:
            action_id, params, decoded = self.action_spec.random_action()
            return action_id, params, decoded

        plan = self.plan(state, deterministic=eval_mode)
        action_id = int(plan["action_id"])
        params = np.asarray(plan["params"], dtype=np.float32)
        decoded = self.action_spec.decode(action_id, params)
        return action_id, params, decoded

    def select_action(self, state: Any, eval_mode: bool = False) -> Tuple[int, np.ndarray, Dict[str, Any]]:
        return self.act(state, eval_mode=eval_mode)

    def update(
        self,
        state: Any,
        action: Any,
        reward: float,
        next_state: Any,
        done: bool,
    ) -> None:
        self.replay.push(state, action, float(reward), next_state, bool(done))
        if len(self.replay) >= 2:
            self.train_step(batch_size=min(32, len(self.replay)))

    def _prepare_action_batch(self, actions: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        action_ids: List[int] = []
        params: List[np.ndarray] = []
        for action in actions:
            action_id, param = _action_to_record(action, self.action_spec)
            action_ids.append(action_id)
            params.append(param)
        return (
            torch.as_tensor(action_ids, dtype=torch.long, device=self.device),
            torch.as_tensor(np.stack(params), dtype=torch.float32, device=self.device),
        )

    def train_step(self, batch_size: int = 32) -> Optional[float]:
        if len(self.replay) < batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay.sample(batch_size)
        states_t = torch.as_tensor(np.stack([flatten_state(s) for s in states]), dtype=torch.float32, device=self.device)
        next_states_t = torch.as_tensor(np.stack([flatten_state(s) for s in next_states]), dtype=torch.float32, device=self.device)
        action_ids_t, params_t = self._prepare_action_batch(actions)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).view(-1)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device).view(-1)

        action_features = _pack_action_features(action_ids_t, params_t, self.action_spec)
        pred_delta = self.dynamics_model(states_t, action_features)
        pred_next = states_t + pred_delta
        dynamics_loss = self.loss_fn(pred_next, next_states_t)

        pred_reward = self.reward_model(states_t, action_features, next_states_t)
        target_reward = rewards_t
        if self.use_reward_smoothing and self.planner.reward_smoother is not None:
            target_reward = self.planner.reward_smoother.smooth(
                self.reward_model,
                states_t,
                action_features,
                next_states_t,
                rewards_t,
            ).detach()
        reward_loss = self.loss_fn(pred_reward, target_reward)

        total_loss = dynamics_loss + reward_loss
        loss_value = float(total_loss.detach().cpu().item())

        # Track best weights for restoration after training
        if loss_value < self._best_loss:
            self._best_loss = loss_value
            self._best_weights = {
                "dynamics_model": {k: v.clone() for k, v in self.dynamics_model.state_dict().items()},
                "reward_model": {k: v.clone() for k, v in self.reward_model.state_dict().items()},
            }

        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.dynamics_model.parameters()) + list(self.reward_model.parameters()),
            5.0,
        )
        self.optimizer.step()
        return loss_value

    def train_models(self, replay_buffer: Any, num_epochs: int = 1) -> Dict[str, float]:
        loss_history = []
        if replay_buffer is not None:
            self.replay = replay_buffer
        for _ in range(max(1, int(num_epochs))):
            if len(self.replay) < 2:
                break
            loss = self.train_step(batch_size=min(32, len(self.replay)))
            if loss is not None:
                loss_history.append(loss)
        return {"loss": float(np.mean(loss_history)) if loss_history else 0.0}

    def save(self, filepath: str, include_memory: bool = False) -> None:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "dynamics_model": self.dynamics_model.state_dict(),
                "reward_model": self.reward_model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "hidden_dim": self.hidden_dim,
                "horizon": self.horizon,
                "best_weights": self._best_weights,
                "best_loss": self._best_loss,
            },
            filepath,
        )

    def load(self, filepath: str, load_memory: bool = False) -> None:
        state = torch.load(filepath, map_location=self.device)
        self.dynamics_model.load_state_dict(state["dynamics_model"])
        self.reward_model.load_state_dict(state["reward_model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.epsilon = float(state["epsilon"])
        self._best_weights = state.get("best_weights")
        self._best_loss = float(state.get("best_loss", float("inf")))

    def to_device(self, device: torch.device) -> None:
        self.device = torch.device(device)
        self.dynamics_model.to(self.device)
        self.reward_model.to(self.device)
        self.planner.device = self.device
        self.planner.dynamics_model = self.dynamics_model
        self.planner.reward_model = self.reward_model
        for state in self.optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(self.device)

    def restore_best_weights(self) -> None:
        """Restore the network weights from the best loss point during training."""
        if self._best_weights is not None:
            self.dynamics_model.load_state_dict(self._best_weights["dynamics_model"])
            self.reward_model.load_state_dict(self._best_weights["reward_model"])

    def reset_training_state(self) -> None:
        """Reset best weights tracking for a new training cycle."""
        self._best_weights = None
        self._best_loss = float("inf")


def test_flexplore() -> None:
    state_dim = 7 * 16 * 41
    agent = FLEXploreAgent(state_dim, 10, horizon=3, device=torch.device("cpu"), num_samples=8, planning_iters=2)
    state = np.random.rand(7, 16, 41).astype(np.float32)
    action = agent.act(state)
    agent.update(state, action, 1.0, state, False)
    plan = agent.plan(state)
    assert "action_id" in plan and "params" in plan and np.isfinite(plan["score"])
    print("FLEXplore smoke test passed")


__all__ = [
    "DynamicsModel",
    "RewardModel",
    "RewardSmoothing",
    "MPPIPlanner",
    "FLEXploreAgent",
]


if __name__ == "__main__":  # pragma: no cover
    test_flexplore()
