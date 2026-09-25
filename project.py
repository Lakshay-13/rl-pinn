from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import pickle
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib import error, request

import numpy as np
import torch
import torch.multiprocessing as mp
from tqdm import tqdm

from agent import PinnAgent
from core.device import get_available_devices, get_cuda_memory_weights
from env_wrapper import TorchHoloEnv
from scan import perform_scan
from strategies.flexplore import FLEXploreAgent
from strategies.mpdqn import MPDQNAgent
from strategies.seq_madac import SeqMADACAgent

ROOT_DIR = Path(__file__).resolve().parent
PROJECT_LOG_ROOT = ROOT_DIR / "logs" / "project"
DEFAULT_STATE_SHAPE = (7, 16, 41)
DEFAULT_STATE_DIM = int(np.prod(DEFAULT_STATE_SHAPE))
DEFAULT_ACTION_DIMS = [19, 15, 2] + [11] * 7
DEFAULT_CYCLES = 15
DEFAULT_TARGET_MEMORIES = 50000
DEFAULT_TRAIN_EPOCHS = 100
DEFAULT_PATIENCE = 10
DEFAULT_MIN_DELTA = 1e-4
DEFAULT_STEPS = 200
DEFAULT_FRONTIER_CYCLES = 5
FRONTIER_TARGET_MEMORY_SCHEDULE = (10000, 20000, 20000, 30000, 30000, 30000, 40000, 40000, 40000, 40000)
FRONTIER_SCALE_BASELINE_SCHEDULE = (0.10, 0.30, 0.50, 0.70, 0.90)
FRONTIER_MEMORY_BLOCK = 10000
DEFAULT_FRONTIER_TARGET_MEMORIES = max(FRONTIER_TARGET_MEMORY_SCHEDULE)
DEFAULT_FRONTIER_EPSILON = 0.15
DEFAULT_FRONTIER_CONTEXT_CHANNELS = 4
DEFAULT_FRONTIER_N_STEP_HORIZON = 10
DEFAULT_FRONTIER_LR_LOG_DELTA = 1.0
DEFAULT_FRONTIER_BETA_DELTA = 0.08
DEFAULT_FRONTIER_SCALE_DELTA = 0.35
DEFAULT_MEM_LIMIT = 0.85
DEFAULT_EPSILON_DECAY = 0.8
DEFAULT_EPSILON_START = 0.99
REPLAY_BUFFER_MAX = 200000
FRONTIER_REPLAY_FILENAME = "replay_buffer.pkl"
SUPPORTED_STRATEGIES = ("default", "mpdqn", "seq-madac", "flexplore")
PROJECT_REWARD_LOG_SCALE = 50.0
PROJECT_SCALE_PENALTY = 800.0
PROJECT_SHAPE_GATE_CYCLE = 3
PROJECT_SHAPE_MIN_CORR = 0.55
PROJECT_SHAPE_MIN_DROP = 0.15
PROJECT_SHAPE_MIN_MIN_INDEX_FRAC = 0.35
PROJECT_SHAPE_MIN_REBOUND = 1e-3
PROJECT_CYCLE_STOP_START = 3
PROJECT_CYCLE_STOP_PATIENCE = 2
PROJECT_CYCLE_STOP_MIN_LOG_DELTA = -1.0
PROJECT_RENDER_DECISIONS = 2000
PROJECT_RENDER_ACTION_EPOCHS = 50
NOTIFYME_ENDPOINT = "https://www.nextgenaischool.in/api/notifyme/send"
PROJECT_NAME = "RL-PINN"

logger = logging.getLogger(__name__)


def _device_label(device: torch.device) -> str:
    if device.type == "cuda":
        index = 0 if device.index is None else int(device.index)
        try:
            name = torch.cuda.get_device_name(index)
        except Exception:
            name = f"CUDA:{index}"
        return f"cuda:{index} ({name})"
    if device.type == "mps":
        return "mps (Apple GPU)"
    return "cpu"


def _normalize_device_name(device: str) -> str:
    value = str(device).strip().lower()
    if value.startswith("cuda"):
        return "cuda"
    if value.startswith("mps"):
        return "mps"
    return "cpu"


def _candidate_device_types() -> List[str]:
    candidates: List[str] = []
    for device in get_available_devices():
        candidate = _normalize_device_name(str(device))
        if candidate not in candidates:
            candidates.append(candidate)
    if "cpu" not in candidates:
        candidates.append("cpu")
    return candidates


def _build_device(device_name: str) -> torch.device:
    if device_name == "cuda":
        return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    if device_name == "mps":
        return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    return torch.device("cpu")


def _project_compute_reward(
    new_loss: Any,
    old_loss: Any,
    s_v: Any,
    step_count: int,
    *,
    gamma: float = 0.99,
    eps: float = 1e-10,
) -> float:
    if isinstance(s_v, torch.Tensor):
        s_v_arr = s_v.detach().cpu().numpy().reshape(-1)
    else:
        s_v_arr = np.asarray(s_v, dtype=np.float32).reshape(-1)
    scales_penalty = PROJECT_SCALE_PENALTY * np.sum(1.0 - s_v_arr)
    log_reward = PROJECT_REWARD_LOG_SCALE * (
        np.log10(float(old_loss) + eps) - np.log10(float(new_loss) + eps)
    )
    reward = log_reward - scales_penalty
    return float((gamma ** max(step_count - 1, 0)) * reward)


def _parse_device_choice(value: Optional[str]) -> Optional[torch.device]:
    if value is None:
        return None
    raw = str(value).strip().lower()
    if not raw:
        return None
    if raw == "auto":
        return None
    if raw in {"cpu", "cuda", "mps"}:
        return _build_device(raw)
    if raw.startswith(("cuda:", "mps")):
        return torch.device(raw)
    raise ValueError(f"Unsupported device selection: {value}")


def _parse_workers_choice(value: Optional[str]) -> Tuple[Optional[int], bool]:
    if value is None:
        return None, False
    raw = str(value).strip().lower()
    if not raw:
        return None, False
    if raw in {"auto", "a"}:
        return None, True
    workers = int(raw)
    if workers <= 0:
        raise ValueError("Workers must be positive")
    return workers, False


def _prompt(message: str, default: Optional[str] = None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    try:
        response = input(f"{message}{suffix}: ").strip()
    except EOFError:
        response = ""
    return response or (default or "")


def _prompt_device() -> torch.device:
    available = get_available_devices()
    print("Available devices:")
    for idx, device in enumerate(available, start=1):
        print(f"  {idx}. {_device_label(device)}")
    while True:
        raw = _prompt("Select device by number or name", default="1").lower()
        if raw.isdigit():
            index = int(raw) - 1
            if 0 <= index < len(available):
                return available[index]
        if raw in {"cpu", "cuda", "mps"}:
            for device in available:
                if _normalize_device_name(str(device)) == raw:
                    return device
        if raw.startswith(("cuda:", "mps")):
            try:
                return torch.device(raw)
            except Exception:
                pass
        print("Invalid device selection. Try again.")


def _prompt_workers() -> Tuple[Optional[int], bool]:
    while True:
        raw = _prompt("Number of workers or 'auto'", default="auto").lower()
        if raw in {"auto", "a"}:
            return None, True
        try:
            workers = int(raw)
        except ValueError:
            print("Please enter a positive integer or 'auto'.")
            continue
        if workers <= 0:
            print("Please enter a positive integer or 'auto'.")
            continue
        return workers, False


def _prompt_strategy() -> str:
    choices = "/".join(SUPPORTED_STRATEGIES)
    while True:
        raw = _prompt(f"Choose strategy ({choices})", default="default").strip().lower()
        if raw in SUPPORTED_STRATEGIES:
            return raw
        print(f"Unknown strategy '{raw}'. Valid choices: {choices}")


def _resolve_strategy(args: argparse.Namespace) -> str:
    if args.strategy in SUPPORTED_STRATEGIES:
        return args.strategy
    return _prompt_strategy()


def _make_run_dir(base_dir: Path) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    existing = []
    for entry in base_dir.iterdir():
        if entry.is_dir() and entry.name.startswith("run_"):
            suffix = entry.name.split("_", 1)[-1]
            if suffix.isdigit():
                existing.append(int(suffix))
    next_run = max(existing, default=0) + 1
    run_dir = base_dir / f"run_{next_run}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _is_direct_strategy_log_dir(base_dir: Path) -> bool:
    parent = base_dir.parent
    return base_dir.name in SUPPORTED_STRATEGIES and parent.name.startswith("run_")


def _build_agent(strategy: str, device: torch.device):
    return _build_agent_with_shape(
        strategy,
        device,
        state_shape=DEFAULT_STATE_SHAPE,
        n_steps=1,
    )


def _agent_state_shape(frontier_curriculum: bool) -> Tuple[int, int, int]:
    if not frontier_curriculum:
        return DEFAULT_STATE_SHAPE
    return (
        DEFAULT_STATE_SHAPE[0] + DEFAULT_FRONTIER_CONTEXT_CHANNELS,
        DEFAULT_STATE_SHAPE[1],
        DEFAULT_STATE_SHAPE[2],
    )


def _build_agent_with_shape(
    strategy: str,
    device: torch.device,
    *,
    state_shape: Tuple[int, int, int],
    n_steps: int,
):
    state_dim = int(np.prod(state_shape))
    if strategy == "default":
        return PinnAgent(state_shape, DEFAULT_ACTION_DIMS, device=device, n_steps=n_steps)
    if strategy == "mpdqn":
        return MPDQNAgent(state_shape, DEFAULT_ACTION_DIMS, device=device, n_steps=n_steps)
    if strategy == "seq-madac":
        return SeqMADACAgent(state_dim, 10, device=device, n_steps=n_steps)
    if strategy == "flexplore":
        return FLEXploreAgent(state_dim, 10, device=device)
    raise ValueError(f"Unknown strategy: {strategy}")


def _agent_buffer(agent: Any):
    if hasattr(agent, "memory"):
        return agent.memory
    if hasattr(agent, "replay"):
        return agent.replay
    raise AttributeError(f"Agent {type(agent).__name__} does not expose a replay buffer")


def _push_transition(agent: Any, transition: Tuple[Any, ...]) -> None:
    _agent_buffer(agent).push(*transition)


def _decode_action(agent: Any, action: Any) -> Dict[str, Any]:
    if isinstance(action, dict):
        return action
    if isinstance(action, (tuple, list)) and len(action) == 3 and isinstance(action[2], dict):
        return action[2]
    if hasattr(agent, "decode_action"):
        return agent.decode_action(action)
    raise ValueError(f"Cannot decode action of type {type(action)!r}")


def _normalize_action_for_buffer(agent: Any, action: Any) -> Any:
    if isinstance(agent, SeqMADACAgent):
        try:
            return agent._trace_from_action(action)
        except Exception:
            pass
    if hasattr(agent, "action_spec") and isinstance(action, dict):
        try:
            action_id, params = agent.action_spec.encode(action)
            return (int(action_id), np.asarray(params, dtype=np.float32), action)
        except Exception:
            pass
    if isinstance(action, dict):
        return action
    if isinstance(action, (tuple, list)) and len(action) == 3 and isinstance(action[2], dict):
        return action
    return action


def _to_float_list(values: Sequence[Any]) -> List[float]:
    result: List[float] = []
    for value in values:
        if hasattr(value, "item"):
            result.append(float(value.item()))
        else:
            result.append(float(value))
    return result


@dataclass(frozen=True)
class FrontierStateMeta:
    cycle_index: int
    total_cycles: int
    decision_index: int
    total_decisions: int
    stage_start_epoch: int
    stage_end_epoch: int
    total_epoch_budget: int


def _frontier_state_meta(
    stage: "CurriculumStage",
    *,
    cycle_index: int,
    total_cycles: int,
    decision_index: int,
    total_decisions: int,
) -> FrontierStateMeta:
    return FrontierStateMeta(
        cycle_index=int(cycle_index),
        total_cycles=max(1, int(total_cycles)),
        decision_index=max(0, int(decision_index)),
        total_decisions=max(1, int(total_decisions)),
        stage_start_epoch=int(stage.start_epoch),
        stage_end_epoch=int(stage.end_epoch),
        total_epoch_budget=max(1, int(stage.total_epoch_budget)),
    )


def _frontier_context_channels(meta: FrontierStateMeta) -> np.ndarray:
    cycle_frac = float(meta.cycle_index + 1) / float(max(1, meta.total_cycles))
    start_frac = float(meta.stage_start_epoch) / float(max(1, meta.total_epoch_budget))
    end_frac = float(meta.stage_end_epoch) / float(max(1, meta.total_epoch_budget))
    remaining_frac = float(max(meta.total_decisions - meta.decision_index, 0)) / float(max(1, meta.total_decisions))

    channels = np.empty(
        (DEFAULT_FRONTIER_CONTEXT_CHANNELS, DEFAULT_STATE_SHAPE[1], DEFAULT_STATE_SHAPE[2]),
        dtype=np.float32,
    )
    channels[0].fill(cycle_frac)
    channels[1].fill(start_frac)
    channels[2].fill(end_frac)
    channels[3].fill(remaining_frac)
    return channels


def _augment_state_with_frontier_context(state_np: np.ndarray, meta: Optional[FrontierStateMeta]) -> np.ndarray:
    arr = np.asarray(state_np, dtype=np.float32)
    if meta is None:
        return arr
    return np.concatenate((arr, _frontier_context_channels(meta)), axis=0).astype(np.float32, copy=False)


def _extract_env_baseline_action(
    env: TorchHoloEnv,
    *,
    scale_baseline: Optional[float] = None,
) -> Dict[str, Any]:
    group = env.optimizer.param_groups[0] if getattr(env.optimizer, "param_groups", None) else {}
    lr = float(group.get("lr", 1e-4))
    beta_1, beta_2 = group.get("betas", (0.9, 0.999))
    if scale_baseline is None:
        scales = np.asarray(getattr(env, "s_v", np.ones(7, dtype=np.float32)), dtype=np.float32).reshape(-1)[:7]
    else:
        scales = np.full(7, float(scale_baseline), dtype=np.float32)
    if scales.size < 7:
        scales = np.pad(scales, (0, 7 - scales.size), constant_values=1.0)
    return {
        "optimizer": "adam",
        "lr": float(np.clip(lr, 1e-9, 1e-2)),
        "betas": tuple(sorted((float(beta_1), float(beta_2)))),
        "s_v": np.clip(scales, 0.0, 1.0).astype(np.float32),
    }


def _default_relative_action(agent: PinnAgent, action: Any, baseline: Dict[str, Any]) -> Dict[str, Any]:
    indices = np.asarray(action, dtype=np.int64).reshape(-1)
    if indices.size < len(agent.action_dims):
        indices = np.pad(indices, (0, len(agent.action_dims) - indices.size), constant_values=0)

    lr_delta_grid = np.linspace(
        -DEFAULT_FRONTIER_LR_LOG_DELTA,
        DEFAULT_FRONTIER_LR_LOG_DELTA,
        len(agent.lr_options),
        dtype=np.float32,
    )
    beta_delta_values = np.linspace(
        -DEFAULT_FRONTIER_BETA_DELTA,
        DEFAULT_FRONTIER_BETA_DELTA,
        6,
        dtype=np.float32,
    )
    beta_delta_pairs = list(itertools.combinations(beta_delta_values.tolist(), 2))
    scale_delta_grid = np.linspace(
        -DEFAULT_FRONTIER_SCALE_DELTA,
        DEFAULT_FRONTIER_SCALE_DELTA,
        len(agent.scale_options),
        dtype=np.float32,
    )

    lr_index = int(indices[0]) % len(lr_delta_grid)
    beta_index = int(indices[1]) % len(beta_delta_pairs)
    lr = float(np.clip(baseline["lr"] * (10.0 ** float(lr_delta_grid[lr_index])), 1e-9, 1e-2))

    base_beta_1, base_beta_2 = baseline["betas"]
    delta_beta_1, delta_beta_2 = beta_delta_pairs[beta_index]
    beta_1 = float(np.clip(base_beta_1 + delta_beta_1, 0.8, 0.999))
    beta_2 = float(np.clip(base_beta_2 + delta_beta_2, 0.8, 0.999))
    betas = tuple(sorted((beta_1, beta_2)))

    scale_indices = indices[3 : 3 + 7]
    scale_deltas = scale_delta_grid[scale_indices % len(scale_delta_grid)]
    scales = np.asarray(baseline["s_v"], dtype=np.float32) + np.asarray(scale_deltas, dtype=np.float32)
    return {
        "optimizer": "adam",
        "lr": lr,
        "betas": betas,
        "s_v": scales.astype(np.float32),
    }


def _relative_value(value: float, half_width: float) -> float:
    clipped = float(np.clip(value, 0.0, 1.0))
    return -half_width + (2.0 * half_width * clipped)


def _param_relative_action(agent: Any, action: Any, baseline: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(action, dict) and hasattr(agent, "action_spec"):
        action_id, params = agent.action_spec.encode(action)
        raw_params = np.asarray(params, dtype=np.float32).reshape(-1)
    elif isinstance(action, np.ndarray):
        arr = np.asarray(action, dtype=object).reshape(-1)
        action_id = int(arr[0]) if arr.size >= 1 else 0
        raw_params = np.asarray(arr[1], dtype=np.float32).reshape(-1) if arr.size >= 2 else np.zeros(10, dtype=np.float32)
    elif isinstance(action, (tuple, list)):
        action_id = int(action[0]) if len(action) >= 1 else 0
        raw_params = np.asarray(action[1], dtype=np.float32).reshape(-1) if len(action) >= 2 else np.zeros(10, dtype=np.float32)
    else:
        action_id = int(action) if np.isscalar(action) else 0
        raw_params = np.zeros(10, dtype=np.float32)

    if raw_params.size < 10:
        raw_params = np.pad(raw_params, (0, 10 - raw_params.size), constant_values=0.5)
    raw_params = np.nan_to_num(raw_params[:10], nan=0.5, posinf=1.0, neginf=0.0)

    lr = float(
        np.clip(
            baseline["lr"] * (10.0 ** _relative_value(float(raw_params[0]), DEFAULT_FRONTIER_LR_LOG_DELTA)),
            1e-9,
            1e-2,
        )
    )
    base_beta_1, base_beta_2 = baseline["betas"]
    beta_1 = float(np.clip(base_beta_1 + _relative_value(float(raw_params[1]), DEFAULT_FRONTIER_BETA_DELTA), 0.8, 0.999))
    beta_2 = float(np.clip(base_beta_2 + _relative_value(float(raw_params[2]), DEFAULT_FRONTIER_BETA_DELTA), 0.8, 0.999))
    betas = tuple(sorted((beta_1, beta_2)))

    base_scales = np.asarray(baseline["s_v"], dtype=np.float32).reshape(-1)[:7]
    deltas = np.asarray(
        [_relative_value(float(value), DEFAULT_FRONTIER_SCALE_DELTA) for value in raw_params[3:10]],
        dtype=np.float32,
    )
    scales = base_scales + deltas
    return {
        "optimizer": "adam" if int(action_id) in (0, 1) else "adam",
        "lr": lr,
        "betas": betas,
        "s_v": scales.astype(np.float32),
    }


def _decode_relative_action(
    agent: Any,
    action: Any,
    env: TorchHoloEnv,
    *,
    scale_baseline: Optional[float] = None,
) -> Dict[str, Any]:
    baseline = _extract_env_baseline_action(env, scale_baseline=scale_baseline)
    if isinstance(agent, PinnAgent):
        return _default_relative_action(agent, action, baseline)
    return _param_relative_action(agent, action, baseline)


def _save_env_state(env: TorchHoloEnv, path: Path, *, include_metrics: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "nets_state": [net.state_dict() for net in env.nets],
        "v_state": env.V.state_dict(),
        "optimizer_state": env.optimizer.state_dict() if hasattr(env, "optimizer") else None,
        "optim_state": dict(env.optim_state),
        "s_v": np.asarray(env.s_v, dtype=np.float32),
    }
    if include_metrics:
        payload["metrics_history"] = {
            key: _to_float_list(values)
            for key, values in env.solver.metrics_history.items()
        }
    torch.save(payload, path)


def _load_env_state(env: TorchHoloEnv, path: Path) -> None:
    state = torch.load(path, map_location=env.device, weights_only=False)
    for net, net_state in zip(env.nets, state["nets_state"]):
        net.load_state_dict(net_state)
    env.V.load_state_dict(state["v_state"])
    if state.get("optimizer_state") is not None:
        env.optimizer.load_state_dict(state["optimizer_state"])
    env.solver.optimizer = env.optimizer
    env.optim_state.clear()
    env.optim_state.update(state.get("optim_state", {"cnt": 0}))
    env.solver.optim_state = env.optim_state
    env.s_v = np.asarray(state.get("s_v", np.ones(7, dtype=np.float32)), dtype=np.float32)
    metrics_history = state.get("metrics_history")
    if metrics_history is not None:
        env.solver.metrics_history = defaultdict(
            list,
            {key: list(values) for key, values in metrics_history.items()},
        )
    else:
        env.solver.metrics_history = defaultdict(
            list,
            {"r2_loss": [], "phi_max": [], "train_loss": [], "valid_loss": []},
        )


def _prepare_strategy_worker(
    device: torch.device,
    cinns_path: str,
    strategy: str,
    state_shape: Tuple[int, int, int],
    n_steps: int,
    env_state_path: Optional[str] = None,
):
    env = TorchHoloEnv(cinns_path=cinns_path, device=device)
    if env_state_path:
        _load_env_state(env, Path(env_state_path))
    agent = _build_agent_with_shape(strategy, device, state_shape=state_shape, n_steps=n_steps)
    return env, agent


def _env_start_state(env: TorchHoloEnv, env_state_path: Optional[str]) -> torch.Tensor:
    if env_state_path:
        return env.get_residuals()
    return env.reset()


def _aggregate_n_step_transitions(
    transitions: Sequence[Tuple[Any, Any, float, Any, bool]],
    horizon: int,
    *,
    gamma: float = 0.99,
) -> List[Tuple[Any, Any, float, Any, bool]]:
    if horizon <= 1:
        return list(transitions)

    aggregated: List[Tuple[Any, Any, float, Any, bool]] = []
    for index, transition in enumerate(transitions):
        state, action, _, _, _ = transition
        reward_sum = 0.0
        end_index = index
        done = False
        for offset in range(horizon):
            current_index = index + offset
            if current_index >= len(transitions):
                break
            _, _, reward, next_state, is_done = transitions[current_index]
            reward_sum += (gamma ** offset) * float(reward)
            end_index = current_index
            done = bool(is_done)
            if done:
                break
        next_state = transitions[end_index][3]
        aggregated.append((state, action, float(reward_sum), next_state, done))
    return aggregated


def _run_episode_steps(
    env: TorchHoloEnv,
    agent: Any,
    stage: "CurriculumStage",
    steps: int,
    *,
    eval_mode: bool,
    collect_transitions: bool,
    state_np: Optional[np.ndarray] = None,
    cycle_index: Optional[int] = None,
    total_cycles: Optional[int] = None,
    n_step_horizon: int = 1,
    frontier_relative_actions: bool = False,
    frontier_scale_baseline: Optional[float] = None,
) -> Tuple[np.ndarray, List[Tuple[Any, Any, float, Any, bool]]]:
    if state_np is None:
        state = env.get_residuals()
        state_np = state.detach().cpu().numpy()

    raw_state_np = np.asarray(state_np, dtype=np.float32)
    memories: List[Tuple[Any, Any, float, Any, bool]] = []
    for local_step in range(1, steps + 1):
        meta = None
        if cycle_index is not None and total_cycles is not None:
            meta = _frontier_state_meta(
                stage,
                cycle_index=cycle_index,
                total_cycles=total_cycles,
                decision_index=local_step - 1,
                total_decisions=steps,
            )
        state_with_context = _augment_state_with_frontier_context(raw_state_np, meta)
        action = agent.act(state_with_context, eval_mode=eval_mode)
        buffer_action, action_dict = _prepare_stage_action(
            agent,
            action,
            stage,
            env=env,
            frontier_relative=frontier_relative_actions,
            frontier_scale_baseline=frontier_scale_baseline,
        )
        residuals, _, old_loss = env.step(action_dict)
        raw_next_state_np = residuals.detach().cpu().numpy()
        if collect_transitions:
            next_meta = None
            if cycle_index is not None and total_cycles is not None:
                next_meta = _frontier_state_meta(
                    stage,
                    cycle_index=cycle_index,
                    total_cycles=total_cycles,
                    decision_index=local_step,
                    total_decisions=steps,
                )
            unweighted_new = env._compute_unweighted_loss()
            reward = _project_compute_reward(unweighted_new, old_loss, action_dict["s_v"], local_step)
            done = local_step == steps
            next_state_with_context = _augment_state_with_frontier_context(raw_next_state_np, next_meta)
            memories.append((state_with_context, buffer_action, reward, next_state_with_context, done))
        raw_state_np = np.asarray(raw_next_state_np, dtype=np.float32)
    if collect_transitions:
        memories = _aggregate_n_step_transitions(memories, n_step_horizon)
    return raw_state_np, memories


def _collect_strategy_episode(
    args: Tuple[
        str,
        str,
        str,
        str,
        Tuple[int, int, int],
        int,
        int,
        int,
        int,
        float,
        bool,
        CurriculumStage,
        bool,
        Optional[str],
    ]
) -> List[Tuple[Any, Any, float, Any, bool]]:
    (
        strategy,
        snapshot_path,
        cinns_path,
        device_str,
        state_shape,
        n_steps,
        cycle_index,
        total_cycles,
        steps_per_episode,
        epsilon,
        eval_mode,
        stage,
        frontier_relative_actions,
        env_state_path,
    ) = args
    device = torch.device(device_str)
    env, agent = _prepare_strategy_worker(
        device,
        cinns_path,
        strategy,
        state_shape=state_shape,
        n_steps=n_steps,
        env_state_path=env_state_path,
    )
    agent.load(snapshot_path)
    if hasattr(agent, "epsilon"):
        agent.epsilon = float(epsilon)
    _set_agent_epsilon_schedule(agent, stage)
    state = _env_start_state(env, env_state_path)
    state_np = state.detach().cpu().numpy()
    frontier_scale_baseline = _frontier_cycle_scale_baseline(cycle_index) if frontier_relative_actions else None
    _, memories = _run_episode_steps(
        env,
        agent,
        stage,
        steps_per_episode,
        eval_mode=eval_mode,
        collect_transitions=True,
        state_np=state_np,
        cycle_index=cycle_index,
        total_cycles=total_cycles,
        n_step_horizon=n_steps,
        frontier_relative_actions=frontier_relative_actions,
        frontier_scale_baseline=frontier_scale_baseline,
    )
    return memories


def _collect_strategy_episode_to_disk(
    args: Tuple[
        str,
        str,
        str,
        str,
        Tuple[int, int, int],
        int,
        int,
        int,
        int,
        float,
        bool,
        CurriculumStage,
        str,
        bool,
        Optional[str],
    ]
) -> str:
    (
        strategy,
        snapshot_path,
        cinns_path,
        device_str,
        state_shape,
        n_steps,
        cycle_index,
        total_cycles,
        steps_per_episode,
        epsilon,
        eval_mode,
        stage,
        spool_dir,
        frontier_relative_actions,
        env_state_path,
    ) = args
    memories = _collect_strategy_episode(
        (
            strategy,
            snapshot_path,
            cinns_path,
            device_str,
            state_shape,
            n_steps,
            cycle_index,
            total_cycles,
            steps_per_episode,
            epsilon,
            eval_mode,
            stage,
            frontier_relative_actions,
            env_state_path,
        )
    )
    with tempfile.NamedTemporaryFile(
        mode="wb",
        delete=False,
        dir=spool_dir,
        prefix="episode_",
        suffix=".pkl",
    ) as tmpf:
        pickle.dump(memories, tmpf, protocol=pickle.HIGHEST_PROTOCOL)
        return tmpf.name


def _resolve_parallel_devices(device: torch.device, worker_devices: Optional[Sequence[torch.device]] = None) -> List[torch.device]:
    if worker_devices is not None:
        return list(worker_devices)
    if device.type == "cpu":
        return [device]
    available = [d for d in get_available_devices() if d.type != "cpu"]
    return available or [device]


@dataclass
class RuntimeConfig:
    strategy: str
    device: torch.device
    workers: int
    episodes_per_cycle: int
    cycles: int
    target_memories: int
    train_epochs: int
    patience: int
    min_delta: float
    steps: int
    mem_limit: float
    cinns_path: str
    log_dir: str
    skip_render: bool
    resume: Optional[str]
    frontier_curriculum: bool
    frontier_epsilon: float
    replay_max_size: int
    frontier_single_epsilon_collection: bool = False
    frontier_memory_schedule: Tuple[int, ...] = FRONTIER_TARGET_MEMORY_SCHEDULE
    frontier_epsilon_schedule: Tuple[float, ...] = ()
    frontier_steps_per_episode: int = 200
    frontier_memory_block: int = FRONTIER_MEMORY_BLOCK
    frontier_free_scales: bool = False
    cycle_stop_enabled: bool = True


@dataclass(frozen=True)
class CurriculumStage:
    name: str
    steps_per_episode: int
    action_epochs: int
    epsilon: float
    scale_floor: float
    freeze_scales: bool = False
    start_epoch: int = 0
    end_epoch: int = 0
    total_epoch_budget: int = 0


@dataclass(frozen=True)
class CollectionPhase:
    name: str
    target_memories: int
    eval_mode: bool
    epsilon: float


@dataclass(frozen=True)
class CycleStopState:
    best_log_metric: Optional[float] = None
    best_cycle: int = 0
    patience_counter: int = 0


def _annotate_stage_bounds(stages: Sequence[CurriculumStage]) -> Tuple[CurriculumStage, ...]:
    total = int(sum(stage.steps_per_episode * stage.action_epochs for stage in stages))
    cursor = 0
    annotated: List[CurriculumStage] = []
    for stage in stages:
        span = int(stage.steps_per_episode * stage.action_epochs)
        annotated.append(
            replace(
                stage,
                start_epoch=cursor,
                end_epoch=cursor + span,
                total_epoch_budget=total,
            )
        )
        cursor += span
    return tuple(annotated)


def _build_flat_frontier_render_stage(
    steps_per_episode: int = PROJECT_RENDER_DECISIONS,
    action_epochs: int = PROJECT_RENDER_ACTION_EPOCHS,
) -> CurriculumStage:
    return _annotate_stage_bounds(
        (
            CurriculumStage(
                name="frontier_render_flat",
                steps_per_episode=int(steps_per_episode),
                action_epochs=int(action_epochs),
                epsilon=0.0,
                scale_floor=0.0,
                freeze_scales=False,
            ),
        )
    )[0]


CURRICULUM_STAGES: Tuple[CurriculumStage, ...] = _annotate_stage_bounds((
    CurriculumStage(
        name="warmup_a",
        steps_per_episode=1000,
        action_epochs=30,
        epsilon=0.00,
        scale_floor=0.10,
    ),
    CurriculumStage(
        name="warmup_b",
        steps_per_episode=1000,
        action_epochs=30,
        epsilon=0.10,
        scale_floor=0.10,
    ),
    CurriculumStage(
        name="transition_a",
        steps_per_episode=750,
        action_epochs=40,
        epsilon=0.15,
        scale_floor=0.20,
    ),
    CurriculumStage(
        name="transition_b",
        steps_per_episode=750,
        action_epochs=40,
        epsilon=0.20,
        scale_floor=0.30,
    ),
    CurriculumStage(
        name="ramp_a",
        steps_per_episode=600,
        action_epochs=50,
        epsilon=0.25,
        scale_floor=0.40,
    ),
    CurriculumStage(
        name="ramp_b",
        steps_per_episode=600,
        action_epochs=50,
        epsilon=0.35,
        scale_floor=0.50,
    ),
    CurriculumStage(
        name="ramp_c",
        steps_per_episode=500,
        action_epochs=60,
        epsilon=0.45,
        scale_floor=0.60,
    ),
    CurriculumStage(
        name="ramp_d",
        steps_per_episode=500,
        action_epochs=60,
        epsilon=0.50,
        scale_floor=0.70,
    ),
    CurriculumStage(
        name="constraint_a",
        steps_per_episode=400,
        action_epochs=75,
        epsilon=0.55,
        scale_floor=0.80,
    ),
    CurriculumStage(
        name="constraint_b",
        steps_per_episode=400,
        action_epochs=75,
        epsilon=0.60,
        scale_floor=0.80,
    ),
    CurriculumStage(
        name="constraint_c",
        steps_per_episode=375,
        action_epochs=80,
        epsilon=0.65,
        scale_floor=0.85,
    ),
    CurriculumStage(
        name="constraint_d",
        steps_per_episode=375,
        action_epochs=80,
        epsilon=0.70,
        scale_floor=0.90,
    ),
    CurriculumStage(
        name="constraint_e",
        steps_per_episode=300,
        action_epochs=100,
        epsilon=0.75,
        scale_floor=0.95,
    ),
    CurriculumStage(
        name="final_a",
        steps_per_episode=300,
        action_epochs=100,
        epsilon=0.75,
        scale_floor=1.00,
        freeze_scales=True,
    ),
    CurriculumStage(
        name="final_b",
        steps_per_episode=300,
        action_epochs=100,
        epsilon=0.75,
        scale_floor=1.00,
        freeze_scales=True,
    ),
))


def _build_frontier_stages(
    epsilon: float,
    *,
    cycles: Optional[int] = None,
    epsilon_schedule: Optional[Sequence[float]] = None,
    steps_per_episode: int = 200,
    freeze_final_scales: bool = True,
) -> Tuple[CurriculumStage, ...]:
    stages = []
    stage_count = int(cycles) if cycles is not None else len(FRONTIER_SCALE_BASELINE_SCHEDULE)
    if stage_count <= 0:
        raise ValueError("Frontier curriculum must contain at least one cycle")
    if int(steps_per_episode) <= 0:
        raise ValueError("Frontier steps per episode must be positive")
    active_epsilon_schedule = tuple(epsilon_schedule or ())
    if active_epsilon_schedule and len(active_epsilon_schedule) not in (1,) and len(active_epsilon_schedule) < stage_count:
        raise ValueError(
            "frontier epsilon schedule must provide at least one value per frontier cycle "
            f"(got {len(active_epsilon_schedule)} for {stage_count})"
        )
    # Keep the stage metadata aligned with the active cycle baseline schedule.
    # Frontier decoding no longer clamps to this value during training or render.
    for index in range(stage_count):
        scale_floor = FRONTIER_SCALE_BASELINE_SCHEDULE[
            min(index, len(FRONTIER_SCALE_BASELINE_SCHEDULE) - 1)
        ]
        stage_epsilon = (
            float(active_epsilon_schedule[min(index, len(active_epsilon_schedule) - 1)])
            if active_epsilon_schedule
            else float(epsilon)
        )
        stages.append(
            CurriculumStage(
                name=f"frontier_{index + 1:02d}",
                steps_per_episode=int(steps_per_episode),
                action_epochs=50,
                epsilon=stage_epsilon,
                scale_floor=float(scale_floor),
                freeze_scales=bool(freeze_final_scales and index == stage_count - 1),
            )
        )
    return _annotate_stage_bounds(tuple(stages))


def _frontier_cycle_target_memories(
    cycle_index: int,
    schedule: Sequence[int] = FRONTIER_TARGET_MEMORY_SCHEDULE,
) -> int:
    if cycle_index < 0:
        raise ValueError("cycle_index must be non-negative")
    if cycle_index >= len(schedule):
        raise ValueError(
            f"Configured cycle_index {cycle_index} exceeds supported frontier memory schedule "
            f"length {len(schedule)}"
        )
    return int(schedule[cycle_index])


def _frontier_cycle_scale_baseline(cycle_index: int) -> float:
    if cycle_index < 0:
        raise ValueError("cycle_index must be non-negative")
    # Additional cycles remain at the final curriculum baseline unless an
    # explicit baseline schedule is introduced.
    baseline_index = min(cycle_index, len(FRONTIER_SCALE_BASELINE_SCHEDULE) - 1)
    return float(FRONTIER_SCALE_BASELINE_SCHEDULE[baseline_index])


def _frontier_collection_plan(
    cycle_index: int,
    epsilon: float,
    schedule: Sequence[int] = FRONTIER_TARGET_MEMORY_SCHEDULE,
    memory_block: int = FRONTIER_MEMORY_BLOCK,
) -> Tuple[CollectionPhase, ...]:
    target_memories = _frontier_cycle_target_memories(cycle_index, schedule)
    memory_block = max(1, int(memory_block))
    greedy_memories = max(0, target_memories - memory_block)
    phases: List[CollectionPhase] = []
    if greedy_memories > 0:
        phases.append(
            CollectionPhase(
                name="greedy",
                target_memories=greedy_memories,
                eval_mode=True,
                epsilon=0.0,
            )
        )
    phases.append(
        CollectionPhase(
            name="exploration",
                target_memories=min(memory_block, target_memories),
            eval_mode=False,
            epsilon=float(epsilon),
        )
    )
    return tuple(phases)


def _frontier_replay_max_size(
    cycles: int,
    schedule: Sequence[int] = FRONTIER_TARGET_MEMORY_SCHEDULE,
) -> int:
    if cycles <= 0:
        return 0
    return max(schedule[: min(int(cycles), len(schedule))])


def _parse_frontier_memory_schedule(raw_schedule: Optional[str]) -> Tuple[int, ...]:
    if not raw_schedule:
        return tuple(FRONTIER_TARGET_MEMORY_SCHEDULE)
    try:
        schedule = tuple(int(part.strip()) for part in raw_schedule.split(",") if part.strip())
    except ValueError as exc:
        raise ValueError("--frontier_memory_schedule must be a comma-separated list of positive integers") from exc
    if not schedule or any(value <= 0 for value in schedule):
        raise ValueError("--frontier_memory_schedule must contain positive integers")
    return schedule


def _parse_frontier_epsilon_schedule(
    raw_schedule: Optional[str],
    fallback: float,
) -> Tuple[float, ...]:
    if not raw_schedule:
        return (float(fallback),)
    try:
        schedule = tuple(float(part.strip()) for part in raw_schedule.split(",") if part.strip())
    except ValueError as exc:
        raise ValueError("--frontier_epsilon_schedule must be comma-separated numeric values") from exc
    if not schedule or any(value < 0.0 or value > 1.0 for value in schedule):
        raise ValueError("--frontier_epsilon_schedule values must be within [0, 1]")
    return schedule


def _curve_correlation(values: Sequence[float], reference: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    ref = np.asarray(reference, dtype=np.float64).reshape(-1)
    if arr.size != ref.size or arr.size == 0:
        return 0.0
    arr_centered = arr - arr.mean()
    ref_centered = ref - ref.mean()
    arr_norm = float(np.linalg.norm(arr_centered))
    ref_norm = float(np.linalg.norm(ref_centered))
    if arr_norm <= 1e-12 or ref_norm <= 1e-12:
        return 0.0
    corr = float(np.dot(arr_centered, ref_centered) / (arr_norm * ref_norm))
    return float(np.clip(corr, -1.0, 1.0))


def _analyze_potential_shape(curve: Sequence[float], theory_curve: Sequence[float]) -> Dict[str, Any]:
    arr = np.asarray(curve, dtype=np.float64).reshape(-1)
    theory = np.asarray(theory_curve, dtype=np.float64).reshape(-1)
    if arr.size == 0 or theory.size != arr.size:
        return {
            "pass": False,
            "corr_to_theory": 0.0,
            "drop_from_start": 0.0,
            "rebound_from_min": 0.0,
            "min_index": -1,
            "min_fraction": 0.0,
        }

    min_index = int(np.argmin(arr))
    start_value = float(arr[0])
    min_value = float(arr[min_index])
    end_value = float(arr[-1])
    drop_from_start = float(start_value - min_value)
    rebound_from_min = float(end_value - min_value)
    min_fraction = float(min_index) / float(max(1, arr.size - 1))
    corr_to_theory = _curve_correlation(arr, theory)
    passed = bool(
        corr_to_theory >= PROJECT_SHAPE_MIN_CORR
        and drop_from_start >= PROJECT_SHAPE_MIN_DROP
        and rebound_from_min >= PROJECT_SHAPE_MIN_REBOUND
        and min_fraction >= PROJECT_SHAPE_MIN_MIN_INDEX_FRAC
    )
    return {
        "pass": passed,
        "corr_to_theory": corr_to_theory,
        "drop_from_start": drop_from_start,
        "rebound_from_min": rebound_from_min,
        "min_index": min_index,
        "min_fraction": min_fraction,
    }


def _shape_gate_report(render_report: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    potential = ((render_report or {}).get("potential") or {})
    theory_curve = potential.get("theory") or []
    current_curve = potential.get("current") or []
    best_curve = potential.get("best") or []
    current_metrics = _analyze_potential_shape(current_curve, theory_curve)
    best_metrics = _analyze_potential_shape(best_curve, theory_curve) if best_curve else None
    passed = bool(current_metrics["pass"] or (best_metrics is not None and best_metrics["pass"]))
    return {
        "pass": passed,
        "current": current_metrics,
        "best": best_metrics,
    }


def _extract_cycle_stop_metric(render_report: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    loss_report = ((render_report or {}).get("loss") or {})
    candidates = (
        ("current_unweighted", loss_report.get("current_unweighted")),
        ("current_validation", loss_report.get("current_validation")),
        ("best_unweighted", loss_report.get("best_unweighted")),
        ("best_validation", loss_report.get("best_validation")),
    )
    for source, value in candidates:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(numeric) or numeric <= 0.0:
            continue
        log_metric = float(np.log10(max(numeric, 1e-300)))
        return {
            "source": source,
            "value": numeric,
            "log10": log_metric,
        }
    return None


def _advance_cycle_stop_state(
    cycle_number: int,
    render_report: Optional[Dict[str, Any]],
    state: CycleStopState,
) -> Tuple[CycleStopState, Dict[str, Any]]:
    metric = _extract_cycle_stop_metric(render_report)
    if metric is None:
        info = {
            "available": False,
            "cycle": int(cycle_number),
            "stop": False,
            "patience_counter": int(state.patience_counter),
        }
        return state, info

    if state.best_log_metric is None:
        new_state = CycleStopState(
            best_log_metric=float(metric["log10"]),
            best_cycle=int(cycle_number),
            patience_counter=0,
        )
        info = {
            "available": True,
            "cycle": int(cycle_number),
            "baseline": True,
            "improved": True,
            "metric": metric,
            "best_log_metric": float(metric["log10"]),
            "best_cycle": int(cycle_number),
            "log_delta": None,
            "patience_counter": 0,
            "stop": False,
        }
        return new_state, info

    log_delta = float(metric["log10"]) - float(state.best_log_metric)
    improved = bool(log_delta <= PROJECT_CYCLE_STOP_MIN_LOG_DELTA)
    if improved:
        new_state = CycleStopState(
            best_log_metric=float(metric["log10"]),
            best_cycle=int(cycle_number),
            patience_counter=0,
        )
    else:
        new_state = CycleStopState(
            best_log_metric=float(state.best_log_metric),
            best_cycle=int(state.best_cycle),
            patience_counter=int(state.patience_counter) + 1,
        )
    stop = bool(new_state.patience_counter >= PROJECT_CYCLE_STOP_PATIENCE)
    info = {
        "available": True,
        "cycle": int(cycle_number),
        "baseline": False,
        "improved": improved,
        "metric": metric,
        "best_log_metric": float(state.best_log_metric),
        "best_cycle": int(state.best_cycle),
        "log_delta": log_delta,
        "patience_counter": int(new_state.patience_counter),
        "stop": stop,
    }
    return new_state, info


def _log_cycle_stop_result(cycle_number: int, stop_info: Dict[str, Any]) -> None:
    if not bool(stop_info.get("available")):
        logger.info(
            "Cycle %s loss stop metric unavailable | patience=%s/%s | stop=%s",
            cycle_number,
            int(stop_info.get("patience_counter", 0)),
            PROJECT_CYCLE_STOP_PATIENCE,
            bool(stop_info.get("stop")),
        )
        return

    metric = stop_info["metric"]
    if bool(stop_info.get("baseline")):
        logger.info(
            "Cycle %s loss stop baseline | source=%s | value=%.6e | log10=%.3f | threshold=%.3f | patience=0/%s",
            cycle_number,
            metric["source"],
            float(metric["value"]),
            float(metric["log10"]),
            PROJECT_CYCLE_STOP_MIN_LOG_DELTA,
            PROJECT_CYCLE_STOP_PATIENCE,
        )
        return

    logger.info(
        "Cycle %s loss stop | source=%s | value=%.6e | log10=%.3f | best_log10=%.3f (cycle %s) | "
        "delta=%.3f | improved=%s | patience=%s/%s | stop=%s",
        cycle_number,
        metric["source"],
        float(metric["value"]),
        float(metric["log10"]),
        float(stop_info["best_log_metric"]),
        int(stop_info["best_cycle"]),
        float(stop_info["log_delta"]),
        bool(stop_info["improved"]),
        int(stop_info["patience_counter"]),
        PROJECT_CYCLE_STOP_PATIENCE,
        bool(stop_info["stop"]),
    )


def _curriculum_stage_for_cycle(
    cycle_index: int,
    schedule: Optional[Sequence[CurriculumStage]] = None,
) -> CurriculumStage:
    active_schedule = tuple(schedule) if schedule is not None else CURRICULUM_STAGES
    if cycle_index < 0:
        raise ValueError("cycle_index must be non-negative")
    if cycle_index >= len(active_schedule):
        raise ValueError(
            f"Configured cycle_index {cycle_index} exceeds supported schedule "
            f"length {len(active_schedule)}"
        )
    return active_schedule[cycle_index]


def _curriculum_scan_steps(default_steps: int, schedule: Optional[Sequence[CurriculumStage]] = None) -> int:
    active_schedule = tuple(schedule) if schedule is not None else CURRICULUM_STAGES
    return max(default_steps, max(stage.steps_per_episode for stage in active_schedule))


def _set_agent_epsilon_schedule(agent: Any, stage: CurriculumStage) -> None:
    if hasattr(agent, "epsilon_decay"):
        agent.epsilon_decay = 1.0
    if hasattr(agent, "epsilon_min"):
        agent.epsilon_min = float(stage.epsilon)


def _clear_agent_buffer(agent: Any) -> None:
    buffer = _agent_buffer(agent)
    if hasattr(buffer, "clear"):
        buffer.clear()


def _trim_agent_buffer(agent: Any, max_size: int) -> None:
    buffer = _agent_buffer(agent)
    raw = getattr(buffer, "buffer", None)
    if raw is None or max_size <= 0:
        return
    while len(raw) > max_size:
        raw.popleft()


def _prepare_stage_action(
    agent: Any,
    action: Any,
    stage: CurriculumStage,
    *,
    env: Optional[TorchHoloEnv] = None,
    frontier_relative: bool = False,
    frontier_scale_baseline: Optional[float] = None,
) -> Tuple[Any, Dict[str, Any]]:
    if frontier_relative and env is not None:
        action_dict = _decode_relative_action(
            agent,
            action,
            env,
            scale_baseline=frontier_scale_baseline,
        )
    else:
        action_dict = dict(_decode_action(agent, action))
    action_dict["epochs"] = int(stage.action_epochs)

    if stage.freeze_scales:
        action_dict["s_v"] = np.ones(7, dtype=np.float32)
    else:
        scales = np.asarray(action_dict.get("s_v", np.ones(7, dtype=np.float32)), dtype=np.float32).reshape(-1)[:7]
        if scales.size < 7:
            scales = np.pad(scales, (0, 7 - scales.size), constant_values=1.0)
        action_dict["s_v"] = np.clip(scales, 0.0, 1.0).astype(np.float32)

    if isinstance(agent, PinnAgent) and stage.freeze_scales:
        frozen_action = list(action) if isinstance(action, (list, tuple, np.ndarray)) else [0] * len(agent.action_dims)
        if len(frozen_action) < len(agent.action_dims):
            frozen_action.extend([0] * (len(agent.action_dims) - len(frozen_action)))
        if len(frozen_action) > 3:
            frozen_index = len(agent.scale_options) - 1
            frozen_action[3:] = [frozen_index] * (len(frozen_action) - 3)
        return frozen_action, action_dict

    if isinstance(agent, PinnAgent):
        return _normalize_action_for_buffer(agent, action), action_dict

    return _normalize_action_for_buffer(agent, action_dict), action_dict


class ProjectRunner:
    def __init__(
        self,
        strategy: str,
        agent: Any,
        device: torch.device,
        cinns_path: str,
        state_shape: Tuple[int, int, int],
        n_step_horizon: int = 1,
        frontier_relative_actions: bool = False,
        num_workers: int = 1,
        worker_devices: Optional[Sequence[torch.device]] = None,
    ) -> None:
        self.strategy = strategy
        self.agent = agent
        self.device = torch.device(device)
        self.cinns_path = str(cinns_path)
        self.state_shape = tuple(state_shape)
        self.n_step_horizon = int(max(1, n_step_horizon))
        self.frontier_relative_actions = bool(frontier_relative_actions)
        self.num_workers = int(num_workers)
        self.worker_devices = list(worker_devices) if worker_devices is not None else None
        self.env = TorchHoloEnv(cinns_path=self.cinns_path, device=self.device) if self.num_workers <= 1 else None

    def _buffer(self):
        return _agent_buffer(self.agent)

    def _buffer_len(self) -> int:
        return len(self._buffer())

    def _set_agent_device(self, device: torch.device) -> None:
        self.device = torch.device(device)
        if hasattr(self.agent, "to_device"):
            self.agent.to_device(self.device)
        if self.env is not None:
            self.env.to_device(self.device)

    def _snapshot_agent(self, snapshot_path: Path) -> None:
        if not hasattr(self.agent, "save"):
            raise AttributeError(f"Agent {type(self.agent).__name__} does not support save()")
        self.agent.save(str(snapshot_path))

    def build_frontier_start_state(
        self,
        stages: Sequence[CurriculumStage],
        output_path: Path,
        *,
        total_cycles: Optional[int] = None,
    ) -> None:
        env = TorchHoloEnv(cinns_path=self.cinns_path, device=self.device)
        old_eps = getattr(self.agent, "epsilon", None)
        if old_eps is not None:
            self.agent.epsilon = 0.0
        try:
            state = env.reset()
            state_np = state.detach().cpu().numpy()
            total_cycle_count = int(total_cycles or len(stages) or 1)
            for stage_index, stage in enumerate(stages):
                frontier_scale_baseline = (
                    _frontier_cycle_scale_baseline(stage_index) if self.frontier_relative_actions else None
                )
                state_np, _ = _run_episode_steps(
                    env,
                    self.agent,
                    stage,
                    stage.steps_per_episode,
                    eval_mode=True,
                    collect_transitions=False,
                    state_np=state_np,
                    cycle_index=stage_index,
                    total_cycles=total_cycle_count,
                    frontier_relative_actions=self.frontier_relative_actions,
                    frontier_scale_baseline=frontier_scale_baseline,
                )
            _save_env_state(env, output_path, include_metrics=False)
        finally:
            if old_eps is not None:
                self.agent.epsilon = old_eps

    def _switch_on_oom(self) -> bool:
        for next_device in get_available_devices():
            if next_device == self.device:
                continue
            try:
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                self._set_agent_device(next_device)
                logger.info(f"Switched runner to {next_device} after OOM.")
                return True
            except Exception as exc:
                logger.warning(f"Failed to switch to {next_device}: {exc}")
        return False

    def _collect_sequential(
        self,
        target_memories: int,
        steps_per_episode: int,
        epsilon: float,
        eval_mode: bool,
        stage: CurriculumStage,
        cycle_index: int,
        total_cycles: int,
        start_env_state_path: Optional[str] = None,
    ) -> None:
        if self.env is None:
            self.env = TorchHoloEnv(cinns_path=self.cinns_path, device=self.device)

        if hasattr(self.agent, "epsilon"):
            self.agent.epsilon = 0.0 if eval_mode else float(epsilon)
        _set_agent_epsilon_schedule(self.agent, stage)

        ep = 0
        total_steps = 0

        while total_steps < target_memories:
            try:
                if start_env_state_path:
                    _load_env_state(self.env, Path(start_env_state_path))
                    state = self.env.get_residuals()
                else:
                    state = self.env.reset()
                state_np = state.detach().cpu().numpy()
                frontier_scale_baseline = (
                    _frontier_cycle_scale_baseline(cycle_index) if self.frontier_relative_actions else None
                )
                _, episode_memories = _run_episode_steps(
                    self.env,
                    self.agent,
                    stage,
                    steps_per_episode,
                    eval_mode=eval_mode,
                    collect_transitions=True,
                    state_np=state_np,
                    cycle_index=cycle_index,
                    total_cycles=total_cycles,
                    n_step_horizon=self.n_step_horizon,
                    frontier_relative_actions=self.frontier_relative_actions,
                    frontier_scale_baseline=frontier_scale_baseline,
                )
                remaining = max(0, target_memories - total_steps)
                for transition in episode_memories[:remaining]:
                    _push_transition(self.agent, transition)
                    total_steps += 1

                ep += 1
                if ep % 5 == 0:
                    logger.info(f"Collected {total_steps}/{target_memories} memories (sequential).")
            except torch.cuda.OutOfMemoryError:
                logger.error(f"OOM on {self.device} during collection.")
                if not self._switch_on_oom():
                    raise

    def _collect_parallel(
        self,
        target_memories: int,
        steps_per_episode: int,
        epsilon: float,
        eval_mode: bool,
        stage: CurriculumStage,
        cycle_index: int,
        total_cycles: int,
        start_env_state_path: Optional[str] = None,
    ) -> None:
        if hasattr(self.agent, "epsilon"):
            self.agent.epsilon = 0.0 if eval_mode else float(epsilon)
        _set_agent_epsilon_schedule(self.agent, stage)
        num_episodes = max(1, (target_memories + steps_per_episode - 1) // steps_per_episode)

        with tempfile.TemporaryDirectory(prefix="rlpinn_project_snapshot_") as tmpdir:
            snapshot_path = Path(tmpdir) / "agent_snapshot.pt"
            spool_dir = Path(tmpdir) / "episodes"
            spool_dir.mkdir(parents=True, exist_ok=True)
            self._snapshot_agent(snapshot_path)
            loaded_memories = 0

            def _build_worker_args() -> List[Tuple[str, str, str, str, Tuple[int, int, int], int, int, int, int, float, bool, CurriculumStage, str, bool, Optional[str]]]:
                available_devices = _resolve_parallel_devices(self.device, self.worker_devices)
                if all(device.type == "cpu" for device in available_devices):
                    weights = [1.0] * len(available_devices)
                else:
                    weights = get_cuda_memory_weights()
                    if len(weights) != len(available_devices):
                        weights = [1.0] * len(available_devices)

                total_w = float(sum(weights)) if weights else 1.0
                episodes_per_dev = [int((w / total_w) * num_episodes) for w in weights]
                remaining = num_episodes - sum(episodes_per_dev)
                for idx in range(remaining):
                    episodes_per_dev[idx % len(available_devices)] += 1

                device_assignment: List[torch.device] = []
                for dev_idx, count in enumerate(episodes_per_dev):
                    device_assignment.extend([available_devices[dev_idx]] * count)

                return [
                    (
                        self.strategy,
                        str(snapshot_path),
                        self.cinns_path,
                        str(device_assignment[i]),
                        self.state_shape,
                        self.n_step_horizon,
                        cycle_index,
                        total_cycles,
                        steps_per_episode,
                        epsilon,
                        eval_mode,
                        stage,
                        str(spool_dir),
                        self.frontier_relative_actions,
                        start_env_state_path,
                    )
                    for i in range(num_episodes)
                ]

            success = False
            comm_retry_count = 0
            while not success:
                try:
                    worker_args = _build_worker_args()
                    _mps_available = torch.backends.mps.is_available()
                    use_ray = self.device.type in ("cpu", "cuda") and not _mps_available
                    
                    if use_ray:
                        try:
                            import ray
                        except ImportError:
                            raise ImportError("Please install ray via 'pip install ray' to use CPU/CUDA parallel collection.")
                        
                        if not ray.is_initialized():
                            # Let Ray see all available logical CPUs so parallel workers
                            # can saturate the machine when requested.
                            ray.init(ignore_reinit_error=True, num_cpus=max(1, int(os.cpu_count() or 1)))
                        
                        @ray.remote(num_cpus=1)
                        def ray_worker_episode_to_disk(args):
                            import os
                            os.environ["OMP_NUM_THREADS"] = "1"
                            os.environ["MKL_NUM_THREADS"] = "1"
                            os.environ["OPENBLAS_NUM_THREADS"] = "1"
                            return _collect_strategy_episode_to_disk(args)

                        futures = [ray_worker_episode_to_disk.remote(arg) for arg in worker_args]
                        unready = futures
                        with tqdm(total=num_episodes, desc="Ray Parallel Collection") as pbar:
                            while unready:
                                ready, unready = ray.wait(unready, num_returns=1)
                                episode_file = ray.get(ready[0])
                                episode_path = Path(episode_file)
                                try:
                                    import pickle
                                    with episode_path.open("rb") as fh:
                                        episode_memories = pickle.load(fh)
                                    remaining = max(0, target_memories - loaded_memories)
                                    if remaining <= 0:
                                        episode_memories = []
                                    else:
                                        episode_memories = episode_memories[:remaining]
                                    for transition in episode_memories:
                                        _push_transition(self.agent, transition)
                                    loaded_memories += len(episode_memories)
                                finally:
                                    try:
                                        episode_path.unlink(missing_ok=True)
                                    except Exception:
                                        pass
                                pbar.update(1)
                    else:
                        with mp.Pool(processes=self.num_workers, maxtasksperchild=1) as pool:
                            for episode_file in tqdm(
                                pool.imap(_collect_strategy_episode_to_disk, worker_args),
                                total=num_episodes,
                                desc="Native Parallel Collection",
                            ):
                                episode_path = Path(episode_file)
                                try:
                                    import pickle
                                    with episode_path.open("rb") as fh:
                                        episode_memories = pickle.load(fh)
                                    remaining = max(0, target_memories - loaded_memories)
                                    if remaining <= 0:
                                        episode_memories = []
                                    else:
                                        episode_memories = episode_memories[:remaining]
                                    for transition in episode_memories:
                                        _push_transition(self.agent, transition)
                                    loaded_memories += len(episode_memories)
                                finally:
                                    try:
                                        episode_path.unlink(missing_ok=True)
                                    except Exception:
                                        pass

                    success = True
                except Exception as exc:
                    for staged in spool_dir.glob("episode_*.pkl"):
                        try:
                            staged.unlink(missing_ok=True)
                        except Exception:
                            pass

                    message = str(exc)
                    if "OutOfMemoryError" in message or "CUDA out of memory" in message:
                        logger.error(f"OOM during parallel collection on {self.device}.")
                        if not self._switch_on_oom():
                            raise
                        if hasattr(self.agent, "epsilon"):
                            self.agent.epsilon = 0.0 if eval_mode else float(epsilon)
                        self._snapshot_agent(snapshot_path)
                        comm_retry_count = 0
                        continue

                    pipe_error_tokens = (
                        "BrokenPipeError",
                        "EOFError",
                        "ConnectionResetError",
                        "ConnectionAbortedError",
                    )
                    is_pipe_error = isinstance(
                        exc,
                        (BrokenPipeError, EOFError, ConnectionResetError, ConnectionAbortedError),
                    ) or any(token in message for token in pipe_error_tokens)

                    if is_pipe_error and comm_retry_count < 3:
                        comm_retry_count += 1
                        logger.warning(
                            "Worker communication error during parallel collection "
                            f"(attempt {comm_retry_count}/3): {exc}"
                        )
                        self._snapshot_agent(snapshot_path)
                        continue
                    else:
                        raise

    def collect_experience(
        self,
        target_memories: int,
        steps_per_episode: int = DEFAULT_STEPS,
        epsilon: float = 1.0,
        eval_mode: bool = False,
        stage: Optional[CurriculumStage] = None,
        cycle_index: int = 0,
        total_cycles: int = 1,
        save_dir: Optional[str] = None,
        clear_after_save: bool = False,
        start_env_state_path: Optional[str] = None,
        trim_max_size: int = REPLAY_BUFFER_MAX,
        replay_save_path: Optional[str] = None,
    ) -> None:
        if stage is None:
            raise ValueError("A curriculum stage is required for experience collection")
        num_episodes = max(1, (target_memories + steps_per_episode - 1) // steps_per_episode)
        logger.info(
            f"Collecting {target_memories} memories (~{num_episodes} episode(s) * {steps_per_episode} steps) "
            f"| Workers: {self.num_workers} | Mode: {'greedy' if eval_mode else 'explore'} "
            f"| Epsilon: {0.0 if eval_mode else epsilon:.2f}"
        )

        if self.num_workers > 1:
            self._collect_parallel(
                target_memories,
                steps_per_episode,
                epsilon,
                eval_mode,
                stage,
                cycle_index,
                total_cycles,
                start_env_state_path=start_env_state_path,
            )
        else:
            self._collect_sequential(
                target_memories,
                steps_per_episode,
                epsilon,
                eval_mode,
                stage,
                cycle_index,
                total_cycles,
                start_env_state_path=start_env_state_path,
            )

        _trim_agent_buffer(self.agent, trim_max_size)
        if trim_max_size > 0:
            logger.info(f"Replay buffer size after trim: {self._buffer_len()} / {trim_max_size}")
        else:
            logger.info(f"Replay buffer size after collection: {self._buffer_len()}")

        target_path: Optional[Path] = None
        if replay_save_path:
            target_path = Path(replay_save_path)
        elif save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            target_path = save_path / "memories.pkl"

        if target_path is not None:
            try:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                _agent_buffer(self.agent).save_to_disk(str(target_path))
                logger.info(f"Saved replay buffer to {target_path}")
                if clear_after_save:
                    _agent_buffer(self.agent).clear()
                    logger.info("Cleared replay buffer after serialization.")
            except Exception as exc:
                logger.error(f"Failed to save replay buffer: {exc}")

    def train_agent(self, epochs: int, batch_size: Optional[int] = None, patience: int = DEFAULT_PATIENCE, min_delta: float = DEFAULT_MIN_DELTA) -> None:
        buffer_len = self._buffer_len()
        if buffer_len <= 0:
            logger.warning("Not enough memory to train. Skipping.")
            return

        batch_size = int(batch_size or getattr(self.agent, "batch_size", 64))
        batch_size = max(1, min(batch_size, buffer_len))
        logger.info(f"Training strategy {self.strategy} for {epochs} epochs on {self.device}...")
        pbar = tqdm(range(epochs), desc="Training")

        recent_losses: List[float] = []
        best_loss = float("inf")
        patience_counter = 0
        total_loss = 0.0
        count = 0

        for epoch in pbar:
            try:
                loss = self.agent.train_step(batch_size=batch_size)
                if loss is None:
                    continue
                loss = float(loss)
                if not np.isfinite(loss):
                    logger.warning(f"Skipping non-finite loss at epoch {epoch + 1}: {loss}")
                    continue

                total_loss += loss
                count += 1
                recent_losses.append(loss)
                if len(recent_losses) > patience:
                    recent_losses.pop(0)

                if len(recent_losses) == patience and patience > 0:
                    moving_avg = sum(recent_losses) / patience
                    if moving_avg < best_loss - min_delta:
                        best_loss = moving_avg
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    if patience_counter >= patience:
                        logger.info(f"Early stopping triggered at epoch {epoch + 1}.")
                        pbar.close()
                        break

                pbar.set_postfix({"loss": f"{loss:.4f}"})
                if self.strategy == "default" and (epoch + 1) % 30 == 0 and hasattr(self.agent, "update_target_net"):
                    self.agent.update_target_net()
            except torch.cuda.OutOfMemoryError:
                logger.error(f"OOM on {self.device} during training.")
                if not self._switch_on_oom():
                    raise
                continue

        avg_loss = total_loss / count if count else 0.0
        logger.info(f"Training complete. Avg Loss: {avg_loss:.4f}")

    def render(
        self,
        save_dir: str,
        validation_steps: int = DEFAULT_STEPS,
        stage: Optional[CurriculumStage] = None,
        frontier_relative_actions_override: Optional[bool] = None,
    ) -> Dict[str, Any]:
        val_env = self.env
        if val_env is None:
            val_env = TorchHoloEnv(cinns_path=self.cinns_path, device=self.device)

        old_eps = getattr(self.agent, "epsilon", None)
        if old_eps is not None:
            self.agent.epsilon = 0.0

        try:
            state = val_env.reset()
            state_np = state.detach().cpu().numpy()
            use_relative_actions = (
                self.frontier_relative_actions
                if frontier_relative_actions_override is None
                else bool(frontier_relative_actions_override)
            )
            for _ in range(validation_steps):
                action = self.agent.act(state_np, eval_mode=True)
                if stage is None:
                    action_dict = _decode_action(self.agent, action)
                else:
                    _, action_dict = _prepare_stage_action(
                        self.agent,
                        action,
                        stage,
                        env=val_env if use_relative_actions else None,
                        frontier_relative=use_relative_actions,
                    )
                residuals, _, _ = val_env.step(action_dict)
                state_np = residuals.detach().cpu().numpy()
            report = val_env.render(save_dir)
            logger.info("Validation rendering successful.")
            return report
        finally:
            if old_eps is not None:
                self.agent.epsilon = old_eps

    def render_schedule(
        self,
        save_dir: str,
        stages: Sequence[CurriculumStage],
        *,
        total_cycles: Optional[int] = None,
        frontier_relative_actions_override: Optional[bool] = None,
    ) -> Dict[str, Any]:
        val_env = TorchHoloEnv(cinns_path=self.cinns_path, device=self.device)
        old_eps = getattr(self.agent, "epsilon", None)
        if old_eps is not None:
            self.agent.epsilon = 0.0

        try:
            state = val_env.reset()
            state_np = state.detach().cpu().numpy()
            total_cycle_count = int(total_cycles or len(stages) or 1)
            use_relative_actions = (
                self.frontier_relative_actions
                if frontier_relative_actions_override is None
                else bool(frontier_relative_actions_override)
            )
            for stage_index, stage in enumerate(stages):
                frontier_scale_baseline = (
                    _frontier_cycle_scale_baseline(stage_index) if use_relative_actions else None
                )
                state_np, _ = _run_episode_steps(
                    val_env,
                    self.agent,
                    stage,
                    stage.steps_per_episode,
                    eval_mode=True,
                    collect_transitions=False,
                    state_np=state_np,
                    cycle_index=stage_index,
                    total_cycles=total_cycle_count,
                    frontier_relative_actions=use_relative_actions,
                    frontier_scale_baseline=frontier_scale_baseline,
                )
            report = val_env.render(save_dir)
            logger.info("Schedule rendering successful.")
            return report
        finally:
            if old_eps is not None:
                self.agent.epsilon = old_eps

    def render_frontier_unconditional(
        self,
        save_dir: str,
        *,
        validation_steps: int = PROJECT_RENDER_DECISIONS,
        action_epochs: int = PROJECT_RENDER_ACTION_EPOCHS,
    ) -> Dict[str, Any]:
        val_env = TorchHoloEnv(cinns_path=self.cinns_path, device=self.device)
        old_eps = getattr(self.agent, "epsilon", None)
        if old_eps is not None:
            self.agent.epsilon = 0.0

        try:
            stage = _build_flat_frontier_render_stage(
                steps_per_episode=validation_steps,
                action_epochs=action_epochs,
            )
            fixed_meta = _frontier_state_meta(
                stage,
                cycle_index=0,
                total_cycles=1,
                decision_index=0,
                total_decisions=1,
            )
            state = val_env.reset()
            raw_state_np = state.detach().cpu().numpy()
            for _ in range(validation_steps):
                state_with_context = _augment_state_with_frontier_context(raw_state_np, fixed_meta)
                action = self.agent.act(state_with_context, eval_mode=True)
                _, action_dict = _prepare_stage_action(
                    self.agent,
                    action,
                    stage,
                    env=None,
                    frontier_relative=False,
                )
                residuals, _, _ = val_env.step(action_dict)
                raw_state_np = residuals.detach().cpu().numpy()
            report = val_env.render(save_dir)
            logger.info("Unconditional frontier rendering successful.")
            return report
        finally:
            if old_eps is not None:
                self.agent.epsilon = old_eps


def _render_checkpoint_worker(
    strategy: str,
    checkpoint_path: str,
    cinns_path: str,
    device_name: str,
    state_shape: Tuple[int, int, int],
    n_step_horizon: int,
    save_dir: str,
    frontier_relative_actions: bool,
    stages: Optional[Sequence[CurriculumStage]] = None,
    total_cycles: Optional[int] = None,
    validation_steps: int = DEFAULT_STEPS,
    stage: Optional[CurriculumStage] = None,
) -> Dict[str, Any]:
    device = torch.device(device_name)
    agent = _build_agent_with_shape(
        strategy,
        device,
        state_shape=state_shape,
        n_steps=n_step_horizon,
    )
    agent.load(checkpoint_path, load_memory=False)
    runner = ProjectRunner(
        strategy=strategy,
        agent=agent,
        device=device,
        cinns_path=cinns_path,
        state_shape=state_shape,
        n_step_horizon=n_step_horizon,
        frontier_relative_actions=frontier_relative_actions,
        num_workers=1,
        worker_devices=[device] if device.type == "cpu" else None,
    )
    if frontier_relative_actions:
        return runner.render_frontier_unconditional(
            save_dir,
            validation_steps=PROJECT_RENDER_DECISIONS,
            action_epochs=PROJECT_RENDER_ACTION_EPOCHS,
        )
    if stages is not None:
        return runner.render_schedule(
            save_dir,
            stages,
            total_cycles=total_cycles,
            frontier_relative_actions_override=False,
        )
    return runner.render(
        save_dir,
        validation_steps=validation_steps,
        stage=stage,
        frontier_relative_actions_override=False,
    )


def _drain_render_futures(
    pending: List[Tuple[int, Future[Dict[str, Any]]]],
    *,
    wait: bool = False,
) -> Tuple[List[Tuple[int, Future[Dict[str, Any]]]], Dict[int, Dict[str, Any]]]:
    remaining: List[Tuple[int, Future[Dict[str, Any]]]] = []
    completed: Dict[int, Dict[str, Any]] = {}
    for cycle_number, future in pending:
        if not wait and not future.done():
            remaining.append((cycle_number, future))
            continue
        try:
            result = future.result()
            if isinstance(result, dict):
                completed[cycle_number] = result
            logger.info(f"Background render complete for cycle {cycle_number}.")
        except Exception as exc:
            logger.warning(f"Background render failed for cycle {cycle_number}: {exc}")
    return remaining, completed


def _log_shape_gate_result(cycle_number: int, gate_report: Dict[str, Any]) -> None:
    current = gate_report.get("current") or {}
    best = gate_report.get("best") or {}
    logger.info(
        "Cycle %s shape gate | pass=%s | current(corr=%.3f drop=%.3f rebound=%.3f min_frac=%.3f) | "
        "best(corr=%s drop=%s rebound=%s min_frac=%s)",
        cycle_number,
        gate_report.get("pass"),
        float(current.get("corr_to_theory", 0.0)),
        float(current.get("drop_from_start", 0.0)),
        float(current.get("rebound_from_min", 0.0)),
        float(current.get("min_fraction", 0.0)),
        f"{float(best.get('corr_to_theory', 0.0)):.3f}" if best else "n/a",
        f"{float(best.get('drop_from_start', 0.0)):.3f}" if best else "n/a",
        f"{float(best.get('rebound_from_min', 0.0)):.3f}" if best else "n/a",
        f"{float(best.get('min_fraction', 0.0)):.3f}" if best else "n/a",
    )


def _benchmark_device_type(device_type: str, cinns_path: str, steps: int) -> Optional[float]:
    run_py = ROOT_DIR / "run.py"
    cmd_template = [
        sys.executable,
        str(run_py),
        "--device",
        device_type,
        "--workers",
        "1",
        "--cycles",
        "1",
        "--target_memories",
        str(max(steps, 1)),
        "--steps",
        str(steps),
        "--train_epochs",
        "1",
        "--patience",
        "1",
        "--cinns_path",
        str(Path(cinns_path).resolve()),
        "--skip_render",
        "--skip_device_scan",
    ]

    start = time.perf_counter()
    try:
        with tempfile.TemporaryDirectory(prefix=f"rlpinn_bench_{device_type}_") as tmpdir:
            cmd = [*cmd_template, "--log_dir", str(Path(tmpdir))]
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout, stderr = process.communicate()
            elapsed = time.perf_counter() - start
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, cmd, output=stdout, stderr=stderr)
        throughput = 60.0 / max(elapsed, 1e-6)
        logger.info(f"Benchmark {device_type}: {throughput:.2f} episodes/min")
        return throughput
    except Exception as exc:
        logger.warning(f"Benchmark failed for {device_type}: {exc}")
        return None


def auto_select_device_and_workers(cinns_path: str, mem_limit: float, target_memories: int, steps: int) -> Tuple[torch.device, int, int]:
    print("\nRunning hardware benchmark to choose the best device...")
    benchmark_steps = _curriculum_scan_steps(steps)
    candidate_types = _candidate_device_types()
    results: List[Tuple[str, float]] = []
    for device_type in candidate_types:
        throughput = _benchmark_device_type(device_type, cinns_path, benchmark_steps)
        if throughput is not None:
            results.append((device_type, throughput))

    if not results:
        chosen_type = "cpu"
    else:
        chosen_type = max(results, key=lambda item: item[1])[0]
    device = _build_device(chosen_type)
    print(f"Auto-selected device: {_device_label(device)}")

    workers, recommended_episodes = perform_scan(
        cinns_path=cinns_path,
        mem_limit=mem_limit,
        device=chosen_type,
        target_memories=target_memories,
        steps=benchmark_steps,
    )
    return device, workers, recommended_episodes


def resolve_runtime_selection(args: argparse.Namespace) -> RuntimeConfig:
    cinns_path = str(Path(args.cinns_path).resolve()) if args.cinns_path else str(ROOT_DIR)
    log_dir = str(Path(args.log_dir).resolve()) if args.log_dir else str(PROJECT_LOG_ROOT)
    strategy = args.strategy if args.strategy in SUPPORTED_STRATEGIES else None
    frontier_curriculum = bool(args.frontier_curriculum)
    frontier_epsilon = float(args.frontier_epsilon)
    frontier_memory_schedule = _parse_frontier_memory_schedule(args.frontier_memory_schedule)
    frontier_epsilon_schedule = _parse_frontier_epsilon_schedule(
        args.frontier_epsilon_schedule,
        frontier_epsilon,
    )
    frontier_steps_per_episode = int(args.frontier_steps_per_episode)
    frontier_memory_block = int(args.frontier_memory_block)
    if frontier_steps_per_episode <= 0:
        raise ValueError("--frontier_steps_per_episode must be positive")
    if frontier_memory_block <= 0:
        raise ValueError("--frontier_memory_block must be positive")
    cycles = int(args.cycles)
    if frontier_curriculum and cycles == DEFAULT_CYCLES:
        cycles = len(FRONTIER_SCALE_BASELINE_SCHEDULE)
    frontier_schedule = (
        _build_frontier_stages(
            frontier_epsilon,
            cycles=cycles,
            epsilon_schedule=frontier_epsilon_schedule,
            steps_per_episode=frontier_steps_per_episode,
            freeze_final_scales=not bool(args.frontier_free_scales),
        )
        if frontier_curriculum
        else CURRICULUM_STAGES
    )
    if frontier_curriculum and len(frontier_memory_schedule) < cycles:
        raise ValueError(
            "--frontier_memory_schedule must provide at least one target per frontier cycle "
            f"(got {len(frontier_memory_schedule)} for {cycles} cycles)"
        )
    target_memories = int(args.target_memories)
    if frontier_curriculum and target_memories == DEFAULT_TARGET_MEMORIES:
        target_memories = max(frontier_memory_schedule[:cycles])
    steps = int(args.steps)
    if frontier_curriculum and steps == DEFAULT_STEPS:
        steps = frontier_steps_per_episode
    replay_max_size = int(
        args.replay_max_size
        if args.replay_max_size is not None
        else (_frontier_replay_max_size(cycles, frontier_memory_schedule) if frontier_curriculum else REPLAY_BUFFER_MAX)
    )

    if args.lmac:
        device = torch.device("cpu")
        workers = 8
        train_epochs = 100
        patience = 10
        min_delta = 1e-4
        episodes_per_cycle = max(1, (target_memories + max(steps, 1) - 1) // max(steps, 1))
        return RuntimeConfig(
            strategy=strategy or _prompt_strategy(),
            device=device,
            workers=workers,
            episodes_per_cycle=episodes_per_cycle,
            cycles=cycles,
            target_memories=target_memories,
            train_epochs=train_epochs,
            patience=patience,
            min_delta=min_delta,
            steps=steps,
            mem_limit=args.mem_limit,
            cinns_path=cinns_path,
            log_dir=log_dir,
            skip_render=args.skip_render,
            resume=args.resume,
            frontier_curriculum=frontier_curriculum,
            frontier_epsilon=frontier_epsilon,
            replay_max_size=replay_max_size,
            frontier_single_epsilon_collection=bool(args.single_epsilon_collection),
            frontier_memory_schedule=frontier_memory_schedule,
            frontier_epsilon_schedule=frontier_epsilon_schedule,
            frontier_steps_per_episode=frontier_steps_per_episode,
            frontier_memory_block=frontier_memory_block,
            frontier_free_scales=bool(args.frontier_free_scales),
            cycle_stop_enabled=not args.disable_cycle_stop,
        )

    if args.lpc:
        device = torch.device("cpu")
        workers = 8
        train_epochs = 100
        patience = 10
        min_delta = 1e-4
        episodes_per_cycle = max(1, (target_memories + max(steps, 1) - 1) // max(steps, 1))
        return RuntimeConfig(
            strategy=strategy or _prompt_strategy(),
            device=device,
            workers=workers,
            episodes_per_cycle=episodes_per_cycle,
            cycles=cycles,
            target_memories=target_memories,
            train_epochs=train_epochs,
            patience=patience,
            min_delta=min_delta,
            steps=steps,
            mem_limit=args.mem_limit,
            cinns_path=cinns_path,
            log_dir=log_dir,
            skip_render=args.skip_render,
            resume=args.resume,
            frontier_curriculum=frontier_curriculum,
            frontier_epsilon=frontier_epsilon,
            replay_max_size=replay_max_size,
            frontier_single_epsilon_collection=bool(args.single_epsilon_collection),
            frontier_memory_schedule=frontier_memory_schedule,
            frontier_epsilon_schedule=frontier_epsilon_schedule,
            frontier_steps_per_episode=frontier_steps_per_episode,
            frontier_memory_block=frontier_memory_block,
            frontier_free_scales=bool(args.frontier_free_scales),
            cycle_stop_enabled=not args.disable_cycle_stop,
        )

    if args.auto:
        scan_steps = _curriculum_scan_steps(steps, frontier_schedule if frontier_curriculum else None)
        device, workers, episodes_per_cycle = auto_select_device_and_workers(
            cinns_path=cinns_path,
            mem_limit=args.mem_limit,
            target_memories=target_memories,
            steps=scan_steps,
        )
        return RuntimeConfig(
            strategy=strategy or _prompt_strategy(),
            device=device,
            workers=workers,
            episodes_per_cycle=episodes_per_cycle,
            cycles=cycles,
            target_memories=target_memories,
            train_epochs=args.train_epochs,
            patience=args.patience,
            min_delta=args.min_delta,
            steps=steps,
            mem_limit=args.mem_limit,
            cinns_path=cinns_path,
            log_dir=log_dir,
            skip_render=args.skip_render,
            resume=args.resume,
            frontier_curriculum=frontier_curriculum,
            frontier_epsilon=frontier_epsilon,
            replay_max_size=replay_max_size,
            frontier_single_epsilon_collection=bool(args.single_epsilon_collection),
            frontier_memory_schedule=frontier_memory_schedule,
            frontier_epsilon_schedule=frontier_epsilon_schedule,
            frontier_steps_per_episode=frontier_steps_per_episode,
            frontier_memory_block=frontier_memory_block,
            frontier_free_scales=bool(args.frontier_free_scales),
            cycle_stop_enabled=not args.disable_cycle_stop,
        )

    device = _parse_device_choice(args.device)
    if device is None:
        device = _prompt_device()

    workers, auto_workers = _parse_workers_choice(args.workers)
    if workers is None and not auto_workers:
        workers, auto_workers = _prompt_workers()

    if auto_workers:
        scan_steps = _curriculum_scan_steps(steps, frontier_schedule if frontier_curriculum else None)
        workers, episodes_per_cycle = perform_scan(
            cinns_path=cinns_path,
            mem_limit=args.mem_limit,
            device=_normalize_device_name(str(device)),
            target_memories=target_memories,
            steps=scan_steps,
        )
    else:
        episodes_per_cycle = max(1, round((target_memories // max(steps, 1)) / workers) * workers)
        if episodes_per_cycle == 0:
            episodes_per_cycle = workers

    return RuntimeConfig(
        strategy=strategy or _prompt_strategy(),
        device=device,
        workers=workers,
        episodes_per_cycle=episodes_per_cycle,
        cycles=cycles,
        target_memories=target_memories,
        train_epochs=args.train_epochs,
        patience=args.patience,
        min_delta=args.min_delta,
        steps=steps,
        mem_limit=args.mem_limit,
        cinns_path=cinns_path,
        log_dir=log_dir,
        skip_render=args.skip_render,
        resume=args.resume,
        frontier_curriculum=frontier_curriculum,
        frontier_epsilon=frontier_epsilon,
        replay_max_size=replay_max_size,
        frontier_single_epsilon_collection=bool(args.single_epsilon_collection),
        frontier_memory_schedule=frontier_memory_schedule,
        frontier_epsilon_schedule=frontier_epsilon_schedule,
        frontier_steps_per_episode=frontier_steps_per_episode,
        frontier_memory_block=frontier_memory_block,
        frontier_free_scales=bool(args.frontier_free_scales),
        cycle_stop_enabled=not args.disable_cycle_stop,
    )


def _configure_logging(run_dir: Path) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(run_dir / "training.log"),
        ],
    )


def _send_notifyme(title: str, body: str) -> None:
    if os.environ.get("RLPINN_SKIP_NOTIFYME") == "1":
        return
    api_key = os.environ.get("NOTIFYME_API_KEY", "").strip()
    endpoint = os.environ.get("NOTIFYME_ENDPOINT", NOTIFYME_ENDPOINT).strip()
    if not api_key or not endpoint:
        return

    payload = json.dumps(
        {
            "title": title,
            "body": body,
            "actionType": "open_history",
        }
    ).encode("utf-8")
    req = request.Request(
        endpoint,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=20):
            return
    except error.URLError as exc:
        logger.warning(f"NotifyMe send failed: {exc}")


def _task_label(config: RuntimeConfig) -> str:
    return f"Strategy run ({config.strategy})"


def _cycle_task_label(config: RuntimeConfig, cycle_number: int) -> str:
    return f"Strategy run ({config.strategy}) cycle {cycle_number}/{config.cycles}"


def _success_notification(config: RuntimeConfig) -> Tuple[str, str]:
    title = f"{PROJECT_NAME} - Task Completion"
    body = "\n".join(
        [
            f"Project: {PROJECT_NAME}",
            f"Task: {_task_label(config)}",
            "Status: Success",
            (
                "Summary: Training completed successfully with curriculum stages, "
                f"{config.cycles} cycle(s), strategy={config.strategy}, log_dir={config.log_dir}."
            ),
        ]
    )
    return title, body


def _failure_notification(config: Optional[RuntimeConfig], exc: BaseException) -> Tuple[str, str]:
    task = _task_label(config) if config is not None else "Curriculum training run"
    title = f"{PROJECT_NAME} - Task Failed"
    body = "\n".join(
        [
            f"Project: {PROJECT_NAME}",
            f"Task: {task}",
            "Status: Failed",
            f"Error: {type(exc).__name__}: {exc}",
        ]
    )
    return title, body


def _cleanup_memory_files(run_dir: Path, cycles: int) -> None:
    logger.info("Cleaning up memory buffer files to free up disk space...")
    for cycle_number in range(1, cycles + 1):
        mem_file = run_dir / f"cycle_{cycle_number}" / "memories.pkl"
        if not mem_file.exists():
            continue
        try:
            mem_file.unlink()
            logger.info(f"Removed memory file: {mem_file}")
        except Exception as exc:
            logger.warning(f"Could not remove {mem_file}: {exc}")
    frontier_replay = run_dir / FRONTIER_REPLAY_FILENAME
    if frontier_replay.exists():
        try:
            frontier_replay.unlink()
            logger.info(f"Removed memory file: {frontier_replay}")
        except Exception as exc:
            logger.warning(f"Could not remove {frontier_replay}: {exc}")


def run_project(config: RuntimeConfig) -> None:
    run_root = Path(config.log_dir)
    schedule = (
        _build_frontier_stages(
            config.frontier_epsilon,
            cycles=config.cycles,
            epsilon_schedule=config.frontier_epsilon_schedule,
            steps_per_episode=config.frontier_steps_per_episode,
            freeze_final_scales=not config.frontier_free_scales,
        )
        if config.frontier_curriculum
        else CURRICULUM_STAGES
    )
    render_executor: Optional[ProcessPoolExecutor] = None
    pending_renders: List[Tuple[int, Future[Dict[str, Any]]]] = []
    run_completed = False
    cycle_stop_state = CycleStopState()
    
    # --- AUTO-RESUME LOGIC ---
    start_cycle = 0
    run_dir = None
    
    if config.resume:
        resume_path = Path(config.resume)
        if resume_path.exists():
            # If resume points to a specific checkpoint in a run directory
            # e.g., logs/project/run_1/agent_cycle_5.pth
            if "agent_cycle_" in resume_path.name:
                try:
                    start_cycle = int(resume_path.stem.split("_")[-1])
                    run_dir = resume_path.parent
                except ValueError:
                    pass
    
    if run_dir is None:
        if _is_direct_strategy_log_dir(run_root):
            run_dir = run_root
            run_dir.mkdir(parents=True, exist_ok=True)
        else:
            run_dir = _make_run_dir(run_root)
    
    _configure_logging(run_dir)

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    if not config.skip_render:
        render_executor = ProcessPoolExecutor(
            max_workers=1,
            mp_context=mp.get_context("spawn"),
        )

    logger.info(
        f"Starting strategy={config.strategy} | device={config.device} | workers={config.workers} "
        f"| episodes/cycle={config.episodes_per_cycle} | start_cycle={start_cycle} "
        f"| frontier_curriculum={config.frontier_curriculum} "
        f"| single_epsilon_collection={config.frontier_single_epsilon_collection}"
    )

    if config.cycles > len(schedule):
        raise ValueError(
            f"Requested {config.cycles} cycles but only {len(schedule)} "
            "are defined in the gradual schedule."
        )

    state_shape = _agent_state_shape(config.frontier_curriculum)
    n_step_horizon = DEFAULT_FRONTIER_N_STEP_HORIZON if config.frontier_curriculum else 1
    agent = _build_agent_with_shape(
        config.strategy,
        config.device,
        state_shape=state_shape,
        n_steps=n_step_horizon,
    )
    if config.resume:
        resume_path = Path(config.resume)
        if resume_path.exists() and hasattr(agent, "load"):
            logger.info(f"Resuming from {resume_path}")
            agent.load(str(resume_path), load_memory=False)
            if not config.frontier_curriculum:
                memory_buffer = _agent_buffer(agent)
                if memory_buffer is not None and start_cycle > 0:
                    logger.info("Reloading historical memories into buffer...")
                    mem_file = run_dir / f"cycle_{start_cycle}" / "memories.pkl"
                    if mem_file.exists() and hasattr(memory_buffer, "load_from_disk"):
                        memory_buffer.load_from_disk(str(mem_file))
                    else:
                        logger.warning(f"Latest memory file {mem_file} not found. Trying fallback loop...")
                        for c in range(start_cycle):
                            m_file = run_dir / f"cycle_{c + 1}" / "memories.pkl"
                            if m_file.exists() and hasattr(memory_buffer, "load_from_disk"):
                                memory_buffer.load_from_disk(str(m_file))
        else:
            logger.warning(f"Resume checkpoint {resume_path} not found or unsupported. Starting fresh.")

    worker_devices = [config.device] if config.device.type == "cpu" else None
    runner = ProjectRunner(
        strategy=config.strategy,
        agent=agent,
        device=config.device,
        cinns_path=config.cinns_path,
        state_shape=state_shape,
        n_step_horizon=n_step_horizon,
        frontier_relative_actions=config.frontier_curriculum,
        num_workers=config.workers,
        worker_devices=worker_devices,
    )

    try:
        for cycle in range(start_cycle, config.cycles):
            cycle_dir = run_dir / f"cycle_{cycle + 1}"
            stage = _curriculum_stage_for_cycle(cycle, schedule=schedule)
            _set_agent_epsilon_schedule(agent, stage)
            agent.epsilon = float(stage.epsilon)

            scale_label = "scale_baseline" if config.frontier_curriculum else "scale_floor"
            logger.info(
                f"=== Cycle {cycle + 1}/{config.cycles} | stage={stage.name} | "
                f"steps/episode={stage.steps_per_episode} | epochs/step={stage.action_epochs} | "
                f"epsilon={stage.epsilon:.2f} | {scale_label}={stage.scale_floor:.2f} | "
                f"freeze_scales={stage.freeze_scales} ==="
            )

            target_memories = (
                _frontier_cycle_target_memories(cycle, config.frontier_memory_schedule)
                if config.frontier_curriculum
                else int(config.target_memories)
            )
            episodes = max(1, (target_memories + stage.steps_per_episode - 1) // stage.steps_per_episode)
            start_env_state_path: Optional[str] = None
            if config.frontier_curriculum and cycle > 0:
                frontier_start_path = cycle_dir / "frontier_start_env.pth"
                logger.info(f"Preparing frontier boundary state for cycle {cycle + 1}...")
                runner.build_frontier_start_state(
                    schedule[:cycle],
                    frontier_start_path,
                    total_cycles=config.cycles,
                )
                start_env_state_path = str(frontier_start_path)

            logger.info(
                f"Collected target: {target_memories} memories | "
                f"episodes: {episodes} -> up to {episodes * stage.steps_per_episode} candidate memories"
            )

            if config.frontier_curriculum:
                _clear_agent_buffer(agent)
                if config.frontier_single_epsilon_collection:
                    logger.info(
                        f"Frontier collection: single epsilon phase ({stage.epsilon:.2f}) "
                        f"for {target_memories} memories"
                    )
                    runner.collect_experience(
                        target_memories=target_memories,
                        steps_per_episode=stage.steps_per_episode,
                        epsilon=stage.epsilon,
                        eval_mode=False,
                        stage=stage,
                        cycle_index=cycle,
                        total_cycles=config.cycles,
                        save_dir=str(cycle_dir),
                        clear_after_save=False,
                        start_env_state_path=start_env_state_path,
                        trim_max_size=target_memories,
                        replay_save_path=None,
                    )
                else:
                    phases = _frontier_collection_plan(
                        cycle,
                        stage.epsilon,
                        schedule=config.frontier_memory_schedule,
                        memory_block=config.frontier_memory_block,
                    )
                    for phase_index, phase in enumerate(phases, start=1):
                        logger.info(
                            f"Frontier collection phase {phase_index}/{len(phases)} "
                            f"({phase.name}): {phase.target_memories} memories"
                        )
                        runner.collect_experience(
                            target_memories=phase.target_memories,
                            steps_per_episode=stage.steps_per_episode,
                            epsilon=phase.epsilon,
                            eval_mode=phase.eval_mode,
                            stage=stage,
                            cycle_index=cycle,
                            total_cycles=config.cycles,
                            save_dir=str(cycle_dir) if phase_index == len(phases) else None,
                            clear_after_save=False,
                            start_env_state_path=start_env_state_path,
                            trim_max_size=target_memories,
                            replay_save_path=None,
                        )
            else:
                runner.collect_experience(
                    target_memories=target_memories,
                    steps_per_episode=stage.steps_per_episode,
                    epsilon=stage.epsilon,
                    eval_mode=False,
                    stage=stage,
                    cycle_index=cycle,
                    total_cycles=config.cycles,
                    save_dir=str(cycle_dir),
                    clear_after_save=False,
                    start_env_state_path=start_env_state_path,
                    trim_max_size=config.replay_max_size,
                    replay_save_path=None,
                )

            runner.train_agent(
                epochs=config.train_epochs,
                batch_size=64 if config.strategy == "default" else 32,
                patience=config.patience,
                min_delta=config.min_delta,
            )

            # Restore best weights from training for rendering and next cycle
            if hasattr(agent, "restore_best_weights"):
                agent.restore_best_weights()

            # Reset training state for next cycle (best weights tracking, etc.)
            if hasattr(agent, "reset_training_state"):
                agent.reset_training_state()

            checkpoint_path = run_dir / f"agent_cycle_{cycle + 1}.pth"
            agent.save(str(checkpoint_path))
            logger.info(f"Saved checkpoint to {checkpoint_path}")

            if render_executor is not None:
                pending_renders, _ = _drain_render_futures(pending_renders)
                logger.info(
                    f"Queueing background render for cycle {cycle + 1} outputs to {cycle_dir}."
                )
                if config.frontier_curriculum:
                    future = render_executor.submit(
                        _render_checkpoint_worker,
                        config.strategy,
                        str(checkpoint_path),
                        config.cinns_path,
                        str(config.device),
                        state_shape,
                        n_step_horizon,
                        str(cycle_dir),
                        config.frontier_curriculum,
                        tuple(schedule),
                        config.cycles,
                    )
                else:
                    future = render_executor.submit(
                        _render_checkpoint_worker,
                        config.strategy,
                        str(checkpoint_path),
                        config.cinns_path,
                        str(config.device),
                        state_shape,
                        n_step_horizon,
                        str(cycle_dir),
                        config.frontier_curriculum,
                        None,
                        None,
                        stage.steps_per_episode,
                        stage,
                    )
                pending_renders.append((cycle + 1, future))
                if (
                    config.frontier_curriculum
                    and config.cycle_stop_enabled
                    and (cycle + 1) >= PROJECT_CYCLE_STOP_START
                ):
                    pending_renders, completed_reports = _drain_render_futures(pending_renders, wait=True)
                    cycle_report = completed_reports.get(cycle + 1)
                    if cycle_report is not None:
                        cycle_stop_state, stop_info = _advance_cycle_stop_state(
                            cycle + 1,
                            cycle_report,
                            cycle_stop_state,
                        )
                        _log_cycle_stop_result(cycle + 1, stop_info)
                        if bool(stop_info.get("stop")):
                            logger.info(
                                "Stopping strategy %s after cycle %s because the cycle-loss patience stop fired.",
                                config.strategy,
                                cycle + 1,
                            )
                            break

        logger.info("Project run complete.")
        run_completed = True
    finally:
        if render_executor is not None:
            pending_renders, _ = _drain_render_futures(pending_renders, wait=run_completed)
            render_executor.shutdown(wait=run_completed, cancel_futures=not run_completed)
        _cleanup_memory_files(run_dir, config.cycles)



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive RL-PINN strategy launcher")
    parser.add_argument("--device", type=str, default=None, help="Device to use: cpu, cuda, mps, or cuda:0 style strings")
    parser.add_argument("--workers", type=str, default=None, help="Number of workers or 'auto'")
    parser.add_argument("--strategy", type=str, default=None, choices=SUPPORTED_STRATEGIES, help="Strategy to run")
    parser.add_argument("--cycles", type=int, default=DEFAULT_CYCLES, help="Number of training cycles")
    parser.add_argument("--target_memories", type=int, default=DEFAULT_TARGET_MEMORIES, help="Memories to collect per cycle")
    parser.add_argument("--train_epochs", type=int, default=DEFAULT_TRAIN_EPOCHS, help="Training epochs per cycle")
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE, help="Early stopping patience")
    parser.add_argument("--min_delta", type=float, default=DEFAULT_MIN_DELTA, help="Early stopping min delta")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="Steps per episode")
    parser.add_argument("--auto", action="store_true", help="Auto-select device and worker count")
    parser.add_argument("--lmac", action="store_true", help="Force the local baseline configuration")
    parser.add_argument("--lpc", action="store_true", help="Force the local CPU profile configuration (cpu, 6 workers)")
    parser.add_argument("--mem_limit", type=float, default=DEFAULT_MEM_LIMIT, help="RAM limit for worker scans")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume from")
    parser.add_argument("--cinns_path", type=str, default=None, help="Path containing the CINNS data folder")
    parser.add_argument("--log_dir", type=str, default=None, help="Base directory for logs")
    parser.add_argument("--skip_render", action="store_true", help="Skip rendering plots after each cycle")
    parser.add_argument("--frontier_curriculum", action="store_true", help="Use the 10-cycle 10k-frontier curriculum.")
    parser.add_argument("--frontier_epsilon", type=float, default=DEFAULT_FRONTIER_EPSILON, help="Fixed epsilon for the frontier curriculum.")
    parser.add_argument(
        "--frontier_epsilon_schedule",
        type=str,
        default=None,
        help="Comma-separated epsilon value per frontier cycle; defaults to --frontier_epsilon.",
    )
    parser.add_argument(
        "--frontier_memory_schedule",
        type=str,
        default=None,
        help="Comma-separated memory target per frontier cycle; defaults to the built-in schedule.",
    )
    parser.add_argument(
        "--frontier_steps_per_episode",
        type=int,
        default=200,
        help="Frontier PINN decisions per episode; 200 decisions x 50 epochs = 10K PINN epochs.",
    )
    parser.add_argument(
        "--frontier_memory_block",
        type=int,
        default=FRONTIER_MEMORY_BLOCK,
        help="Exploration memory block when greedy-plus-exploration collection is enabled.",
    )
    parser.add_argument(
        "--single_epsilon_collection",
        action="store_true",
        help="Collect each frontier cycle in one epsilon-driven phase instead of greedy plus exploration phases.",
    )
    parser.add_argument(
        "--frontier_free_scales",
        action="store_true",
        help="Keep frontier s_v actions free in every cycle, including the final cycle.",
    )
    parser.add_argument(
        "--disable_cycle_stop",
        action="store_true",
        help="Run all configured frontier cycles without the cycle-level patience stop.",
    )
    parser.add_argument("--replay_max_size", type=int, default=None, help="Replay trim size. Defaults to cycles*target_memories in frontier mode.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    config: Optional[RuntimeConfig] = None
    try:
        config = resolve_runtime_selection(args)
        run_project(config)
    except Exception as exc:
        title, body = _failure_notification(config, exc)
        _send_notifyme(title, body)
        raise

    title, body = _success_notification(config)
    _send_notifyme(title, body)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
