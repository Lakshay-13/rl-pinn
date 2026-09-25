"""Smoke tests for the canonical strategy package."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch


RL_PINN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if RL_PINN_DIR not in sys.path:
    sys.path.insert(0, RL_PINN_DIR)


def _random_state() -> np.ndarray:
    return np.random.rand(7, 16, 41).astype(np.float32)


def test_mpdqn() -> bool:
    print("Testing MPDQN...")
    try:
        from strategies.common import HyperparamActionSpec
        from strategies.mpdqn import MPDQNAgent

        spec = HyperparamActionSpec()
        assert np.allclose(spec.sanitize_params(1, np.ones(10))[1:3], 0.0), "LBFGS must mask beta parameters"

        agent = MPDQNAgent((7, 16, 41), [19, 15, 2, 11, 11, 11, 11, 11, 11], hidden_size=64, batch_size=1)
        state = _random_state()
        action_id, params, decoded = agent.select_action(state)
        assert decoded["optimizer"] in spec.action_names
        assert decoded["s_v"].shape == (7,)
        assert params.shape == (spec.param_dim,)

        agent.memory.push(state, (action_id, params, decoded), 1.0, state, False)
        loss = agent.train_step(batch_size=1)
        assert loss is not None and np.isfinite(loss), "MPDQN train_step must produce a finite loss"
        print(f"✓ MPDQN training loss: {loss:.6f}")

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "mpdqn.pt"
            reference = next(agent.q_net.parameters()).detach().cpu().clone()
            agent.save(str(ckpt_path))
            with torch.no_grad():
                next(agent.q_net.parameters()).add_(1.0)
            agent.load(str(ckpt_path))
            restored = next(agent.q_net.parameters()).detach().cpu().clone()
            assert torch.allclose(reference, restored), "MPDQN load() should restore saved weights"
        print("✓ MPDQN smoke test passed")
        return True
    except Exception as exc:  # pragma: no cover - diagnostic path
        print(f"✗ MPDQN test failed: {exc}")
        import traceback

        traceback.print_exc()
        return False


def test_flexplore() -> bool:
    print("Testing FLEXplore...")
    try:
        from strategies.flexplore import FLEXploreAgent

        agent = FLEXploreAgent(
            state_dim=7 * 16 * 41,
            action_dim=10,
            hidden_dim=64,
            horizon=2,
            device=torch.device("cpu"),
            num_samples=8,
            planning_iters=2,
        )
        state = _random_state()
        action = agent.act(state)
        assert isinstance(action, tuple) and len(action) == 3
        assert action[2]["s_v"].shape == (7,)

        agent.update(state, action, 1.0, state, False)
        loss = agent.train_step(batch_size=1)
        assert loss is not None and np.isfinite(loss), "FLEXplore train_step must produce a finite loss"

        plan = agent.plan(state)
        assert "action_id" in plan and "params" in plan and np.isfinite(plan["score"])
        assert plan["sequence"], "Planner should return a rollout summary"
        print(f"✓ FLEXplore planning score: {plan['score']:.6f}")
        print("✓ FLEXplore smoke test passed")
        return True
    except Exception as exc:  # pragma: no cover - diagnostic path
        print(f"✗ FLEXplore test failed: {exc}")
        import traceback

        traceback.print_exc()
        return False


def test_seq_madac() -> bool:
    print("Testing Seq-MADAC...")
    try:
        from strategies.seq_madac import SeqMADACAgent

        agent = SeqMADACAgent(
            state_dim=7 * 16 * 41,
            action_dim=10,
            hidden_dim=64,
            num_steps=10,
            device=torch.device("cpu"),
            batch_size=1,
        )
        state = _random_state()
        action_id, params, decoded = agent.select_action(state)
        assert decoded["optimizer"] in {"adam", "lbfgs"}
        assert params.shape == (10,)
        assert decoded["s_v"].shape == (7,)

        agent.update(state, (action_id, params, decoded), 1.0, state, False)
        loss = agent.train_step(batch_size=1)
        assert loss is not None and np.isfinite(loss), "Seq-MADAC train_step must produce a finite loss"
        print(f"✓ Seq-MADAC training loss: {loss:.6f}")
        print("✓ Seq-MADAC smoke test passed")
        return True
    except Exception as exc:  # pragma: no cover - diagnostic path
        print(f"✗ Seq-MADAC test failed: {exc}")
        import traceback

        traceback.print_exc()
        return False


def test_frozen_scale_buffer_actions() -> bool:
    print("Testing frozen-scale buffer normalization...")
    try:
        from project import CurriculumStage, _prepare_stage_action
        from strategies.flexplore import FLEXploreAgent
        from strategies.mpdqn import MPDQNAgent
        from strategies.seq_madac import SeqMADACAgent

        stage = CurriculumStage(
            name="final_a",
            steps_per_episode=300,
            action_epochs=100,
            epsilon=0.75,
            scale_floor=1.0,
            freeze_scales=True,
        )
        state = _random_state()

        mpdqn = MPDQNAgent((7, 16, 41), [19, 15, 2, 11, 11, 11, 11, 11, 11], hidden_size=64, batch_size=1)
        mpdqn_action = mpdqn.select_action(state)
        mpdqn_buffer_action, mpdqn_decoded = _prepare_stage_action(mpdqn, mpdqn_action, stage)
        assert isinstance(mpdqn_buffer_action, tuple) and len(mpdqn_buffer_action) == 3
        assert np.asarray(mpdqn_buffer_action[1]).shape == (10,)
        assert np.allclose(mpdqn_decoded["s_v"], np.ones(7, dtype=np.float32))
        mpdqn.memory.push(state, mpdqn_action, 1.0, state, False)
        mpdqn.memory.push(state, mpdqn_buffer_action, 1.0, state, False)
        mpdqn.memory.sample(2)

        seq = SeqMADACAgent(
            state_dim=7 * 16 * 41,
            action_dim=10,
            hidden_dim=64,
            num_steps=10,
            device=torch.device("cpu"),
            batch_size=1,
        )
        seq_action = seq.select_action(state)
        seq_buffer_action, seq_decoded = _prepare_stage_action(seq, seq_action, stage)
        assert isinstance(seq_buffer_action, np.ndarray) and seq_buffer_action.shape == (seq.num_steps,)
        assert np.allclose(seq_decoded["s_v"], np.ones(7, dtype=np.float32))
        seq.replay.push(state.reshape(-1), seq._trace_from_action(seq_action), 1.0, state.reshape(-1), False)
        seq.replay.push(state.reshape(-1), seq_buffer_action, 1.0, state.reshape(-1), False)
        seq.replay.sample(2)

        flexplore = FLEXploreAgent(
            state_dim=7 * 16 * 41,
            action_dim=10,
            hidden_dim=64,
            horizon=2,
            device=torch.device("cpu"),
            num_samples=8,
            planning_iters=2,
        )
        flexplore_action = flexplore.act(state)
        flexplore_buffer_action, flexplore_decoded = _prepare_stage_action(flexplore, flexplore_action, stage)
        assert isinstance(flexplore_buffer_action, tuple) and len(flexplore_buffer_action) == 3
        assert np.asarray(flexplore_buffer_action[1]).shape == (10,)
        assert np.allclose(flexplore_decoded["s_v"], np.ones(7, dtype=np.float32))
        flexplore.replay.push(state, flexplore_action, 1.0, state, False)
        flexplore.replay.push(state, flexplore_buffer_action, 1.0, state, False)
        flexplore.replay.sample(2)

        print("✓ Frozen-scale buffer normalization passed")
        return True
    except Exception as exc:  # pragma: no cover - diagnostic path
        print(f"✗ Frozen-scale buffer normalization failed: {exc}")
        import traceback

        traceback.print_exc()
        return False


def test_project_reward_penalty() -> bool:
    print("Testing active project reward penalty...")
    try:
        from project import PROJECT_SCALE_PENALTY, _project_compute_reward

        assert PROJECT_SCALE_PENALTY == 800.0
        low_scales = np.full(7, 0.1, dtype=np.float32)
        full_scales = np.ones(7, dtype=np.float32)
        reward_low = _project_compute_reward(0.5, 1.0, low_scales, 1)
        reward_full = _project_compute_reward(0.5, 1.0, full_scales, 1)
        expected_gap = 800.0 * np.sum(1.0 - low_scales)
        actual_gap = reward_full - reward_low
        assert np.isclose(actual_gap, expected_gap, atol=1e-5), "Active project reward must apply 800x scale penalty"
        print("✓ Active project reward penalty passed")
        return True
    except Exception as exc:  # pragma: no cover - diagnostic path
        print(f"✗ Active project reward penalty failed: {exc}")
        import traceback

        traceback.print_exc()
        return False


def test_frontier_stage_keeps_low_scales() -> bool:
    print("Testing frontier stage keeps decoded low scales...")
    try:
        from project import CurriculumStage, _prepare_stage_action

        stage = CurriculumStage(
            name="frontier_06",
            steps_per_episode=200,
            action_epochs=50,
            epsilon=0.15,
            scale_floor=0.70,
            freeze_scales=False,
        )
        action = {
            "optimizer": "adam",
            "lr": 1e-4,
            "betas": (0.9, 0.999),
            "s_v": np.asarray([0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.2], dtype=np.float32),
        }

        _, decoded = _prepare_stage_action(object(), action, stage)
        expected = np.asarray([0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0], dtype=np.float32)
        assert np.allclose(decoded["s_v"], expected), "Frontier stages must not raise scales to the stage floor"
        print("✓ Frontier stage keeps decoded low scales")
        return True
    except Exception as exc:  # pragma: no cover - diagnostic path
        print(f"✗ Frontier stage keeps decoded low scales failed: {exc}")
        import traceback

        traceback.print_exc()
        return False


def test_frontier_memory_schedule() -> bool:
    print("Testing frontier memory schedule...")
    try:
        from project import (
            FRONTIER_SCALE_BASELINE_SCHEDULE,
            FRONTIER_TARGET_MEMORY_SCHEDULE,
            _build_frontier_stages,
            _frontier_collection_plan,
            _frontier_cycle_scale_baseline,
            _frontier_cycle_target_memories,
        )

        expected = (10000, 20000, 20000, 30000, 30000, 30000, 40000, 40000, 40000, 40000)
        expected_baselines = (0.10, 0.30, 0.50, 0.70, 0.90)
        assert FRONTIER_TARGET_MEMORY_SCHEDULE == expected
        assert FRONTIER_SCALE_BASELINE_SCHEDULE == expected_baselines
        assert tuple(_frontier_cycle_target_memories(i) for i in range(len(expected))) == expected
        assert tuple(_frontier_cycle_scale_baseline(i) for i in range(len(expected_baselines))) == expected_baselines
        assert tuple(stage.scale_floor for stage in _build_frontier_stages(0.15)) == expected_baselines

        cycle_1 = _frontier_collection_plan(0, 0.15)
        assert len(cycle_1) == 1
        assert cycle_1[0].name == "exploration"
        assert cycle_1[0].target_memories == 10000
        assert cycle_1[0].eval_mode is False

        cycle_7 = _frontier_collection_plan(6, 0.15)
        assert len(cycle_7) == 2
        assert cycle_7[0].name == "greedy"
        assert cycle_7[0].target_memories == 30000
        assert cycle_7[0].eval_mode is True
        assert cycle_7[1].name == "exploration"
        assert cycle_7[1].target_memories == 10000
        assert sum(phase.target_memories for phase in cycle_7) == expected[6]

        print("✓ Frontier memory schedule passed")
        return True
    except Exception as exc:  # pragma: no cover - diagnostic path
        print(f"✗ Frontier memory schedule failed: {exc}")
        import traceback

        traceback.print_exc()
        return False


def test_frontier_render_budget() -> bool:
    print("Testing frontier render budget...")
    try:
        from project import PROJECT_RENDER_ACTION_EPOCHS, PROJECT_RENDER_DECISIONS, _build_flat_frontier_render_stage, _build_frontier_stages

        stages = _build_frontier_stages(0.15)
        total_epochs = sum(stage.steps_per_episode * stage.action_epochs for stage in stages)
        assert total_epochs == 50000
        flat_stage = _build_flat_frontier_render_stage()
        assert flat_stage.steps_per_episode == PROJECT_RENDER_DECISIONS
        assert flat_stage.action_epochs == PROJECT_RENDER_ACTION_EPOCHS
        assert flat_stage.steps_per_episode * flat_stage.action_epochs == 100000
        print("✓ Frontier render budget passed")
        return True
    except Exception as exc:  # pragma: no cover - diagnostic path
        print(f"✗ Frontier render budget failed: {exc}")
        import traceback

        traceback.print_exc()
        return False


def test_cycle_three_shape_gate() -> bool:
    print("Testing cycle-3 shape gate...")
    try:
        from project import _shape_gate_report

        phi = np.linspace(0.0, 1.8, 100, dtype=np.float32)
        theory = -3.0 - 7.0 * np.exp(-((phi - 1.55) ** 2) / 0.12) + 2.8 * np.exp(-((phi - 1.82) ** 2) / 0.02)
        flat = np.zeros_like(theory)
        report_good = _shape_gate_report(
            {
                "potential": {
                    "theory": theory.tolist(),
                    "current": theory.tolist(),
                    "best": None,
                }
            }
        )
        report_bad = _shape_gate_report(
            {
                "potential": {
                    "theory": theory.tolist(),
                    "current": flat.tolist(),
                    "best": flat.tolist(),
                }
            }
        )
        assert report_good["pass"] is True
        assert report_bad["pass"] is False
        print("✓ Cycle-3 shape gate passed")
        return True
    except Exception as exc:  # pragma: no cover - diagnostic path
        print(f"✗ Cycle-3 shape gate failed: {exc}")
        import traceback

        traceback.print_exc()
        return False


def test_cycle_loss_stop_patience() -> bool:
    print("Testing cycle-loss patience stop...")
    try:
        from project import CycleStopState, _advance_cycle_stop_state

        state = CycleStopState()
        state, info = _advance_cycle_stop_state(3, {"loss": {"current_unweighted": 1e-3}}, state)
        assert info["baseline"] is True and info["stop"] is False
        assert state.best_cycle == 3 and np.isclose(state.best_log_metric, -3.0)

        state, info = _advance_cycle_stop_state(4, {"loss": {"current_unweighted": 5e-4}}, state)
        assert info["improved"] is False and info["stop"] is False
        assert state.patience_counter == 1

        state, info = _advance_cycle_stop_state(5, {"loss": {"current_unweighted": 4e-4}}, state)
        assert info["improved"] is False and info["stop"] is True
        assert state.patience_counter == 2

        state = CycleStopState()
        state, _ = _advance_cycle_stop_state(3, {"loss": {"current_unweighted": 1e-3}}, state)
        state, info = _advance_cycle_stop_state(4, {"loss": {"current_unweighted": 1e-4}}, state)
        assert info["improved"] is True and info["stop"] is False
        assert state.patience_counter == 0 and state.best_cycle == 4

        print("✓ Cycle-loss patience stop passed")
        return True
    except Exception as exc:  # pragma: no cover - diagnostic path
        print(f"✗ Cycle-loss patience stop failed: {exc}")
        import traceback

        traceback.print_exc()
        return False


def test_render_cycle_grid_numeric_order() -> bool:
    print("Testing render cycle-grid numeric ordering...")
    try:
        from render_cycle_grids import _discover_cycle_images

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for cycle in (1, 10, 2):
                cycle_dir = root / f"cycle_{cycle}"
                cycle_dir.mkdir(parents=True, exist_ok=True)
                (cycle_dir / "potential.png").touch()
            entries = _discover_cycle_images(root, "potential", 10)
            assert [cycle for cycle, _ in entries] == [1, 2, 10]

        print("✓ Render cycle-grid numeric ordering passed")
        return True
    except Exception as exc:  # pragma: no cover - diagnostic path
        print(f"✗ Render cycle-grid numeric ordering failed: {exc}")
        import traceback

        traceback.print_exc()
        return False


def main() -> int:
    print("Running strategy smoke tests...")
    mpdqn_passed = test_mpdqn()
    flexplore_passed = test_flexplore()
    seq_madac_passed = test_seq_madac()
    frozen_scale_passed = test_frozen_scale_buffer_actions()
    reward_penalty_passed = test_project_reward_penalty()
    low_scale_frontier_passed = test_frontier_stage_keeps_low_scales()
    frontier_schedule_passed = test_frontier_memory_schedule()
    frontier_render_passed = test_frontier_render_budget()
    shape_gate_passed = test_cycle_three_shape_gate()
    cycle_stop_passed = test_cycle_loss_stop_patience()
    grid_order_passed = test_render_cycle_grid_numeric_order()

    if (
        mpdqn_passed
        and flexplore_passed
        and seq_madac_passed
        and frozen_scale_passed
        and reward_penalty_passed
        and low_scale_frontier_passed
        and frontier_schedule_passed
        and frontier_render_passed
        and shape_gate_passed
        and cycle_stop_passed
        and grid_order_passed
    ):
        print("\n✓ All strategy tests passed")
        return 0

    print("\n✗ Some strategy tests failed")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
