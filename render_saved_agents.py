#!/usr/bin/env python3.11
"""Render retained frontier agents for a configurable inference budget."""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path

import torch

import project


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--decisions", type=int, default=6000)
    parser.add_argument("--action-epochs", type=int, default=50)
    parser.add_argument("--output-name", default="render_300k")
    parser.add_argument("--schedule-root", type=Path, action="append", required=True)
    parser.add_argument("--status-path", type=Path, required=True)
    return parser.parse_args()


def latest_checkpoint(root: Path) -> tuple[Path, int]:
    checkpoints = []
    for path in root.glob("agent_cycle_*.pth"):
        try:
            cycle = int(path.stem.rsplit("_", 1)[-1])
        except ValueError:
            continue
        checkpoints.append((cycle, path))
    if not checkpoints:
        raise FileNotFoundError(f"No agent_cycle_*.pth checkpoint found under {root}")
    cycle, checkpoint = max(checkpoints, key=lambda item: item[0])
    return checkpoint, cycle


def render_schedule(project_root: Path, root: Path, decisions: int, action_epochs: int, output_name: str) -> dict:
    checkpoint, checkpoint_cycle = latest_checkpoint(root)
    output = root / output_name
    started = time.time()
    device = torch.device("cpu")
    state_shape = project._agent_state_shape(True)
    agent = project._build_agent_with_shape(
        "flexplore",
        device,
        state_shape=state_shape,
        n_steps=project.DEFAULT_FRONTIER_N_STEP_HORIZON,
    )
    agent.load(str(checkpoint), load_memory=False)
    runner = project.ProjectRunner(
        strategy="flexplore",
        agent=agent,
        device=device,
        cinns_path=str(project_root),
        state_shape=state_shape,
        n_step_horizon=project.DEFAULT_FRONTIER_N_STEP_HORIZON,
        frontier_relative_actions=True,
        num_workers=1,
        worker_devices=[device],
    )
    report = runner.render_frontier_unconditional(
        str(output),
        validation_steps=decisions,
        action_epochs=action_epochs,
    )
    return {
        "status": "completed",
        "checkpoint": str(checkpoint),
        "checkpoint_cycle": checkpoint_cycle,
        "output": str(output),
        "decisions": decisions,
        "action_epochs": action_epochs,
        "inference_epochs": decisions * action_epochs,
        "elapsed_sec": time.time() - started,
        "loss": report.get("loss"),
    }


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    project.ROOT_DIR = project_root
    import os

    os.environ["RLPINN_LOCALIZATION_TARGET"] = "both"
    os.environ["RLPINN_LOCALIZATION_KERNEL"] = "gaussian"

    results: dict[str, dict] = {}
    for root_arg in args.schedule_root:
        root = root_arg.resolve()
        schedule_id = root.name
        print(f"[{schedule_id}] rendering {args.decisions * args.action_epochs:,} inference epochs", flush=True)
        try:
            results[schedule_id] = render_schedule(project_root, root, args.decisions, args.action_epochs, args.output_name)
            print(f"[{schedule_id}] completed", flush=True)
        except Exception as exc:
            results[schedule_id] = {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
            print(f"[{schedule_id}] failed: {exc}", flush=True)

    status_path = args.status_path.resolve()
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    failed = [key for key, value in results.items() if value["status"] != "completed"]
    notify = os.environ.get("RLPINN_SKIP_NOTIFYME") != "1"
    if failed:
        if notify:
            project._send_notifyme(
                f"RL-PINN - {args.output_name} Agent Render Failed",
                "\n".join([
                    "Project: RL-PINN",
                    f"Task: {args.output_name} inference render for localized RL agent",
                    "Status: Failed",
                    f"Error: Failed schedules: {', '.join(failed)}",
                    f"Summary: See {status_path}.",
                ]),
            )
        return 1
    if notify:
        project._send_notifyme(
            f"RL-PINN - {args.output_name} Agent Render Complete",
            "\n".join([
                "Project: RL-PINN",
                f"Task: {args.output_name} inference render for localized RL agent",
                "Status: Success",
                f"Summary: Cycle-5 checkpoint rendered for {args.decisions * args.action_epochs:,} inference epochs without training or scale floors. Results: {status_path}",
            ]),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
