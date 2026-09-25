#!/usr/bin/env python3
"""Run the RL-PINN schedule campaign from the public configuration registry."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "schedules.json"
SMOKE_TEST = ROOT / "strategies" / "test_strategies.py"
PROJECT_SCRIPT = ROOT / "project.py"
SAVED_RENDER_SCRIPT = ROOT / "render_saved_agents.py"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_config() -> dict[str, Any]:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if payload.get("project") != "RL-PINN" or not payload.get("schedules"):
        raise ValueError(f"Invalid schedule registry: {CONFIG_PATH}")
    ids = [row["id"] for row in payload["schedules"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Schedule ids must be unique")
    for row in payload["schedules"]:
        if len(row["memory_targets"]) != row["cycles"]:
            raise ValueError(f"{row['id']}: memory target count must equal cycle count")
        if len(row["epsilon_schedule"]) != row["cycles"]:
            raise ValueError(f"{row['id']}: epsilon count must equal cycle count")
        reset_epochs = row["steps_per_episode"] * payload["runtime"]["action_epochs_per_decision"]
        if int(row.get("pin_reset_epochs", -1)) != reset_epochs:
            raise ValueError(f"{row['id']}: PINN reset epochs do not match the configured step window")
        if not row.get("replicates"):
            raise ValueError(f"{row['id']}: at least one replicate is required")
    return payload


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def build_training_command(
    spec: dict[str, Any],
    output_dir: Path,
    python_bin: str,
    workers: int,
    runtime: dict[str, Any],
) -> list[str]:
    command = [
        python_bin,
        str(PROJECT_SCRIPT),
        "--strategy",
        "flexplore",
        "--device",
        "cpu",
        "--workers",
        str(workers),
        "--cycles",
        str(spec["cycles"]),
        "--train_epochs",
        str(runtime["train_epochs"]),
        "--patience",
        str(runtime["patience"]),
        "--min_delta",
        str(runtime["min_delta"]),
        "--frontier_curriculum",
        "--frontier_epsilon",
        str(spec["epsilon_schedule"][0]),
        "--frontier_epsilon_schedule",
        ",".join(str(value) for value in spec["epsilon_schedule"]),
        "--frontier_memory_schedule",
        ",".join(str(value) for value in spec["memory_targets"]),
        "--frontier_steps_per_episode",
        str(spec["steps_per_episode"]),
        "--frontier_memory_block",
        str(spec["memory_block"]),
        "--log_dir",
        str(output_dir),
    ]
    if spec["collection_mode"] == "single_epsilon":
        command.append("--single_epsilon_collection")
    elif spec["collection_mode"] != "greedy_then_epsilon":
        raise ValueError(f"Unsupported collection mode: {spec['collection_mode']}")
    if not spec["cycle_stop"]:
        command.append("--disable_cycle_stop")
    if spec.get("frontier_free_scales", False):
        command.append("--frontier_free_scales")
    if spec["cycle_evaluation"] == "unconditional_frontier":
        command.append("--skip_render")
    elif spec["cycle_evaluation"] != "schedule_context":
        raise ValueError(f"Unsupported cycle evaluation: {spec['cycle_evaluation']}")
    return command


def run_subprocess(command: list[str], cwd: Path, env: dict[str, str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("Command: " + json.dumps(command) + "\n")
        handle.flush()
        return subprocess.run(
            command,
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        ).returncode


def latest_run_dir(output_dir: Path) -> Path:
    candidates = []
    for path in output_dir.glob("run_*"):
        suffix = path.name.split("_", 1)[-1]
        if path.is_dir() and suffix.isdigit():
            candidates.append((int(suffix), path))
    if not candidates:
        raise FileNotFoundError(f"project.py did not create a run directory under {output_dir}")
    return max(candidates, key=lambda item: item[0])[1]


def render_unconditional_cycle_checkpoints(run_dir: Path, schedule_id: str, cycles: int) -> None:
    import torch

    sys.path.insert(0, str(ROOT))
    import project

    os.environ["RLPINN_LOCALIZATION_TARGET"] = "both"
    os.environ["RLPINN_LOCALIZATION_KERNEL"] = "gaussian"
    torch.set_num_threads(1)
    device = torch.device("cpu")
    state_shape = project._agent_state_shape(True)
    for cycle in range(1, cycles + 1):
        checkpoint = run_dir / f"agent_cycle_{cycle}.pth"
        if not checkpoint.exists():
            raise FileNotFoundError(f"{schedule_id}: missing cycle checkpoint {checkpoint.name}")
        output = run_dir / f"cycle_{cycle}"
        report_path = output / "render_report.json"
        if report_path.exists():
            continue
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
            cinns_path=str(ROOT),
            state_shape=state_shape,
            n_step_horizon=project.DEFAULT_FRONTIER_N_STEP_HORIZON,
            frontier_relative_actions=True,
            num_workers=1,
            worker_devices=[device],
        )
        runner.render_frontier_unconditional(
            str(output),
            validation_steps=2000,
            action_epochs=50,
        )
        if not report_path.exists():
            raise RuntimeError(f"{schedule_id}: render did not create {report_path}")
        print(f"[{schedule_id}] rendered cycle {cycle}/{cycles}", flush=True)
        del runner, agent


def render_saved_agent(
    run_dir: Path,
    output_dir: Path,
    epochs: int,
    python_bin: str,
    env: dict[str, str],
) -> None:
    action_epochs = int(load_config()["runtime"]["action_epochs_per_decision"])
    if epochs <= 0 or epochs % action_epochs:
        raise ValueError(f"Inference budget must be a positive multiple of {action_epochs}: {epochs}")
    output_name = f"render_{epochs // 1000}k"
    command = [
        python_bin,
        str(SAVED_RENDER_SCRIPT),
        "--project-root",
        str(ROOT),
        "--decisions",
        str(epochs // action_epochs),
        "--action-epochs",
        str(action_epochs),
        "--output-name",
        output_name,
        "--schedule-root",
        str(run_dir),
        "--status-path",
        str(output_dir / f"{output_name}_status.json"),
    ]
    log_path = output_dir / f"{output_name}.log"
    code = run_subprocess(command, ROOT, env, log_path)
    report_path = run_dir / output_name / "render_report.json"
    if code != 0 or not report_path.exists():
        raise RuntimeError(f"Saved-agent render failed for {output_name} (exit {code}); see {log_path}")


def report_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    potential = payload.get("potential", {})
    best = potential.get("best")
    theory = potential.get("theory")
    rmse = None
    if best is not None and theory is not None and len(best) == len(theory) and len(best) > 0:
        differences = [float(predicted) - float(reference) for predicted, reference in zip(best, theory)]
        rmse = math.sqrt(sum(value * value for value in differences) / len(differences))
    return {
        "report": str(path),
        "report_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "best_potential_rmse": rmse,
        "reported_best_loss": payload.get("loss", {}).get("best_unweighted"),
        "best_available": payload.get("best_available"),
    }


def select_jobs(
    registry: dict[str, Any],
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> list[tuple[dict[str, Any], int, str, str]]:
    schedules = {row["id"]: row for row in registry["schedules"]}
    selected: list[tuple[dict[str, Any], int, str, str]] = []
    include_reference = bool(args.reference or args.all or args.include_reference)
    if args.all:
        selected_schedules = list(registry["schedules"])
    elif args.schedule:
        if args.schedule not in schedules:
            parser.error(f"Unknown schedule {args.schedule!r}; choose from {', '.join(schedules)}")
        selected_schedules = [schedules[args.schedule]]
    else:
        selected_schedules = []

    for spec in selected_schedules:
        replicates = spec["replicates"]
        if args.replicate is not None:
            if args.replicate not in replicates:
                if args.all:
                    continue
                parser.error(f"{spec['id']} has replicates {replicates}; requested {args.replicate}")
            replicates = [args.replicate]
        selected.extend((spec, replicate, spec["id"], "both") for replicate in replicates)

    if include_reference:
        reference = registry["reference"]
        spec = schedules[reference["schedule_id"]]
        if args.replicate is None or args.replicate in reference["replicates"]:
            selected.append((spec, reference["replicates"][0], "reference", reference["localization_target"]))
        elif args.reference or args.include_reference:
            parser.error(f"The non-localized reference has replicate {reference['replicates'][0]} only")

    if args.include_extensions:
        allowed = {(row["schedule_id"], int(row["replicate"])) for row in registry["inference_extensions"]}
        selected_pairs = {(spec["id"], replicate) for spec, replicate, _, _ in selected}
        missing = allowed - selected_pairs
        if missing:
            parser.error("--include-extensions requires Schedule 1 replicate 1 and Schedule 2 replicate 2")

    if not selected:
        parser.error("Choose --all, --schedule ID, --reference, --list-schedules, or --smoke-test")
    return selected


def run_job(
    spec: dict[str, Any],
    replicate: int,
    label: str,
    localization_target: str,
    output_root: Path,
    python_bin: str,
    workers: int,
    registry: dict[str, Any],
    parent_env: dict[str, str],
) -> dict[str, Any]:
    job_dir = output_root / label / f"rep_{replicate}"
    job_dir.mkdir(parents=True, exist_ok=True)
    command = build_training_command(spec, job_dir, python_bin, workers, registry["runtime"])
    config = {
        "project": registry["project"],
        "schedule_id": spec["id"],
        "label": label,
        "replicate": replicate,
        "strategy": registry["runtime"]["strategy"],
        "localization_target": localization_target,
        "localization_kernel": registry["runtime"]["localization_kernel"],
        "device": "cpu",
        "workers": workers,
        "cycles": spec["cycles"],
        "memory_targets": spec["memory_targets"],
        "epsilon_schedule": spec["epsilon_schedule"],
        "collection_mode": spec["collection_mode"],
        "steps_per_episode": spec["steps_per_episode"],
        "pin_reset_epochs": spec["pin_reset_epochs"],
        "memory_block": spec["memory_block"],
        "cycle_stop": spec["cycle_stop"],
        "cycle_evaluation": spec["cycle_evaluation"],
        "frontier_free_scales": bool(spec.get("frontier_free_scales", False)),
        "additional_inference_epochs": spec.get("additional_inference_epochs", []),
        "command": command,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    config_path = job_dir / "config.json"
    result_path = job_dir / "result.json"
    write_json(config_path, config)
    write_json(result_path, {**config, "status": "running"})
    environment = parent_env.copy()
    environment["RLPINN_LOCALIZATION_TARGET"] = localization_target
    environment["RLPINN_LOCALIZATION_KERNEL"] = registry["runtime"]["localization_kernel"]
    environment["RLPINN_SKIP_NOTIFYME"] = "1"
    log_path = job_dir / "training.log"
    print(f"[{label}/rep_{replicate}] training started", flush=True)
    started = time.monotonic()
    exit_code = run_subprocess(command, ROOT, environment, log_path)
    if exit_code != 0:
        raise RuntimeError(f"project.py exited with code {exit_code}; see {log_path}")

    run_dir = latest_run_dir(job_dir)
    checkpoints = sorted(
        run_dir.glob("agent_cycle_*.pth"),
        key=lambda path: int(path.stem.rsplit("_", 1)[-1]),
    )
    if not checkpoints:
        raise RuntimeError(f"No saved agent checkpoint was produced under {run_dir}")

    if spec["cycle_evaluation"] == "unconditional_frontier":
        render_unconditional_cycle_checkpoints(run_dir, spec["id"], spec["cycles"])

    cycle_reports = sorted(
        (path for path in run_dir.glob("cycle_*/render_report.json") if path.is_file()),
        key=lambda path: int(path.parent.name.split("_", 1)[-1]),
    )
    if not cycle_reports:
        raise RuntimeError(f"No cycle render reports were produced under {run_dir}")
    if not spec["cycle_stop"] and len(cycle_reports) < spec["cycles"]:
        raise RuntimeError(
            f"Expected {spec['cycles']} cycle reports, found {len(cycle_reports)} under {run_dir}"
        )

    required_budgets = [int(value) for value in spec.get("additional_inference_epochs", [])]
    if required_budgets:
        for budget in required_budgets:
            render_saved_agent(run_dir, job_dir, budget, python_bin, environment)

    if required_budgets:
        standard_report = run_dir / "render_100k" / "render_report.json"
    else:
        standard_report = cycle_reports[-1]
    if not standard_report.exists():
        raise RuntimeError(f"Missing selected result report: {standard_report}")

    result = {
        **config,
        "status": "completed",
        "exit_code": exit_code,
        "cycles_with_checkpoints": len(checkpoints),
        "cycles_with_reports": len(cycle_reports),
        "elapsed_seconds": time.monotonic() - started,
        "artifact_directory": str(run_dir),
        "selected_evaluation": report_summary(standard_report),
    }
    write_json(result_path, result)
    print(f"[{label}/rep_{replicate}] complete", flush=True)
    return result


def run_extensions(
    registry: dict[str, Any],
    output_root: Path,
    python_bin: str,
    env: dict[str, str],
) -> list[dict[str, Any]]:
    results = []
    for extension in registry["inference_extensions"]:
        label = extension["schedule_id"]
        replicate = int(extension["replicate"])
        run_dir = latest_run_dir(output_root / label / f"rep_{replicate}")
        for budget in extension["epochs"]:
            render_saved_agent(run_dir, run_dir.parent, int(budget), python_bin, env)
            report = run_dir / f"render_{int(budget) // 1000}k" / "render_report.json"
            results.append(
                {
                    "schedule_id": label,
                    "replicate": replicate,
                    "inference_epochs": int(budget),
                    **report_summary(report),
                }
            )
    return results


def notify_completion(status: str, summary: str) -> None:
    try:
        sys.path.insert(0, str(ROOT))
        import project

        project._send_notifyme(
            f"RL-PINN - Reproduction {status.title()}",
            f"Project: RL-PINN\nTask: Reproduce configured results\nStatus: {status}\nSummary: {summary}",
        )
    except Exception as exc:
        print(f"NotifyMe notification failed: {type(exc).__name__}: {exc}", file=sys.stderr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--all", action="store_true", help="Run the reference and all configured schedules")
    selection.add_argument("--schedule", help="Run one schedule, for example schedule_3")
    selection.add_argument("--reference", action="store_true", help="Run the non-localized Flexplore reference")
    selection.add_argument("--list-schedules", action="store_true", help="List configured schedules and replicates")
    selection.add_argument("--smoke-test", action="store_true", help="Run the lightweight strategy smoke checks")
    parser.add_argument("--replicate", type=int, help="Run only this replicate")
    parser.add_argument("--include-reference", action="store_true", help="Add the reference to a selected schedule")
    parser.add_argument("--include-extensions", action="store_true", help="Render the configured 200K/500K extensions")
    parser.add_argument("--workers", type=int, default=None, help="Collection workers per experiment; default 8")
    parser.add_argument("--python", default=sys.executable, help="Python executable for child runs")
    parser.add_argument("--output-dir", type=Path, help="Output root; defaults to repro_runs/run_<UTC timestamp>")
    parser.add_argument("--dry-run", action="store_true", help="Print the training commands without running them")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    registry = load_config()
    if args.smoke_test:
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(ROOT) + os.pathsep + environment.get("PYTHONPATH", "")
        code = subprocess.run([args.python, str(SMOKE_TEST)], cwd=ROOT, env=environment, check=False).returncode
        return code
    if args.list_schedules:
        for spec in registry["schedules"]:
            print(f"{spec['id']}: {spec['cycles']} cycles; replicates {spec['replicates']}; {spec['collection_mode']}")
        print("reference: non-localized Flexplore; replicate 1")
        return 0
    if args.include_reference and (args.all or args.reference):
        raise SystemExit("--include-reference is only used with --schedule")
    if args.workers is not None and args.workers <= 0:
        raise SystemExit("--workers must be positive")
    if args.replicate is not None and args.replicate <= 0:
        raise SystemExit("--replicate must be positive")

    parser = argparse.ArgumentParser(add_help=False)
    jobs = select_jobs(registry, args, parser)
    workers = args.workers or int(registry["runtime"]["workers"])
    output_root = (args.output_dir or ROOT / "repro_runs" / f"run_{timestamp()}").resolve()
    commands = []
    for spec, replicate, label, localization_target in jobs:
        job_dir = output_root / label / f"rep_{replicate}"
        commands.append(
            {
                "label": label,
                "replicate": replicate,
                "localization_target": localization_target,
                "command": build_training_command(
                    spec,
                    job_dir,
                    args.python,
                    workers,
                    registry["runtime"],
                ),
            }
        )
    if args.dry_run:
        print(json.dumps({"output_root": str(output_root), "jobs": commands}, indent=2))
        return 0
    if output_root.exists() and any(output_root.iterdir()):
        raise SystemExit(f"Output directory is not empty: {output_root}")

    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "project": registry["project"],
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "runtime": {**registry["runtime"], "workers": workers, "python": args.python},
        "jobs": commands,
        "include_extensions": bool(args.include_extensions),
    }
    write_json(output_root / "manifest.json", manifest)
    environment = os.environ.copy()
    environment["RLPINN_SKIP_NOTIFYME"] = "1"
    results = []
    failures = []
    for spec, replicate, label, localization_target in jobs:
        try:
            results.append(
                run_job(
                    spec,
                    replicate,
                    label,
                    localization_target,
                    output_root,
                    args.python,
                    workers,
                    registry,
                    environment,
                )
            )
        except Exception as exc:
            failure = {
                "label": label,
                "replicate": replicate,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
            failures.append(failure)
            write_json(output_root / label / f"rep_{replicate}" / "result.json", failure)
            print(f"[{label}/rep_{replicate}] failed: {failure['error']}", file=sys.stderr, flush=True)

    extension_results = []
    if args.include_extensions and not failures:
        try:
            extension_results = run_extensions(registry, output_root, args.python, environment)
        except Exception as exc:
            failures.append({"status": "failed", "error": f"extensions: {type(exc).__name__}: {exc}"})

    status = "failed" if failures else "completed"
    final = {
        **manifest,
        "status": status,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "results": results,
        "failures": failures,
        "inference_extensions": extension_results,
    }
    write_json(output_root / "status.json", final)
    summary = f"{len(results)} run(s) completed, {len(failures)} failed."
    if failures:
        error_summary = "; ".join(
            str(item.get("error", "unknown error")) for item in failures[:3]
        )
        summary += f" Error: {error_summary}."
    summary += f" Artifacts: {output_root}"
    notify_completion("failed" if failures else "completed", summary)
    print(summary)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
