#!/usr/bin/env python3
"""Random U-Net-guidance planning followed by COAL/TOPP-RA/MPPI/Rerun."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time
from typing import Any

import numpy as np

from coal_collision import CoalCollisionChecker
from evaluate_se3_diffusion import dense_path, path_metrics
from ompl_se3_planner import OMPLSE3Planner
from run_overfit_cube_single_pipeline import (
    load_environment,
    quaternion_distance,
    sample_valid_pose,
)


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_ENVIRONMENT = PROJECT_DIR / "environment_multihomotopy_v002.json"
DEFAULT_URDF = (
    PROJECT_DIR
    / "etc/URDF-for-gazebo/urdf/HDJQR-0102-0055.SLDASM.urdf"
)
DEFAULT_MODEL = PROJECT_DIR / "hnuter206_4_5kg.xml"
DEFAULT_EXPERIMENT = PROJECT_DIR / "results/diffusion_se3_three_stage_v002"
DEFAULT_TORCH_PYTHON = Path(
    "/home/z017/research/diffusion_model/.envs/mpd-splines/bin/python"
)
SAMPLER = PROJECT_DIR / "sample_unet_guided_paths.py"
SAMPLE_AND_FILTER = PROJECT_DIR / "sample_and_filter_candidates.py"
PIPELINE = PROJECT_DIR / "run_overfit_cube_single_pipeline.sh"
CUROBO_FILTER = PROJECT_DIR / "curobo_collision.py"
DEFAULT_INTEGRATED_PYTHON = Path("/home/z017/research/curobo_env/bin/python")
DEFAULT_CUROBO_PYTHON = Path("/home/z017/research/curobo_env/bin/python")
DEFAULT_CUROBO_SPHERES = (
    PROJECT_DIR
    / "etc"
    / "URDF-for-gazebo"
    / "config"
    / "HDJQR-0102-0055.SLDASM_curobo_spheres.yml"
)


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    raise TypeError(f"cannot JSON-encode {type(value).__name__}")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload, indent=2, ensure_ascii=False, default=json_default
        ) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--candidate-count", type=int, default=32)
    parser.add_argument("--maximum-pair-attempts", type=int, default=8)
    parser.add_argument("--maximum-execution-candidates", type=int, default=3)
    parser.add_argument("--acceptance-clearance", type=float, default=0.08)
    parser.add_argument("--endpoint-clearance", type=float, default=0.16)
    parser.add_argument("--max-tilt-deg", type=float, default=42.0)
    parser.add_argument("--minimum-pair-distance", type=float, default=3.0)
    parser.add_argument("--minimum-y-separation", type=float, default=2.35)
    parser.add_argument("--minimum-attitude-separation-deg", type=float, default=15.0)
    parser.add_argument("--mppi-samples", type=int, default=512)
    parser.add_argument("--no-gif", action="store_true")
    parser.add_argument("--environment", type=Path, default=DEFAULT_ENVIRONMENT)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--prepared", type=Path,
        default=DEFAULT_EXPERIMENT / "prepared_dataset.npz",
    )
    parser.add_argument(
        "--checkpoint", type=Path,
        default=DEFAULT_EXPERIMENT / "models/unet/best.pt",
    )
    parser.add_argument("--torch-python", type=Path, default=DEFAULT_TORCH_PYTHON)
    parser.add_argument(
        "--sampler-python",
        type=Path,
        default=DEFAULT_INTEGRATED_PYTHON,
        help="Python of the integrated sampler+cuRobo environment",
    )
    parser.add_argument(
        "--no-integrated-sampler",
        action="store_true",
        help="use the legacy two-subprocess sampler plus cuRobo filter path",
    )
    parser.add_argument(
        "--no-persistent-sampler",
        action="store_true",
        help="start the integrated sampler per pair attempt instead of "
        "keeping it resident",
    )
    parser.add_argument(
        "--esdf",
        type=Path,
        help="cached ESDF NPZ passed to the integrated sampler",
    )
    parser.add_argument(
        "--curobo-python",
        type=Path,
        default=DEFAULT_CUROBO_PYTHON,
        help="Python interpreter of the cuRobo environment used for GPU "
        "coarse filtering",
    )
    parser.add_argument(
        "--curobo-spheres",
        type=Path,
        default=DEFAULT_CUROBO_SPHERES,
        help="cuRobo collision-sphere YAML for the robot",
    )
    parser.add_argument(
        "--no-curobo-filter",
        action="store_true",
        help="skip the cuRobo coarse stage and COAL-check every candidate",
    )
    parser.add_argument(
        "--candidate-mode",
        choices=("staged", "curobo_first"),
        default="curobo_first",
        help=(
            "staged: cuRobo coarse accept then dense COAL for every remaining "
            "candidate; curobo_first: try cuRobo-ranked candidates one by one "
            "through spline fit + COAL validation until one succeeds"
        ),
    )
    parser.add_argument(
        "--ddim-steps",
        type=int,
        default=25,
        help="number of inference DDIM denoising steps (<= training steps)",
    )
    parser.add_argument(
        "pipeline_args", nargs=argparse.REMAINDER,
        help="extra execution-pipeline arguments after --",
    )
    args = parser.parse_args()
    if args.pipeline_args and args.pipeline_args[0] == "--":
        args.pipeline_args = args.pipeline_args[1:]
    if args.seed is not None and args.seed < 0:
        parser.error("seed must be non-negative")
    if (
        args.candidate_count <= 0
        or args.maximum_pair_attempts <= 0
        or args.maximum_execution_candidates <= 0
        or args.acceptance_clearance < 0.0
        or args.endpoint_clearance < args.acceptance_clearance
        or args.mppi_samples < 2
    ):
        parser.error("invalid candidate, clearance, attempt, or MPPI setting")
    return args


def candidate_metrics(
    paths: np.ndarray,
    checker: CoalCollisionChecker,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    required_clearance: float,
) -> list[dict[str, Any]]:
    records = []
    for index, path in enumerate(paths):
        workspace_valid = bool(
            np.all(np.isfinite(path))
            and np.all(path[:, :3] >= bounds_min)
            and np.all(path[:, :3] <= bounds_max)
        )
        error = None
        if workspace_valid:
            try:
                dense = dense_path(
                    path, translation_step=0.04, rotation_step_deg=3.0
                )
                clearance = checker.clearance(dense[:, :3], dense[:, 3:7])
                minimum_clearance = float(np.min(clearance))
            except (RuntimeError, ValueError) as caught:
                workspace_valid = False
                minimum_clearance = -math.inf
                error = str(caught)
        else:
            minimum_clearance = -math.inf
        metrics = path_metrics(path)
        records.append({
            "candidate_index": index,
            "workspace_valid": workspace_valid,
            "minimum_physical_clearance_m": minimum_clearance,
            "accepted_8cm": bool(
                workspace_valid and minimum_clearance >= required_clearance
            ),
            "error": error,
            **metrics,
        })
    return records


def candidate_metrics_staged(
    paths: np.ndarray,
    checker: CoalCollisionChecker,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    required_clearance: float,
    coarse_accept: np.ndarray | None = None,
    coarse_clearance: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Filter candidates with a cuRobo coarse stage in front of COAL.

    Candidates accepted by the conservative cuRobo sphere filter are certified
    at ``required_clearance`` without a COAL query; every other candidate is
    checked against the full URDF geometry in COAL as before.  The acceptance
    certificate is therefore at least as strict as the COAL-only path (the
    coarse stage only ever skips checks it can prove safe).
    """

    records = []
    for index, path in enumerate(paths):
        workspace_valid = bool(
            np.all(np.isfinite(path))
            and np.all(path[:, :3] >= bounds_min)
            and np.all(path[:, :3] <= bounds_max)
        )
        error = None
        if (
            workspace_valid
            and coarse_accept is not None
            and bool(coarse_accept[index])
        ):
            # The narrow cuRobo query certifies clearance >= required, and the
            # wide-eta estimate is a conservative lower bound, so the reported
            # value is a valid certified bound.
            estimated = (
                float(coarse_clearance[index])
                if coarse_clearance is not None
                else required_clearance
            )
            minimum_clearance = max(required_clearance, estimated)
            verified_by = "curobo_spheres"
        elif workspace_valid:
            try:
                dense = dense_path(
                    path, translation_step=0.04, rotation_step_deg=3.0
                )
                clearance = checker.clearance(dense[:, :3], dense[:, 3:7])
                minimum_clearance = float(np.min(clearance))
                verified_by = "coal"
            except (RuntimeError, ValueError) as caught:
                workspace_valid = False
                minimum_clearance = -math.inf
                verified_by = "coal"
                error = str(caught)
        else:
            minimum_clearance = -math.inf
            verified_by = "coal"
        metrics = path_metrics(path)
        records.append({
            "candidate_index": index,
            "workspace_valid": workspace_valid,
            "minimum_physical_clearance_m": minimum_clearance,
            "accepted_8cm": bool(
                workspace_valid and minimum_clearance >= required_clearance
            ),
            "verified_by": verified_by,
            "error": error,
            **metrics,
        })
    return records


def run_curobo_coarse_filter(
    args: argparse.Namespace,
    candidate_file: Path,
    environment_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Run the cuRobo GPU coarse filter in its dedicated environment."""

    curobo_python = args.curobo_python.expanduser().resolve()
    if not curobo_python.is_file():
        raise FileNotFoundError(f"cuRobo Python not found: {curobo_python}")
    command = [
        str(curobo_python),
        str(CUROBO_FILTER),
        "--candidates", str(candidate_file),
        "--environment", str(environment_path),
        "--spheres", str(args.curobo_spheres.expanduser().resolve()),
        "--activation-distance", str(args.acceptance_clearance),
        "--output", str(output_path),
    ]
    completed = subprocess.run(
        command,
        cwd=PROJECT_DIR,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "cuRobo coarse filter failed:\n" + completed.stderr.strip()
        )
    if not output_path.exists():
        raise RuntimeError("cuRobo coarse filter produced no output JSON")
    return json.loads(output_path.read_text(encoding="utf-8"))


def default_pipeline_arguments() -> argparse.Namespace:
    """Build the execution-pipeline argument defaults for in-process fitting."""

    import run_overfit_cube_single_pipeline as pipeline_module

    previous_argv = sys.argv[:]
    sys.argv = ["run_overfit_cube_single_pipeline.py"]
    try:
        return pipeline_module.parse_args()
    finally:
        sys.argv = previous_argv


def fit_validate_candidate(
    validation_planner: OMPLSE3Planner,
    pipeline_args: argparse.Namespace,
    generated_paths: np.ndarray,
    candidate_index: int,
    sampling_time_s: float,
    required_clearance: float,
    selected_path: Path,
    source: str,
) -> tuple[bool, dict[str, Any], str | None, float]:
    """Fit and COAL-validate one candidate; returns (ok, record, error, time)."""

    from run_overfit_cube_single_pipeline import plan_external_diffusion_path

    path = generated_paths[candidate_index]
    np.savez_compressed(
        selected_path,
        poses_wxyz=path,
        candidate_index=np.asarray(candidate_index),
        sampling_time_s=np.asarray(sampling_time_s),
        source=np.asarray(source),
    )
    metrics = path_metrics(path)
    record: dict[str, Any] = {
        "candidate_index": int(candidate_index),
        "accepted_8cm": True,
        "minimum_physical_clearance_m": required_clearance,
        "verified_by": "curobo_spheres+spline_fit",
        **{key: metrics[key] for key in (
            "translation_length_m",
            "rotation_length_deg",
            "position_acceleration_rms",
            "position_jerk_rms",
        )},
    }
    started = time.monotonic()
    try:
        plan_external_diffusion_path(
            validation_planner, selected_path.resolve(), pipeline_args
        )
        return True, record, None, time.monotonic() - started
    except (RuntimeError, ValueError, np.linalg.LinAlgError) as error:
        return False, record, str(error), time.monotonic() - started


class SamplerClient:
    """Resident or one-shot sampler+cuRobo-filter backend for the demo."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self._process: subprocess.Popen[str] | None = None
        self.integrated = False
        self.persistent = False
        if not args.no_integrated_sampler:
            integrated_python = args.sampler_python.expanduser().resolve()
            if integrated_python.is_file():
                self.integrated = True
                if not args.no_persistent_sampler:
                    self._start_persistent(integrated_python)

    def _start_persistent(self, integrated_python: Path) -> None:
        args = self.args
        command = [
            str(integrated_python),
            str(SAMPLE_AND_FILTER),
            "--serve",
            "--prepared", str(args.prepared.expanduser().resolve()),
            "--checkpoint", str(args.checkpoint.expanduser().resolve()),
            "--environment", str(args.environment.expanduser().resolve()),
            "--spheres", str(args.curobo_spheres.expanduser().resolve()),
            "--activation-distance", str(args.acceptance_clearance),
            "--ddim-steps", str(args.ddim_steps),
            "--device", "auto",
        ]
        if args.esdf is not None:
            command += ["--esdf", str(args.esdf.expanduser().resolve())]
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        environment.pop("LD_LIBRARY_PATH", None)
        self._process = subprocess.Popen(
            command,
            cwd=PROJECT_DIR,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=environment,
        )
        assert self._process.stdout is not None
        ready = False
        for _ in range(600):  # allow up to ~60 s for model/scene init
            line = self._process.stdout.readline()
            if not line:
                break
            if line.strip() == "READY":
                ready = True
                break
        if not ready:
            self.close()
            raise RuntimeError(
                "integrated sampler worker did not become ready; "
                "falling back to one-shot mode"
            )
        self.persistent = True

    def request(
        self,
        start_pose: np.ndarray,
        goal_pose: np.ndarray,
        seed: int,
        candidate_count: int,
        output_dir: Path,
    ) -> dict[str, Any]:
        """Generate and coarse-filter candidates, returning result metadata."""

        if self.integrated:
            if self.persistent:
                return self._request_persistent(
                    start_pose, goal_pose, seed, candidate_count, output_dir
                )
            return self._request_one_shot(
                start_pose, goal_pose, seed, candidate_count, output_dir
            )
        return self._request_legacy(
            start_pose, goal_pose, seed, candidate_count, output_dir
        )

    def _request_persistent(
        self,
        start_pose: np.ndarray,
        goal_pose: np.ndarray,
        seed: int,
        candidate_count: int,
        output_dir: Path,
    ) -> dict[str, Any]:
        assert self._process is not None and self._process.stdin is not None
        assert self._process.stdout is not None
        request = {
            "start_pose": np.asarray(start_pose, dtype=np.float64).tolist(),
            "goal_pose": np.asarray(goal_pose, dtype=np.float64).tolist(),
            "seed": int(seed),
            "candidate_count": int(candidate_count),
            "output_dir": str(output_dir.expanduser().resolve()),
        }
        self._process.stdin.write(json.dumps(request) + "\n")
        self._process.stdin.flush()
        response_line = self._process.stdout.readline()
        if not response_line:
            raise RuntimeError("integrated sampler worker closed the pipe")
        response = json.loads(response_line)
        if not response.get("ok", False):
            raise RuntimeError(
                f"integrated sampler failed: {response.get('error')}"
            )
        return response

    def _request_one_shot(
        self,
        start_pose: np.ndarray,
        goal_pose: np.ndarray,
        seed: int,
        candidate_count: int,
        output_dir: Path,
    ) -> dict[str, Any]:
        args = self.args
        command = [
            str(args.sampler_python.expanduser().resolve()),
            str(SAMPLE_AND_FILTER),
            "--prepared", str(args.prepared.expanduser().resolve()),
            "--checkpoint", str(args.checkpoint.expanduser().resolve()),
            "--environment", str(args.environment.expanduser().resolve()),
            "--spheres", str(args.curobo_spheres.expanduser().resolve()),
            "--activation-distance", str(args.acceptance_clearance),
            "--ddim-steps", str(args.ddim_steps),
            "--output-dir", str(output_dir.expanduser().resolve()),
            "--start-pose", *map(str, np.asarray(start_pose, dtype=np.float64)),
            "--goal-pose", *map(str, np.asarray(goal_pose, dtype=np.float64)),
            "--candidate-count", str(candidate_count),
            "--sampling-batch-size", str(min(candidate_count, 32)),
            "--seed", str(seed),
            "--ddim-steps", str(args.ddim_steps),
            "--device", "auto",
        ]
        if args.esdf is not None:
            command += ["--esdf", str(args.esdf.expanduser().resolve())]
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        environment.pop("LD_LIBRARY_PATH", None)
        completed = subprocess.run(
            command,
            cwd=PROJECT_DIR,
            capture_output=True,
            text=True,
            env=environment,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "integrated sampler failed:\n" + completed.stderr.strip()
            )
        for line in completed.stdout.splitlines():
            if line.startswith("SAMPLE_FILTER="):
                return json.loads(line[len("SAMPLE_FILTER="):])
        raise RuntimeError("integrated sampler produced no summary line")

    def _request_legacy(
        self,
        start_pose: np.ndarray,
        goal_pose: np.ndarray,
        seed: int,
        candidate_count: int,
        output_dir: Path,
    ) -> dict[str, Any]:
        args = self.args
        candidate_file = output_dir.expanduser().resolve() / "diffusion_candidates.npz"
        command = [
            str(args.torch_python.expanduser().resolve()),
            str(SAMPLER),
            "--prepared", str(args.prepared.expanduser().resolve()),
            "--checkpoint", str(args.checkpoint.expanduser().resolve()),
            "--output", str(candidate_file),
            "--start-pose", *map(str, np.asarray(start_pose, dtype=np.float64)),
            "--goal-pose", *map(str, np.asarray(goal_pose, dtype=np.float64)),
            "--candidate-count", str(candidate_count),
            "--sampling-batch-size", str(min(candidate_count, 32)),
            "--seed", str(seed),
            "--device", "auto",
        ]
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        environment.pop("LD_LIBRARY_PATH", None)
        completed = subprocess.run(
            command,
            cwd=PROJECT_DIR,
            capture_output=True,
            text=True,
            env=environment,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "PyTorch sampler failed:\n" + completed.stderr.strip()
            )
        with np.load(candidate_file) as payload:
            sampling_time_s = float(payload["sampling_time_s"])
        coarse_accept: list[bool] = [True] * candidate_count
        coarse_clearance = [float("nan")] * candidate_count
        if not args.no_curobo_filter:
            curobo_output = output_dir.expanduser().resolve() / "curobo_coarse_filter.json"
            curobo_result = run_curobo_coarse_filter(
                args, candidate_file, args.environment.expanduser().resolve(),
                curobo_output,
            )
            coarse_accept = curobo_result["coarse_accept"]
            coarse_clearance = curobo_result[
                "min_sphere_clearance_estimate_m"
            ]
        return {
            "output_dir": str(output_dir.expanduser().resolve()),
            "candidate_file": str(candidate_file),
            "candidate_count": candidate_count,
            "sampling_time_s": sampling_time_s,
            "coarse_accept": coarse_accept,
            "min_sphere_clearance_estimate_m": coarse_clearance,
            "coarse_accept_count": int(sum(coarse_accept)),
        }

    def close(self) -> None:
        if self._process is None:
            return
        try:
            if self._process.stdin is not None:
                self._process.stdin.write("quit\n")
                self._process.stdin.flush()
            self._process.wait(timeout=10)
        except Exception:  # noqa: BLE001 - best-effort shutdown
            self._process.kill()
        self._process = None


def run_streaming(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        return process.wait()


def main() -> None:
    args = parse_args()
    seed = args.seed if args.seed is not None else secrets.randbelow(2**31)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    root = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else (PROJECT_DIR / "results" / f"unet_guidance_demo_{stamp}_{seed}")
    )
    if root.exists():
        raise FileExistsError(f"output directory already exists: {root}")
    root.mkdir(parents=True)
    environment_path = args.environment.expanduser().resolve()
    urdf_path = args.urdf.expanduser().resolve()
    environment_data, environment = load_environment(environment_path)
    checker = CoalCollisionChecker.from_urdf(
        urdf_path, environment, safety_margin=0.0
    )
    sampling_bounds = environment_data["sampling_space"]["position_bounds"]
    bounds_min = np.asarray(sampling_bounds["min"], dtype=np.float64)
    bounds_max = np.asarray(sampling_bounds["max"], dtype=np.float64)
    validation_planner: OMPLSE3Planner | None = None
    if args.candidate_mode == "curobo_first":
        validation_checker = CoalCollisionChecker.from_urdf(
            urdf_path, environment, safety_margin=args.acceptance_clearance
        )
        validation_planner = OMPLSE3Planner(
            bounds_min=bounds_min,
            bounds_max=bounds_max,
            obstacles=(),
            vehicle_radius=0.0,
            safety_margin=0.0,
            seed=seed,
            collision_checker=validation_checker,
        )
    task = environment_data.get("task_sampling")
    if not task:
        raise RuntimeError("environment has no task_sampling start/goal regions")
    south_min = np.maximum(bounds_min, np.asarray(task["south_region"]["min"]))
    south_max = np.minimum(bounds_max, np.asarray(task["south_region"]["max"]))
    north_min = np.maximum(bounds_min, np.asarray(task["north_region"]["min"]))
    north_max = np.minimum(bounds_max, np.asarray(task["north_region"]["max"]))
    planner = OMPLSE3Planner(
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        obstacles=(),
        vehicle_radius=0.0,
        safety_margin=0.0,
        seed=seed,
        collision_checker=checker,
    )
    rng = np.random.default_rng(seed)
    pair_attempts: list[dict[str, Any]] = []
    accepted_records: list[dict[str, Any]] | None = None
    generated_paths: np.ndarray | None = None
    selected_start = selected_goal = None
    candidate_file = root / "diffusion_candidates.npz"
    sampler = SamplerClient(args)
    try:
      for pair_attempt in range(1, args.maximum_pair_attempts + 1):
        start, start_tries, start_clearance = sample_valid_pose(
            rng, planner, south_min, south_max,
            max_tilt_deg=args.max_tilt_deg,
            minimum_clearance=args.endpoint_clearance,
            maximum_attempts=3000,
        )
        goal, goal_tries, goal_clearance = sample_valid_pose(
            rng, planner, north_min, north_max,
            max_tilt_deg=args.max_tilt_deg,
            minimum_clearance=args.endpoint_clearance,
            maximum_attempts=3000,
        )
        separation = float(np.linalg.norm(goal.position - start.position))
        y_separation = float(abs(goal.position[1] - start.position[1]))
        attitude_deg = math.degrees(
            quaternion_distance(start.quaternion, goal.quaternion)
        )
        record: dict[str, Any] = {
            "pair_attempt": pair_attempt,
            "start_pose": np.concatenate((start.position, start.quaternion)),
            "goal_pose": np.concatenate((goal.position, goal.quaternion)),
            "start_sampling_attempts": start_tries,
            "goal_sampling_attempts": goal_tries,
            "start_clearance_m": start_clearance,
            "goal_clearance_m": goal_clearance,
            "position_separation_m": separation,
            "y_separation_m": y_separation,
            "attitude_separation_deg": attitude_deg,
        }
        pair_attempts.append(record)
        if (
            separation < args.minimum_pair_distance
            or y_separation < args.minimum_y_separation
            or attitude_deg < args.minimum_attitude_separation_deg
        ):
            record["result"] = "rejected_separation"
            print(f"pair {pair_attempt}: rejected by separation", flush=True)
            continue
        sampler_seed = int(
            np.random.SeedSequence([seed, pair_attempt, 9000])
            .generate_state(1, dtype=np.uint32)[0] & 0x7FFFFFFF
        )
        print(
            f"pair {pair_attempt}: sampling {args.candidate_count} guided paths",
            flush=True,
        )
        record["sampler_seed"] = sampler_seed
        sampler_started = time.monotonic()
        try:
            result = sampler.request(
                np.concatenate((start.position, start.quaternion)),
                np.concatenate((goal.position, goal.quaternion)),
                sampler_seed,
                args.candidate_count,
                root,
            )
        except Exception as error:
            record["result"] = "sampling_failed"
            record["sampler_error"] = str(error)
            write_json(root / "pair_attempts.json", pair_attempts)
            raise RuntimeError(
                "sampler failed before candidate validation:\n" + str(error)
            )
        record["sampler_wall_time_s"] = round(
            time.monotonic() - sampler_started, 4
        )
        record["sampling_time_s"] = result["sampling_time_s"]
        record["curobo_filter_time_s"] = result.get(
            "curobo_filter_time_s", 0.0
        )
        candidate_file = Path(result["candidate_file"])
        with np.load(candidate_file) as payload:
            generated_paths = payload["poses_wxyz"].astype(np.float64)
        coarse_accept: np.ndarray | None = None
        coarse_clearance: np.ndarray | None = None
        if not args.no_curobo_filter:
            coarse_accept = np.asarray(result["coarse_accept"], dtype=bool)
            coarse_clearance = np.asarray(
                result["min_sphere_clearance_estimate_m"], dtype=np.float64
            )
            record["curobo_coarse_accept_count"] = int(coarse_accept.sum())
            print(
                f"pair {pair_attempt}: cuRobo coarse accepts "
                f"{int(coarse_accept.sum())}/{len(coarse_accept)}",
                flush=True,
            )
        record["candidate_count"] = len(generated_paths)
        if (
            args.candidate_mode == "curobo_first"
            and coarse_accept is not None
        ):
            order = [
                index for index, ok in enumerate(coarse_accept) if bool(ok)
            ]
            order.sort(key=lambda index: -float(coarse_clearance[index]))
            record["candidate_mode"] = "curobo_first"
            record["curobo_candidate_order"] = order
            fit_attempts: list[dict[str, Any]] = []
            fit_records: list[dict[str, Any]] = []
            filter_started = time.monotonic()
            pipeline_args = default_pipeline_arguments()
            assert validation_planner is not None
            for attempt_index, candidate_index in enumerate(
                order[: args.maximum_execution_candidates]
            ):
                selected_path = (
                    root / f"selection_candidate_{attempt_index:02d}.npz"
                )
                ok, fit_record, error, fit_time = fit_validate_candidate(
                    validation_planner,
                    pipeline_args,
                    generated_paths,
                    candidate_index,
                    result["sampling_time_s"],
                    args.acceptance_clearance,
                    selected_path,
                    "U-Net diffusion + inference guidance, cuRobo-ranked",
                )
                fit_attempts.append({
                    "candidate_index": candidate_index,
                    "ok": ok,
                    "fit_time_s": round(fit_time, 4),
                    "error": error,
                })
                print(
                    f"pair {pair_attempt}: candidate {candidate_index} "
                    f"spline fit+COAL {'OK' if ok else 'FAILED'} "
                    f"({fit_time:.2f}s)",
                    flush=True,
                )
                if ok:
                    fit_records.append(fit_record)
                    break
            record["fit_time_s"] = round(
                time.monotonic() - filter_started, 4
            )
            record["fit_attempts"] = fit_attempts
            accepted = fit_records
            if not accepted:
                print(
                    "no cuRobo-ranked candidate fit; falling back to staged "
                    "COAL filter",
                    flush=True,
                )
                filter_started = time.monotonic()
                records = candidate_metrics_staged(
                    generated_paths, checker, bounds_min, bounds_max,
                    args.acceptance_clearance, coarse_accept, coarse_clearance,
                )
                record["filter_time_s"] = round(
                    time.monotonic() - filter_started, 4
                )
                accepted = [item for item in records if item["accepted_8cm"]]
                accepted_records = records
            else:
                record["filter_time_s"] = record["fit_time_s"]
                accepted_records = fit_records
            record["accepted_candidate_count"] = len(accepted)
            record["result"] = "accepted" if accepted else "no_safe_candidate"
            print(
                f"pair {pair_attempt}: {len(accepted)} feasible candidate(s)",
                flush=True,
            )
            if accepted:
                selected_start, selected_goal = start, goal
                break
        else:
            filter_started = time.monotonic()
            records = candidate_metrics_staged(
                generated_paths, checker, bounds_min, bounds_max,
                args.acceptance_clearance, coarse_accept, coarse_clearance,
            )
            record["filter_time_s"] = round(
                time.monotonic() - filter_started, 4
            )
            accepted = [item for item in records if item["accepted_8cm"]]
            record["accepted_candidate_count"] = len(accepted)
            record["result"] = "accepted" if accepted else "no_safe_candidate"
            print(
                f"pair {pair_attempt}: {len(accepted)}/{len(records)} "
                f"candidates meet {100 * args.acceptance_clearance:.0f} cm "
                "clearance",
                flush=True,
            )
            if accepted:
                accepted_records = records
                selected_start, selected_goal = start, goal
                break
    finally:
        sampler.close()
    write_json(root / "pair_attempts.json", pair_attempts)
    if accepted_records is None or generated_paths is None:
        raise RuntimeError(
            "no sampled endpoint pair produced a COAL-certified diffusion path; "
            f"see {root / 'pair_attempts.json'}"
        )
    ranked = sorted(
        (item for item in accepted_records if item["accepted_8cm"]),
        key=lambda item: (
            -item["minimum_physical_clearance_m"],
            item["translation_length_m"],
            item["position_acceleration_rms"],
        ),
    )
    write_json(root / "candidate_metrics.json", {
        "required_clearance_m": args.acceptance_clearance,
        "candidate_count": len(accepted_records),
        "accepted_count": len(ranked),
        "records": accepted_records,
    })
    successful_summary: Path | None = None
    selected_record: dict[str, Any] | None = None
    for execution_rank, record in enumerate(
        ranked[: args.maximum_execution_candidates]
    ):
        candidate_index = int(record["candidate_index"])
        execution_dir = root / f"execution_{execution_rank:02d}"
        selected_path = root / f"selected_path_{execution_rank:02d}.npz"
        with np.load(candidate_file) as payload:
            sampling_time_s = float(payload["sampling_time_s"])
        np.savez_compressed(
            selected_path,
            poses_wxyz=generated_paths[candidate_index],
            candidate_index=np.asarray(candidate_index),
            sampling_time_s=np.asarray(sampling_time_s),
            source=np.asarray("U-Net diffusion + inference guidance, COAL selected"),
        )
        command = [
            str(PIPELINE),
            "--external-path", str(selected_path),
            "--diffusion-candidates", str(candidate_file),
            "--diffusion-candidate-metrics", str(
                root / "candidate_metrics.json"
            ),
            "--output-dir", str(execution_dir),
            "--environment", str(environment_path),
            "--urdf", str(urdf_path),
            "--model", str(args.model.expanduser().resolve()),
            "--seed", str(seed),
            "--coal-safety-margin", str(args.acceptance_clearance),
            "--path-clearance", "0.0",
            "--endpoint-clearance", "0.0",
            "--mppi-samples", str(args.mppi_samples),
            *(["--no-gif"] if args.no_gif else []),
            *args.pipeline_args,
        ]
        print(
            f"executing candidate {candidate_index} "
            f"(clearance={record['minimum_physical_clearance_m']:.3f} m)",
            flush=True,
        )
        return_code = run_streaming(
            command, root / f"execution_{execution_rank:02d}.log"
        )
        summary_path = execution_dir / "single_pipeline_summary.json"
        if return_code == 0 and summary_path.exists():
            successful_summary = summary_path
            selected_record = record
            break
        print(
            f"candidate {candidate_index} execution failed; trying next safe candidate",
            flush=True,
        )
    if successful_summary is None or selected_record is None:
        raise RuntimeError(
            "safe candidates were generated but none passed B-spline/TOPP-RA/"
            f"MPPI execution; see logs under {root}"
        )
    execution = json.loads(successful_summary.read_text(encoding="utf-8"))
    overall = {
        "pipeline_success": True,
        "seed": seed,
        "output_directory": str(root),
        "start_pose": np.concatenate((
            selected_start.position, selected_start.quaternion
        )),
        "goal_pose": np.concatenate((
            selected_goal.position, selected_goal.quaternion
        )),
        "selected_candidate": selected_record,
        "execution_summary": str(successful_summary),
        "rerun_recording": execution["outputs"]["rerun_recording"],
        "tracking_gif": execution["outputs"]["mujoco_tracking_gif"],
        "tracking_plot": execution["outputs"]["tracking_plot"],
        "view_command": (
            f"rerun --port auto {execution['outputs']['rerun_recording']}"
        ),
    }
    write_json(root / "unet_guidance_demo_summary.json", overall)
    print("UNET_GUIDANCE_DEMO=" + json.dumps(overall, default=json_default))
    print("VIEW=" + overall["view_command"])


if __name__ == "__main__":
    main()
