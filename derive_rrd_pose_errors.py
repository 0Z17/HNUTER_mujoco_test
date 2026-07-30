#!/usr/bin/env python3
"""Derive position and orientation error channels from a Rerun RRD.

The source of truth is the pair of Transform3D streams stored at
``world/uav/actual`` and ``world/uav/reference``.  The output keeps the
complete input recording and adds error channels under ``plots/tracking``.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pyarrow as pa
import rerun as rr
from numpy.typing import NDArray
from rerun.experimental import Chunk, LazyChunkStream, RrdReader, StoreEntry

from mppi.quaternion import (
    normalize_quaternion,
    quaternion_error_vector,
    quaternion_to_euler,
)


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

ACTUAL_POSE_PATH = "/world/uav/actual"
REFERENCE_POSE_PATH = "/world/uav/reference"
POSITION_ERROR_PATH = "/plots/tracking/position_error"
ORIENTATION_ERROR_PATH = "/plots/tracking/orientation_error"


@dataclass(frozen=True)
class PoseSeries:
    control_steps: IntArray
    sim_time_ns: IntArray
    positions: FloatArray
    quaternions_wxyz: FloatArray


@dataclass(frozen=True)
class DerivedErrors:
    position_components_m: FloatArray
    position_norm_m: FloatArray
    euler_deg: FloatArray
    so3_angle_deg: FloatArray
    so3_axis_reference: FloatArray


def _column(batch: pa.RecordBatch, name: str) -> pa.Array:
    try:
        return batch.column(batch.schema.names.index(name))
    except ValueError as error:
        raise ValueError(f"RRD chunk is missing required column {name!r}") from error


def _single_instance_vectors(column: pa.Array, width: int) -> FloatArray:
    rows = column.to_pylist()
    result = np.asarray(
        [
            row[0]
            if row is not None and len(row) == 1
            else np.full(width, np.nan)
            for row in rows
        ],
        dtype=np.float64,
    )
    if result.shape != (len(rows), width) or not np.all(np.isfinite(result)):
        raise ValueError(
            f"expected one finite {width}-vector per Transform3D row"
        )
    return result


def _read_pose_series(
    reader: RrdReader, recording: StoreEntry, entity_path: str
) -> PoseSeries:
    rows: list[tuple[int, int, FloatArray, FloatArray]] = []
    stream = reader.stream(store=recording).filter(content=entity_path)
    for chunk in stream:
        batch = chunk.to_record_batch()
        if "Transform3D:quaternion" not in batch.schema.names:
            continue
        control_steps = np.asarray(
            _column(batch, "control_step").to_pylist(), dtype=np.int64
        )
        sim_time_ns = np.asarray(
            _column(batch, "sim_time").cast(pa.int64()).to_pylist(),
            dtype=np.int64,
        )
        positions = _single_instance_vectors(
            _column(batch, "Transform3D:translation"), 3
        )
        quaternions_xyzw = _single_instance_vectors(
            _column(batch, "Transform3D:quaternion"), 4
        )
        quaternions_wxyz = quaternions_xyzw[:, [3, 0, 1, 2]]
        rows.extend(
            (
                int(step),
                int(time_ns),
                position,
                quaternion,
            )
            for step, time_ns, position, quaternion in zip(
                control_steps,
                sim_time_ns,
                positions,
                quaternions_wxyz,
            )
        )

    if not rows:
        raise ValueError(f"no Transform3D samples found at {entity_path}")
    rows.sort(key=lambda row: row[0])
    steps = np.asarray([row[0] for row in rows], dtype=np.int64)
    if len(np.unique(steps)) != len(steps):
        raise ValueError(f"duplicate control_step values at {entity_path}")
    return PoseSeries(
        control_steps=steps,
        sim_time_ns=np.asarray([row[1] for row in rows], dtype=np.int64),
        positions=np.stack([row[2] for row in rows]),
        quaternions_wxyz=normalize_quaternion(
            np.stack([row[3] for row in rows])
        ),
    )


def _derive_errors(
    actual: PoseSeries, reference: PoseSeries
) -> DerivedErrors:
    if not np.array_equal(actual.control_steps, reference.control_steps):
        raise ValueError("actual/reference control_step samples do not align")
    if not np.array_equal(actual.sim_time_ns, reference.sim_time_ns):
        raise ValueError("actual/reference sim_time samples do not align")

    position_components_m = actual.positions - reference.positions
    position_norm_m = np.linalg.norm(position_components_m, axis=1)

    actual_euler = quaternion_to_euler(actual.quaternions_wxyz)
    reference_euler = quaternion_to_euler(reference.quaternions_wxyz)
    euler_error = (
        actual_euler - reference_euler + np.pi
    ) % (2.0 * np.pi) - np.pi

    rotation_vectors = quaternion_error_vector(
        actual.quaternions_wxyz,
        reference.quaternions_wxyz,
    )
    so3_angle_rad = np.linalg.norm(rotation_vectors, axis=1)
    so3_axis_reference = np.divide(
        rotation_vectors,
        so3_angle_rad[:, np.newaxis],
        out=np.zeros_like(rotation_vectors),
        where=so3_angle_rad[:, np.newaxis] > 1.0e-12,
    )
    return DerivedErrors(
        position_components_m=position_components_m,
        position_norm_m=position_norm_m,
        euler_deg=np.degrees(euler_error),
        so3_angle_deg=np.degrees(so3_angle_rad),
        so3_axis_reference=so3_axis_reference,
    )


def _read_scalar_series(
    reader: RrdReader, recording: StoreEntry, entity_path: str
) -> tuple[IntArray, FloatArray] | None:
    rows: list[tuple[int, float]] = []
    stream = reader.stream(store=recording).filter(content=entity_path)
    for chunk in stream:
        batch = chunk.to_record_batch()
        if "Scalars:scalars" not in batch.schema.names:
            continue
        steps = _column(batch, "control_step").to_pylist()
        values = _column(batch, "Scalars:scalars").to_pylist()
        for step, value in zip(steps, values):
            if value is not None and len(value) == 1:
                rows.append((int(step), float(value[0])))
    if not rows:
        return None
    rows.sort(key=lambda row: row[0])
    return (
        np.asarray([row[0] for row in rows], dtype=np.int64),
        np.asarray([row[1] for row in rows], dtype=np.float64),
    )


def _validate_against_existing_channels(
    reader: RrdReader,
    recording: StoreEntry,
    steps: IntArray,
    errors: DerivedErrors,
) -> None:
    comparisons = (
        (
            "/plots/tracking/position_error_m",
            errors.position_norm_m,
            1.0e-5,
        ),
        (
            "/plots/tracking/attitude_error_deg",
            errors.so3_angle_deg,
            1.0e-4,
        ),
    )
    for entity_path, derived, tolerance in comparisons:
        existing = _read_scalar_series(reader, recording, entity_path)
        if existing is None:
            continue
        existing_steps, existing_values = existing
        if not np.array_equal(existing_steps, steps):
            raise ValueError(
                f"existing validation channel does not align: {entity_path}"
            )
        max_difference = float(np.max(np.abs(existing_values - derived)))
        if max_difference > tolerance:
            raise ValueError(
                f"derived values disagree with {entity_path}: "
                f"max difference {max_difference:.6g} exceeds {tolerance}"
            )


def _scalar_chunk(
    entity_path: str,
    steps: IntArray,
    sim_time_ns: IntArray,
    values: FloatArray,
) -> Chunk:
    return Chunk.from_columns(
        entity_path,
        indexes=(
            rr.TimeColumn("control_step", sequence=steps),
            rr.TimeColumn(
                "sim_time",
                duration=sim_time_ns.astype("timedelta64[ns]"),
            ),
        ),
        columns=rr.Scalars.columns(scalars=values),
    )


def _derived_chunks(
    source_path: Path,
    actual: PoseSeries,
    errors: DerivedErrors,
) -> list[Chunk]:
    steps = actual.control_steps
    times = actual.sim_time_ns
    channels: dict[str, FloatArray] = {
        f"{POSITION_ERROR_PATH}/norm_m": errors.position_norm_m,
        f"{POSITION_ERROR_PATH}/x_m": errors.position_components_m[:, 0],
        f"{POSITION_ERROR_PATH}/y_m": errors.position_components_m[:, 1],
        f"{POSITION_ERROR_PATH}/z_m": errors.position_components_m[:, 2],
        f"{ORIENTATION_ERROR_PATH}/euler_deg/roll": errors.euler_deg[:, 0],
        f"{ORIENTATION_ERROR_PATH}/euler_deg/pitch": errors.euler_deg[:, 1],
        f"{ORIENTATION_ERROR_PATH}/euler_deg/yaw": errors.euler_deg[:, 2],
        f"{ORIENTATION_ERROR_PATH}/so3/angle_deg": errors.so3_angle_deg,
        (
            f"{ORIENTATION_ERROR_PATH}/so3/axis_reference/x"
        ): errors.so3_axis_reference[:, 0],
        (
            f"{ORIENTATION_ERROR_PATH}/so3/axis_reference/y"
        ): errors.so3_axis_reference[:, 1],
        (
            f"{ORIENTATION_ERROR_PATH}/so3/axis_reference/z"
        ): errors.so3_axis_reference[:, 2],
    }
    chunks = [
        _scalar_chunk(path, steps, times, values)
        for path, values in channels.items()
    ]
    metadata = f"""# Derived pose-error channels

- Source RRD: `{source_path.name}`
- Samples: {len(steps)}
- Position components: `actual - reference`, expressed in world axes, meters.
- Euler error: wrapped `actual - reference` ZYX roll/pitch/yaw, degrees.
- SO(3) error: `R_reference^T R_actual`, shortest relative rotation.
- SO(3) angle: degrees in `[0, 180]`.
- SO(3) axis: unit axis expressed in the reference body frame; `[0, 0, 0]`
  when the angle is numerically zero.
"""
    chunks.append(
        Chunk.from_columns(
            "/metadata/derived_pose_errors",
            indexes=(),
            columns=rr.TextDocument.columns(
                text=[metadata],
                media_type=["text/markdown"],
            ),
        )
    )
    return chunks


def _rerun_cli() -> Path:
    beside_python = Path(sys.executable).with_name("rerun")
    if beside_python.is_file():
        return beside_python
    executable = shutil.which("rerun")
    if executable is None:
        raise RuntimeError(
            "could not find the rerun CLI beside Python or on PATH"
        )
    return Path(executable)


def _merge_rrds(
    input_path: Path,
    supplemental_path: Path,
    output_path: Path,
    temporary_directory: Path,
) -> None:
    merged_path = temporary_directory / output_path.name
    environment = os.environ.copy()
    environment["XDG_CACHE_HOME"] = str(temporary_directory / "cache")
    environment["XDG_CONFIG_HOME"] = str(temporary_directory / "config")
    result = subprocess.run(
        [
            str(_rerun_cli()),
            "rrd",
            "merge",
            "--output",
            str(merged_path),
            str(input_path),
            str(supplemental_path),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "rerun rrd merge failed:\n"
            + (result.stderr.strip() or result.stdout.strip())
        )
    if not merged_path.is_file() or merged_path.stat().st_size == 0:
        raise RuntimeError("rerun rrd merge did not create a nonempty output")
    os.replace(merged_path, output_path)


def _validate_output(
    output_path: Path,
    expected_recording_id: str,
    expected_samples: int,
) -> None:
    reader = RrdReader(output_path)
    recordings = reader.recordings()
    if not any(
        recording.recording_id == expected_recording_id
        for recording in recordings
    ):
        raise RuntimeError("output RRD does not contain the source recording")
    for entity_path in (
        f"{POSITION_ERROR_PATH}/norm_m",
        f"{ORIENTATION_ERROR_PATH}/euler_deg/roll",
        f"{ORIENTATION_ERROR_PATH}/so3/angle_deg",
        f"{ORIENTATION_ERROR_PATH}/so3/axis_reference/x",
    ):
        count = sum(
            chunk.num_rows
            for chunk in reader.stream().filter(content=entity_path)
            if "Scalars:scalars"
            in chunk.to_record_batch().schema.names
        )
        if count != expected_samples:
            raise RuntimeError(
                f"output channel {entity_path} has {count} rows; "
                f"expected {expected_samples}"
            )
    if not reader.blueprints():
        raise RuntimeError("output RRD lost the source blueprint")


def derive_rrd(
    input_path: Path, output_path: Path, *, force: bool = False
) -> DerivedErrors:
    input_path = input_path.resolve()
    output_path = output_path.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"input RRD does not exist: {input_path}")
    if input_path.suffix.lower() != ".rrd":
        raise ValueError("input path must use the .rrd extension")
    if output_path.suffix.lower() != ".rrd":
        raise ValueError("output path must use the .rrd extension")
    if input_path == output_path:
        raise ValueError("output path must differ from input path")
    if output_path.exists() and not force:
        raise FileExistsError(
            f"output already exists (pass --force to replace it): {output_path}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    reader = RrdReader(input_path)
    recordings = reader.recordings()
    if len(recordings) != 1:
        raise ValueError(
            f"expected exactly one recording store, found {len(recordings)}"
        )
    recording = recordings[0]
    actual = _read_pose_series(reader, recording, ACTUAL_POSE_PATH)
    reference = _read_pose_series(reader, recording, REFERENCE_POSE_PATH)
    errors = _derive_errors(actual, reference)
    _validate_against_existing_channels(
        reader, recording, actual.control_steps, errors
    )

    with tempfile.TemporaryDirectory(
        prefix=".derive_rrd_pose_errors-", dir=output_path.parent
    ) as directory:
        temporary_directory = Path(directory)
        supplemental_path = temporary_directory / "derived_channels.rrd"
        LazyChunkStream.from_iter(
            _derived_chunks(input_path, actual, errors)
        ).write_rrd(
            supplemental_path,
            application_id=recording.application_id,
            recording_id=recording.recording_id,
        )
        _merge_rrds(
            input_path,
            supplemental_path,
            output_path,
            temporary_directory,
        )

    _validate_output(
        output_path,
        recording.recording_id,
        len(actual.control_steps),
    )
    return errors


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Preserve an RRD and add position, Euler, and SO(3) pose-error "
            "channels derived from its actual/reference Transform3D streams."
        )
    )
    parser.add_argument("input", type=Path, help="source .rrd file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=(
            "output .rrd path (default: INPUT_pose_errors.rrd beside input)"
        ),
    )
    parser.add_argument(
        "--force", action="store_true", help="replace an existing output file"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    input_path: Path = args.input
    output_path = args.output or input_path.with_name(
        f"{input_path.stem}_pose_errors.rrd"
    )
    errors = derive_rrd(input_path, output_path, force=args.force)
    print(f"Output: {output_path.resolve()}")
    print(f"Samples: {len(errors.position_norm_m)}")
    print(
        "Position error RMSE / max: "
        f"{np.sqrt(np.mean(np.square(errors.position_norm_m))):.9f} / "
        f"{np.max(errors.position_norm_m):.9f} m"
    )
    print(
        "SO(3) error RMSE / max: "
        f"{np.sqrt(np.mean(np.square(errors.so3_angle_deg))):.9f} / "
        f"{np.max(errors.so3_angle_deg):.9f} deg"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
