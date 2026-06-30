"""Clean local LeRobot parquet datasets by dropping nearly static frames.

The script creates a cleaned copy by default. It keeps the first frame of each
episode, then drops a frame when all selected motion columns changed less than
``--threshold`` from the comparison frame. Metadata under ``meta/`` is updated
so the cleaned directory can be used as a local LeRobot dataset.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import tqdm


MotionNorm = Literal["max", "l2"]
CompareTo = Literal["previous", "last-kept"]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _episode_path(dataset_root: Path, info: dict[str, Any], episode_index: int) -> Path:
    chunk = episode_index // info["chunks_size"]
    rel = info["data_path"].format(episode_chunk=chunk, episode_index=episode_index)
    return dataset_root / rel


def _column_to_numpy(table: pa.Table, column: str) -> np.ndarray:
    return np.asarray(table[column].to_pylist(), dtype=np.float64)


def _motion_delta(a: np.ndarray, b: np.ndarray, norm: MotionNorm) -> float:
    diff = np.abs(a - b)
    if norm == "max":
        return float(np.max(diff))
    if norm == "l2":
        return float(np.linalg.norm(diff))
    raise ValueError(f"Unsupported norm: {norm}")


def _parse_zero_dim_specs(specs: list[str]) -> dict[str, list[int]]:
    parsed: dict[str, list[int]] = {}
    for spec in specs:
        if ":" not in spec:
            raise ValueError(f"Expected zero-dim spec like 'action:14,15', got: {spec}")
        column, dims = spec.split(":", 1)
        parsed.setdefault(column, []).extend(int(dim) for dim in dims.split(",") if dim)
    return parsed


def _has_zero_dim(
    arrays: dict[str, np.ndarray], zero_dim_specs: dict[str, list[int]], row: int, zero_atol: float
) -> bool:
    for column, dims in zero_dim_specs.items():
        values = arrays[column][row, dims]
        if np.any(np.isclose(values, 0.0, atol=zero_atol)):
            return True
    return False


def _build_keep_mask(
    table: pa.Table,
    *,
    motion_columns: list[str],
    threshold: float,
    norm: MotionNorm,
    compare_to: CompareTo,
    drop_zero_rows: bool,
    zero_dim_specs: dict[str, list[int]],
    zero_atol: float,
) -> tuple[np.ndarray, list[float]]:
    num_rows = table.num_rows
    if num_rows == 0:
        return np.zeros(0, dtype=bool), []

    needed_columns = sorted(set(motion_columns) | set(zero_dim_specs))
    arrays = {column: _column_to_numpy(table, column) for column in needed_columns}
    keep = np.zeros(num_rows, dtype=bool)
    keep[0] = True
    last_kept = 0
    motion_scores = [float("inf")]

    if drop_zero_rows or zero_dim_specs:
        first_is_zero = drop_zero_rows and any(np.allclose(values[0], 0.0) for values in arrays.values())
        first_has_zero_dim = _has_zero_dim(arrays, zero_dim_specs, 0, zero_atol)
        if (first_is_zero or first_has_zero_dim) and num_rows > 1:
            keep[0] = False
            motion_scores[0] = 0.0

    for i in range(1, num_rows):
        row_is_zero = drop_zero_rows and any(np.allclose(values[i], 0.0) for values in arrays.values())
        if row_is_zero or _has_zero_dim(arrays, zero_dim_specs, i, zero_atol):
            motion_scores.append(0.0)
            continue

        ref = i - 1 if compare_to == "previous" else last_kept
        score = max(_motion_delta(arrays[column][i], arrays[column][ref], norm) for column in motion_columns)
        motion_scores.append(score)
        if score > threshold:
            keep[i] = True
            last_kept = i

    if not np.any(keep):
        # Avoid producing empty episodes. Keep the first non-zero row when possible.
        fallback = 0
        if drop_zero_rows or zero_dim_specs:
            valid_rows = [
                i
                for i in range(num_rows)
                if not (drop_zero_rows and any(np.allclose(values[i], 0.0) for values in arrays.values()))
                and not _has_zero_dim(arrays, zero_dim_specs, i, zero_atol)
            ]
            fallback = valid_rows[0] if valid_rows else 0
        keep[fallback] = True

    return keep, motion_scores


def _replace_column(table: pa.Table, name: str, values: np.ndarray) -> pa.Table:
    idx = table.schema.get_field_index(name)
    if idx < 0:
        return table
    field = table.schema.field(idx)
    return table.set_column(idx, field, pa.array(values, type=field.type))


def _rewrite_indices(table: pa.Table, *, episode_index: int, global_start: int, fps: float) -> pa.Table:
    length = table.num_rows
    frame_index = np.arange(length, dtype=np.int64)
    table = _replace_column(table, "frame_index", frame_index)
    table = _replace_column(table, "timestamp", (frame_index / fps).astype(np.float32))
    table = _replace_column(table, "episode_index", np.full(length, episode_index, dtype=np.int64))
    table = _replace_column(table, "index", np.arange(global_start, global_start + length, dtype=np.int64))
    return table


def _vector_float_columns(info: dict[str, Any]) -> list[str]:
    columns: list[str] = []
    for name, feature in info["features"].items():
        if feature.get("dtype") not in {"float32", "float64"}:
            continue
        shape = feature.get("shape") or []
        if len(shape) == 1 and int(shape[0]) > 1:
            columns.append(name)
    return columns


def _stats_for_array(values: np.ndarray, *, include_quantiles: bool, count_as_list: bool) -> dict[str, Any]:
    quantiles = np.quantile(values, [0.01, 0.10, 0.50, 0.90, 0.99], axis=0)
    stats = {
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "count": [int(values.shape[0])] if count_as_list else int(values.shape[0]),
    }
    if include_quantiles:
        stats.update(
            {
                "q01": quantiles[0].tolist(),
                "q10": quantiles[1].tolist(),
                "q50": quantiles[2].tolist(),
                "q90": quantiles[3].tolist(),
                "q99": quantiles[4].tolist(),
            }
        )
    return stats


def _episode_stats(table: pa.Table, columns: list[str]) -> dict[str, Any]:
    stats = {}
    for column in columns:
        if table.schema.get_field_index(column) >= 0:
            stats[column] = _stats_for_array(
                _column_to_numpy(table, column), include_quantiles=False, count_as_list=True
            )
    return stats


def _global_stats(tables: list[pa.Table], columns: list[str]) -> dict[str, Any]:
    stats = {}
    for column in columns:
        chunks = [_column_to_numpy(table, column) for table in tables if table.schema.get_field_index(column) >= 0]
        if chunks:
            stats[column] = _stats_for_array(
                np.concatenate(chunks, axis=0), include_quantiles=True, count_as_list=False
            )
    return stats


def _prepare_output(dataset_root: Path, output_root: Path, *, overwrite: bool) -> None:
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output_root}. Pass --overwrite to replace it.")
        shutil.rmtree(output_root)

    def ignore(_: str, names: list[str]) -> set[str]:
        return {name for name in names if name in {"data", ".clean_static_frames_backups"}}

    shutil.copytree(dataset_root, output_root, ignore=ignore)
    (output_root / "data").mkdir(parents=True, exist_ok=True)


def _backup_in_place(dataset_root: Path, backup_root: Path, *, overwrite: bool) -> None:
    if backup_root.exists():
        if not overwrite:
            raise FileExistsError(f"Backup already exists: {backup_root}. Pass --overwrite to replace it.")
        shutil.rmtree(backup_root)
    shutil.copytree(dataset_root, backup_root)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path, help="Local LeRobot dataset root.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Cleaned output dataset root. Defaults to <dataset_root>_cleaned.",
    )
    parser.add_argument("--in-place", action="store_true", help="Rewrite dataset_root instead of creating a copy.")
    parser.add_argument(
        "--backup-root",
        type=Path,
        default=None,
        help="Backup directory for --in-place. Defaults to <dataset_root>_backup_before_clean.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing output or backup directories.")
    parser.add_argument(
        "--motion-columns",
        nargs="+",
        default=["observation.state", "action"],
        help="Vector columns used to decide whether adjacent frames moved.",
    )
    parser.add_argument("--threshold", type=float, default=1e-3, help="Motion threshold for dropping a frame.")
    parser.add_argument("--norm", choices=["max", "l2"], default="max")
    parser.add_argument(
        "--compare-to",
        choices=["previous", "last-kept"],
        default="previous",
        help="Comparison frame for motion. last-kept preserves slow cumulative motion better.",
    )
    parser.add_argument(
        "--drop-zero-rows",
        action="store_true",
        help="Also drop rows where any motion column is an all-zero vector.",
    )
    parser.add_argument(
        "--drop-zero-dims",
        nargs="*",
        default=[],
        help="Drop rows where any listed vector dimension is zero, e.g. action:14,15 observation.state:14,15.",
    )
    parser.add_argument("--zero-atol", type=float, default=1e-8, help="Tolerance for --drop-zero-dims.")
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    info_path = dataset_root / "meta/info.json"
    episodes_path = dataset_root / "meta/episodes.jsonl"
    if not info_path.exists() or not episodes_path.exists():
        raise FileNotFoundError(f"Expected LeRobot meta files under: {dataset_root / 'meta'}")

    info = json.loads(info_path.read_text())
    episodes = _load_jsonl(episodes_path)
    fps = float(info.get("fps", 30))
    zero_dim_specs = _parse_zero_dim_specs(args.drop_zero_dims)

    if args.in_place:
        output_root = dataset_root
        backup_root = args.backup_root or dataset_root.with_name(f"{dataset_root.name}_backup_before_clean")
        _backup_in_place(dataset_root, backup_root, overwrite=args.overwrite)
        print(f"Backed up original dataset to: {backup_root}")
    else:
        output_root = (args.output_root or dataset_root.with_name(f"{dataset_root.name}_cleaned")).resolve()
        _prepare_output(dataset_root, output_root, overwrite=args.overwrite)

    cleaned_episodes: list[dict[str, Any]] = []
    cleaned_episode_stats: list[dict[str, Any]] = []
    cleaned_tables: list[pa.Table] = []
    vector_stat_columns = _vector_float_columns(info)
    global_index = 0
    total_in = 0
    total_out = 0

    for episode in tqdm.tqdm(episodes, desc="Cleaning episodes"):
        episode_index = int(episode["episode_index"])
        source_path = _episode_path(dataset_root, info, episode_index)
        output_path = _episode_path(output_root, info, episode_index)
        table = pq.read_table(source_path)

        missing = [
            column
            for column in sorted(set(args.motion_columns) | set(zero_dim_specs))
            if table.schema.get_field_index(column) < 0
        ]
        if missing:
            raise KeyError(f"{source_path} is missing motion columns: {missing}")

        keep, scores = _build_keep_mask(
            table,
            motion_columns=args.motion_columns,
            threshold=args.threshold,
            norm=args.norm,
            compare_to=args.compare_to,
            drop_zero_rows=args.drop_zero_rows,
            zero_dim_specs=zero_dim_specs,
            zero_atol=args.zero_atol,
        )
        indices = np.flatnonzero(keep)
        cleaned = table.take(pa.array(indices, type=pa.int64()))
        cleaned = _rewrite_indices(cleaned, episode_index=episode_index, global_start=global_index, fps=fps)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(cleaned, output_path)

        length = cleaned.num_rows
        global_index += length
        total_in += table.num_rows
        total_out += length
        cleaned_tables.append(cleaned)

        new_episode = dict(episode)
        new_episode["length"] = length
        cleaned_episodes.append(new_episode)
        cleaned_episode_stats.append({"episode_index": episode_index, "stats": _episode_stats(cleaned, vector_stat_columns)})

        dropped = table.num_rows - length
        finite_scores = [score for score in scores if np.isfinite(score)]
        max_score = max(finite_scores) if finite_scores else 0.0
        print(
            f"episode {episode_index:06d}: {table.num_rows} -> {length} "
            f"(dropped {dropped}, max_motion={max_score:.6g})"
        )

    info["total_frames"] = total_out
    info["total_episodes"] = len(cleaned_episodes)
    info["total_chunks"] = max((int(ep["episode_index"]) // info["chunks_size"] for ep in cleaned_episodes), default=0) + 1
    if "splits" in info and "train" in info["splits"]:
        info["splits"]["train"] = f"0:{len(cleaned_episodes)}"

    meta_dir = output_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n")
    _write_jsonl(meta_dir / "episodes.jsonl", cleaned_episodes)
    _write_jsonl(meta_dir / "episodes_stats.jsonl", cleaned_episode_stats)
    (meta_dir / "stats.json").write_text(
        json.dumps(_global_stats(cleaned_tables, vector_stat_columns), ensure_ascii=False, indent=2) + "\n"
    )

    print(f"Input frames: {total_in}")
    print(f"Output frames: {total_out}")
    print(f"Dropped frames: {total_in - total_out}")
    print(f"Cleaned dataset: {output_root}")


if __name__ == "__main__":
    main()
