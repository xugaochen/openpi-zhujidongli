#!/usr/bin/env python3
"""Compute OpenPI norm stats from a local LeRobot v2.1 parquet dataset.

The default arguments match the local Tron dataset used by the ``pi0_tron``
training config:

    /workspace/openpi/datasets/test_lerobot_v2.1

The output is an OpenPI-compatible ``norm_stats.json`` with ``state`` and
``actions`` entries, ready to place under the training assets directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import tqdm

import openpi.shared.normalize as normalize


DEFAULT_DATASET_ROOT = Path("/workspace/openpi/datasets/test_lerobot_v2.1")
DEFAULT_OUTPUT_PATH = Path("/workspace/openpi/assets/test_lerobot_v2.1")
DEFAULT_STATE_KEY = "observation.state"
DEFAULT_ACTION_KEY = "action"
DEFAULT_ACTION_HORIZON = 50


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_info(dataset_root: Path) -> dict[str, Any]:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing dataset metadata: {info_path}")
    return json.loads(info_path.read_text(encoding="utf-8"))


def _load_episodes(dataset_root: Path) -> list[dict[str, Any]]:
    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    if not episodes_path.is_file():
        raise FileNotFoundError(f"Missing episode metadata: {episodes_path}")
    return _load_jsonl(episodes_path)


def _episode_path(dataset_root: Path, info: dict[str, Any], episode: dict[str, Any]) -> Path:
    episode_index = int(episode["episode_index"])
    episode_chunk = episode_index // int(info["chunks_size"])
    data_chunk = int(episode.get("data/chunk_index", episode_chunk))
    data_file = int(episode.get("data/file_index", episode_index))
    relative_path = info["data_path"].format(
        episode_chunk=episode_chunk,
        episode_index=episode_index,
        chunk_index=data_chunk,
        file_index=data_file,
    )
    return dataset_root / relative_path


def _read_vector_column(path: Path, column: str) -> np.ndarray:
    try:
        table = pq.read_table(path, columns=[column])
    except Exception as exc:
        raise RuntimeError(f"Failed to read column '{column}' from {path}") from exc

    array = np.asarray(table[column].to_pylist(), dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Expected column '{column}' in {path} to be 2D, got shape {array.shape}")
    return array


def _action_sequences(actions: np.ndarray, horizon: int) -> np.ndarray:
    frame_indices = np.arange(actions.shape[0])[:, None]
    offsets = np.arange(horizon)[None, :]
    query_indices = np.clip(frame_indices + offsets, 0, actions.shape[0] - 1)
    return actions[query_indices]


def _save_norm_stats(output_path: Path, norm_stats: dict[str, normalize.NormStats]) -> None:
    if output_path.suffix == ".json":
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(normalize.serialize_json(norm_stats), encoding="utf-8")
    else:
        normalize.save(output_path, norm_stats)


def _check_feature(info: dict[str, Any], key: str) -> None:
    features = info.get("features", {})
    if key not in features:
        available = ", ".join(sorted(features))
        raise KeyError(f"Feature '{key}' not found in meta/info.json. Available features: {available}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute OpenPI-compatible norm_stats.json from a local LeRobot v2.1 parquet dataset."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help=f"Local LeRobot dataset root. Default: {DEFAULT_DATASET_ROOT}",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=(
            "Output directory or full path ending in norm_stats.json. "
            f"Default: {DEFAULT_OUTPUT_PATH}"
        ),
    )
    parser.add_argument("--state-key", default=DEFAULT_STATE_KEY)
    parser.add_argument("--action-key", default=DEFAULT_ACTION_KEY)
    parser.add_argument("--action-horizon", type=int, default=DEFAULT_ACTION_HORIZON)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional frame limit for quick validation runs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()

    info = _load_info(dataset_root)
    _check_feature(info, args.state_key)
    _check_feature(info, args.action_key)
    episodes = _load_episodes(dataset_root)

    stats = {
        "state": normalize.RunningStats(),
        "actions": normalize.RunningStats(),
    }
    frames_seen = 0

    progress = tqdm.tqdm(episodes, desc=f"Computing stats: {dataset_root.name}")
    for episode in progress:
        if args.max_frames is not None and frames_seen >= args.max_frames:
            break

        parquet_path = _episode_path(dataset_root, info, episode)
        if not parquet_path.is_file():
            raise FileNotFoundError(f"Episode parquet file not found: {parquet_path}")

        state = _read_vector_column(parquet_path, args.state_key)
        actions = _read_vector_column(parquet_path, args.action_key)
        if state.shape[0] != actions.shape[0]:
            raise ValueError(
                f"Frame count mismatch in {parquet_path}: state={state.shape[0]}, action={actions.shape[0]}"
            )

        if args.max_frames is not None:
            remaining = args.max_frames - frames_seen
            state = state[:remaining]
            actions = actions[:remaining]

        stats["state"].update(state)
        stats["actions"].update(_action_sequences(actions, args.action_horizon))
        frames_seen += int(state.shape[0])

    if frames_seen < 2:
        raise ValueError(f"Need at least 2 frames to compute norm stats, got {frames_seen}")

    norm_stats = {key: value.get_statistics() for key, value in stats.items()}
    _save_norm_stats(args.output_path.expanduser(), norm_stats)

    output_file = args.output_path if args.output_path.suffix == ".json" else args.output_path / "norm_stats.json"
    print(f"Processed {frames_seen} frame(s) from {len(episodes)} episode(s).")
    print(f"State dim: {norm_stats['state'].mean.shape[0]}")
    print(f"Action dim: {norm_stats['actions'].mean.shape[0]}, horizon: {args.action_horizon}")
    print(f"Wrote: {output_file}")


if __name__ == "__main__":
    main()
