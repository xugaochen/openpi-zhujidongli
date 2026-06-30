"""Compute one OpenPI norm_stats.json from multiple local LeRobot parquet datasets.

This helper follows the same core logic as compute_norm_stats_from_lerobot_parquet.py:
it reads vector columns directly from local LeRobot parquet files, skips image
decoding, expands actions across the action horizon, and writes OpenPI-compatible
normalization statistics.
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


DEFAULT_STATE_KEY = "observation.state"
DEFAULT_ACTION_KEY = "action"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_episode_metadata(dataset_root: Path) -> list[dict[str, Any]]:
    jsonl_path = dataset_root / "meta/episodes.jsonl"
    if jsonl_path.exists():
        return _load_jsonl(jsonl_path)

    parquet_dir = dataset_root / "meta/episodes"
    if parquet_dir.is_dir():
        episodes: list[dict[str, Any]] = []
        for path in sorted(parquet_dir.rglob("*.parquet")):
            episodes.extend(pq.read_table(path).to_pylist())
        return sorted(episodes, key=lambda episode: int(episode["episode_index"]))

    raise FileNotFoundError(f"Expected {jsonl_path} or parquet files under {parquet_dir}")


def _column_to_numpy(path: Path, column: str) -> np.ndarray:
    table = pq.read_table(path, columns=[column])
    return np.asarray(table[column].to_pylist(), dtype=np.float32)


def _episode_path(dataset_root: Path, info: dict[str, Any], episode: dict[str, Any]) -> Path:
    episode_index = int(episode["episode_index"])
    episode_chunk = episode_index // int(info["chunks_size"])
    data_chunk = int(episode.get("data/chunk_index", episode_chunk))
    data_file = int(episode.get("data/file_index", episode_index))
    rel = info["data_path"].format(
        episode_chunk=episode_chunk,
        episode_index=episode_index,
        chunk_index=data_chunk,
        file_index=data_file,
    )
    return dataset_root / rel


def _action_sequences(actions: np.ndarray, horizon: int) -> np.ndarray:
    frame_indices = np.arange(actions.shape[0])[:, None]
    offsets = np.arange(horizon)[None, :]
    query_indices = np.clip(frame_indices + offsets, 0, actions.shape[0] - 1)
    return actions[query_indices]


def _save_norm_stats(output_path: Path, norm_stats: dict[str, normalize.NormStats]) -> None:
    if output_path.suffix == ".json":
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(normalize.serialize_json(norm_stats))
    else:
        normalize.save(output_path, norm_stats)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_paths",
        type=Path,
        nargs="+",
        required=True,
        help="One or more local LeRobot dataset roots.",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        required=True,
        help="Output directory, or a full path ending in norm_stats.json.",
    )
    parser.add_argument("--action-horizon", type=int, default=50)
    parser.add_argument("--state-key", default=DEFAULT_STATE_KEY)
    parser.add_argument("--action-key", default=DEFAULT_ACTION_KEY)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional total frame cap across all datasets, useful for quick checks.",
    )
    args = parser.parse_args()

    stats = {"state": normalize.RunningStats(), "actions": normalize.RunningStats()}
    frames_seen = 0

    for dataset_root in args.data_paths:
        dataset_root = dataset_root.expanduser().resolve()
        info = json.loads((dataset_root / "meta/info.json").read_text())
        episodes = _load_episode_metadata(dataset_root)
        progress = tqdm.tqdm(episodes, desc=f"Computing stats: {dataset_root.name}")

        for episode in progress:
            if args.max_frames is not None and frames_seen >= args.max_frames:
                break

            path = _episode_path(dataset_root, info, episode)
            state = _column_to_numpy(path, args.state_key)
            actions = _column_to_numpy(path, args.action_key)

            if args.max_frames is not None:
                remaining = args.max_frames - frames_seen
                state = state[:remaining]
                actions = actions[:remaining]

            stats["state"].update(state)
            stats["actions"].update(_action_sequences(actions, args.action_horizon))
            frames_seen += state.shape[0]

    norm_stats = {key: value.get_statistics() for key, value in stats.items()}
    print(f"Processed {frames_seen} frame(s) from {len(args.data_paths)} dataset(s).")
    print(f"Writing stats to: {args.output_path}")
    _save_norm_stats(args.output_path, norm_stats)


if __name__ == "__main__":
    main()
