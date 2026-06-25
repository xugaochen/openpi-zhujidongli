"""Compute OpenPI norm stats directly from local LeRobot parquet files.

This is a local-data repair/interop helper for LeRobot parquet datasets that are
otherwise awkward to load through the pinned LeRobot + Hugging Face datasets
stack. It intentionally reads only vector columns needed for normalization and
skips image decoding entirely.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import tqdm

import openpi.shared.normalize as normalize
import openpi.training.config as _config


def _load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _column_to_numpy(path: Path, column: str) -> np.ndarray:
    table = pq.read_table(path, columns=[column])
    return np.asarray(table[column].to_pylist(), dtype=np.float32)


def _episode_path(dataset_root: Path, info: dict, episode_index: int) -> Path:
    chunk = episode_index // info["chunks_size"]
    rel = info["data_path"].format(episode_chunk=chunk, episode_index=episode_index)
    return dataset_root / rel


def _action_sequences(actions: np.ndarray, horizon: int) -> np.ndarray:
    frame_indices = np.arange(actions.shape[0])[:, None]
    offsets = np.arange(horizon)[None, :]
    query_indices = np.clip(frame_indices + offsets, 0, actions.shape[0] - 1)
    return actions[query_indices]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    config = _config.get_config(args.config_name)
    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    if tuple(data_config.action_sequence_keys) != ("action",):
        raise ValueError(f"Only action_sequence_keys=('action',) is supported, got {data_config.action_sequence_keys}")

    dataset_root = args.dataset_root
    if dataset_root is None:
        hf_lerobot_home = Path(os.environ.get("HF_LEROBOT_HOME", Path.home() / ".cache/huggingface/lerobot"))
        dataset_root = hf_lerobot_home / data_config.repo_id

    info = json.loads((dataset_root / "meta/info.json").read_text())
    episodes = _load_jsonl(dataset_root / "meta/episodes.jsonl")

    stats = {"state": normalize.RunningStats(), "actions": normalize.RunningStats()}
    frames_seen = 0

    for episode in tqdm.tqdm(episodes, desc="Computing stats"):
        path = _episode_path(dataset_root, info, int(episode["episode_index"]))
        state = _column_to_numpy(path, "observation.state")
        actions = _column_to_numpy(path, "action")

        if args.max_frames is not None:
            remaining = args.max_frames - frames_seen
            if remaining <= 0:
                break
            state = state[:remaining]
            actions = actions[:remaining]

        stats["state"].update(state)
        stats["actions"].update(_action_sequences(actions, config.model.action_horizon))
        frames_seen += state.shape[0]

    norm_stats = {key: value.get_statistics() for key, value in stats.items()}
    output_path = config.assets_dirs / data_config.repo_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    main()
