"""Fix incompatible Hugging Face parquet metadata in local LeRobot data files.

Some LeRobot parquet files can contain embedded Hugging Face datasets feature
metadata with entries such as ``"_type": "List"``. Older ``datasets`` versions
used by this repo do not know that feature type and fail before reading the
Arrow schema. This script rewrites that metadata to ``"_type": "Sequence"`` so
vector columns remain typed and image columns keep their Image decoding metadata.

This script only touches ``data/**/*.parquet`` files under a local dataset root,
and only rewrites files whose Hugging Face metadata contains ``"_type": "List"``.
It runs as a dry run unless ``--apply`` is passed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


HF_METADATA_KEY = b"huggingface"
BAD_FEATURE_MARKER = b'"_type": "List"'


def _needs_fix(path: Path) -> bool:
    schema = pq.read_schema(path)
    metadata = schema.metadata or {}
    return BAD_FEATURE_MARKER in metadata.get(HF_METADATA_KEY, b"")


def _replace_list_with_sequence(obj: Any) -> None:
    if isinstance(obj, dict):
        if obj.get("_type") == "List":
            obj["_type"] = "Sequence"
        for value in obj.values():
            _replace_list_with_sequence(value)
    elif isinstance(obj, list):
        for value in obj:
            _replace_list_with_sequence(value)


def _rewrite_hf_metadata(path: Path, *, dataset_root: Path, backup: bool) -> None:
    table = pq.read_table(path)
    metadata = dict(table.schema.metadata or {})
    hf_metadata = metadata.get(HF_METADATA_KEY)
    if hf_metadata is None:
        return

    hf_info = json.loads(hf_metadata)
    _replace_list_with_sequence(hf_info)
    metadata[HF_METADATA_KEY] = json.dumps(hf_info).encode()

    fixed = table.replace_schema_metadata(metadata)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(fixed, tmp_path)

    if backup:
        backup_path = dataset_root / ".parquet_metadata_backups" / path.relative_to(dataset_root)
        if backup_path.exists():
            raise FileExistsError(f"Backup already exists: {backup_path}")
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        path.replace(backup_path)
    else:
        path.unlink()

    tmp_path.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset_root",
        type=Path,
        help="Local LeRobot dataset root, for example /workspace/openpi/datasets/test_lerobot_v2.1",
    )
    parser.add_argument("--apply", action="store_true", help="Rewrite affected parquet files.")
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Move original parquet files to .parquet_metadata_backups/ before rewriting.",
    )
    args = parser.parse_args()

    data_dir = args.dataset_root / "data"
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Expected data directory: {data_dir}")

    parquet_files = sorted(data_dir.rglob("*.parquet"))
    affected = [path for path in parquet_files if _needs_fix(path)]

    print(f"Scanned {len(parquet_files)} parquet file(s).")
    print(f"Found {len(affected)} file(s) with incompatible Hugging Face List metadata.")
    for path in affected:
        print(path)

    if not args.apply:
        print("Dry run only. Re-run with --apply to rewrite affected files.")
        return

    for path in affected:
        _rewrite_hf_metadata(path, dataset_root=args.dataset_root, backup=args.backup)

    print("Done.")


if __name__ == "__main__":
    main()
