#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
compute_localdataset_norm_stats.py

此脚本仿照 compute_localdata_norm_stats_unitree.py 的运行方式：
递归遍历指定的一个或多个 data_path 目录，处理所有包含 meta 和 data 子目录的 LeRobot v2.1 数据集，
读取 observation.state 和 action 字段，计算统计量，并输出为 OpenPI 可读取的 norm_stats.json 格式。
"""

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


STATE_FIELD = "observation.state"
ACTION_FIELD = "action"


def parse_arguments():
    """
    解析命令行参数。

    返回:
        argparse.Namespace: 解析后的命令行参数。
    """
    parser = argparse.ArgumentParser(
        description="处理所有符合条件的本地 LeRobot v2.1 数据目录，计算 norm_stats.json。"
    )
    parser.add_argument(
        "--data_paths",
        type=str,
        required=True,
        help="一个或多个根目录路径，多个路径可以用逗号分隔。",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="输出统计结果的 JSON 文件路径，例如 /workspace/openpi/assets/pi0_tron/test_lerobot_v2.1/norm_stats.json。",
    )
    parser.add_argument(
        "--norm_dim",
        type=int,
        default=32,
        help="指定 norm_stats 的维度。若原始数据维度小于该值，后面的维度填充 0；若大于该值，则截断。",
    )
    args = parser.parse_args()

    if "," in args.data_paths:
        args.data_paths = [p.strip() for p in args.data_paths.split(",") if p.strip()]
    else:
        args.data_paths = [args.data_paths]

    return args


def find_valid_subdirs(root_paths: List[str]) -> List[Dict]:
    """
    在多个根目录下查找所有有效的数据集目录（包含 meta 和 data 子目录）。
    """
    valid_dirs = []

    for root_path in root_paths:
        print(f"\n检查根目录: {root_path}")
        if not os.path.isdir(root_path):
            print(f"警告：目录不存在或不可访问：{root_path}")
            continue

        meta_dir = os.path.join(root_path, "meta")
        data_dir = os.path.join(root_path, "data")
        info_file = os.path.join(meta_dir, "info.json")

        if os.path.isdir(meta_dir) and os.path.isdir(data_dir) and os.path.isfile(info_file):
            valid_dirs.append(
                {
                    "path": root_path,
                    "meta_file": info_file,
                    "data_dir": data_dir,
                    "name": os.path.basename(root_path.rstrip(os.sep)),
                    "root_path": os.path.dirname(root_path.rstrip(os.sep)),
                }
            )
            print(f"找到有效目录: {os.path.basename(root_path.rstrip(os.sep))}")
            continue

        try:
            subdirs = [
                d
                for d in os.listdir(root_path)
                if os.path.isdir(os.path.join(root_path, d)) and d not in ["meta", "data"]
            ]
        except Exception as e:
            print(f"错误：无法读取根目录 {root_path}。原因：{e}")
            continue

        for subdir in subdirs:
            subdir_path = os.path.join(root_path, subdir)
            meta_dir = os.path.join(subdir_path, "meta")
            data_dir = os.path.join(subdir_path, "data")
            info_file = os.path.join(meta_dir, "info.json")

            if os.path.isdir(meta_dir) and os.path.isdir(data_dir) and os.path.isfile(info_file):
                valid_dirs.append(
                    {
                        "path": subdir_path,
                        "meta_file": info_file,
                        "data_dir": data_dir,
                        "name": subdir,
                        "root_path": root_path,
                    }
                )
                print(f"找到有效目录: {subdir}")
            else:
                print(f"警告：目录 {subdir_path} 不是有效的数据集目录（缺少 meta/info.json 或 data 子目录）")

    return valid_dirs


def find_parquet_files(data_path):
    """
    递归查找 data_path 下的所有 .parquet 文件。
    """
    parquet_files = []
    for root, _, files in os.walk(data_path):
        for file in files:
            if file.endswith(".parquet"):
                parquet_files.append(os.path.join(root, file))
    return sorted(parquet_files)


def read_parquet_file(file_path):
    """
    读取指定 parquet 文件中的 observation.state 和 action 字段。
    """
    try:
        schema = pq.read_schema(file_path)
        available_fields = [field.name for field in schema]

        required_fields = [STATE_FIELD, ACTION_FIELD]
        missing_fields = [field for field in required_fields if field not in available_fields]
        if missing_fields:
            print(f"警告：文件 {file_path} 缺少字段 {missing_fields}，跳过该文件")
            return pd.DataFrame()

        table = pq.read_table(file_path, columns=required_fields)
        return table.to_pandas()
    except Exception as e:
        print(f"错误：无法读取 {file_path} 文件。原因：{e}")
        return pd.DataFrame()


def to_1d_float_list(value, field_name, file_path):
    """
    将 parquet/pandas 读取出来的向量字段统一转换成一维 float list。
    """
    try:
        array = np.asarray(value, dtype=np.float32).reshape(-1)
    except Exception as e:
        print(f"警告：文件 {file_path} 字段 {field_name} 无法转换为 float 向量，跳过该行。原因：{e}")
        return None

    if array.size == 0:
        print(f"警告：文件 {file_path} 字段 {field_name} 为空，跳过该行")
        return None

    return array.tolist()


def compute_statistics(data_array, norm_dim):
    """
    计算 mean、std、q01、q99，并根据 norm_dim 进行填充或截断。
    """
    stats = {
        "mean": np.mean(data_array, axis=0).tolist(),
        "std": np.std(data_array, axis=0).tolist(),
        "q01": np.percentile(data_array, 1, axis=0).tolist(),
        "q99": np.percentile(data_array, 99, axis=0).tolist(),
    }

    for key in stats:
        current_length = len(stats[key])
        if current_length < norm_dim:
            padding = [0] * (norm_dim - current_length)
            stats[key].extend(padding)
            print(
                f"信息：'{key}' 统计量列表长度为 {current_length}，"
                f"已填充 {norm_dim - current_length} 个0以达到 {norm_dim} 维度。"
            )
        else:
            stats[key] = stats[key][:norm_dim]
            if current_length > norm_dim:
                print(f"信息：'{key}' 统计量列表长度为 {current_length}，已截断到 {norm_dim} 维度。")

    return stats


def main():
    args = parse_arguments()
    root_paths = args.data_paths
    output_path = args.output_path
    norm_dim = args.norm_dim

    print("将处理以下目录：")
    for path in root_paths:
        print(f"- {path}")

    valid_root_paths = []
    for path in root_paths:
        print(path)
        if os.path.isdir(path):
            valid_root_paths.append(path)
        else:
            print(f"警告：目录不存在或不可访问：{path}")

    if not valid_root_paths:
        print("错误：未找到任何有效的根目录")
        sys.exit(1)

    valid_dirs = find_valid_subdirs(valid_root_paths)
    if not valid_dirs:
        print("错误：未找到任何有效的数据集目录（需要包含 meta 和 data 子目录）")
        sys.exit(1)

    print(f"\n总共找到 {len(valid_dirs)} 个有效的数据集目录:")
    for dir_info in valid_dirs:
        print(f"- {dir_info['name']} (在 {dir_info['root_path']} 中)")

    all_state_data = []
    all_actions_data = []

    for dir_info in valid_dirs:
        print(f"\n处理目录：{dir_info['name']} (来自 {dir_info['root_path']})")

        try:
            with open(dir_info["meta_file"], "r", encoding="utf-8") as f:
                data_info = json.load(f)
            features = data_info.get("features", {})
            print(f"数据集版本: {data_info.get('codebase_version', 'unknown')}")
            print(f"机器人类型: {data_info.get('robot_type', 'unknown')}")
            if STATE_FIELD not in features or ACTION_FIELD not in features:
                print(
                    f"警告：meta/info.json 未声明 {STATE_FIELD} 或 {ACTION_FIELD}，"
                    "将继续以 parquet 实际字段为准。"
                )
        except Exception as e:
            print(f"警告：无法读取 meta 文件 {dir_info['meta_file']}，跳过此目录。原因：{e}")
            continue

        parquet_files = find_parquet_files(dir_info["data_dir"])
        if not parquet_files:
            print(f"警告：在目录 {dir_info['data_dir']} 中未找到任何 .parquet 文件，跳过此目录")
            continue

        print(f"在 {dir_info['name']} 中找到 {len(parquet_files)} 个 .parquet 文件")

        for idx, file in enumerate(parquet_files, 1):
            print(f"处理文件 [{idx}/{len(parquet_files)}]：{os.path.basename(file)}")
            df = read_parquet_file(file)
            if df.empty:
                continue

            for _, row in df.iterrows():
                state = to_1d_float_list(row[STATE_FIELD], STATE_FIELD, file)
                act = to_1d_float_list(row[ACTION_FIELD], ACTION_FIELD, file)
                if state is None or act is None:
                    continue

                all_state_data.append(state)
                all_actions_data.append(act)

    if not all_state_data or not all_actions_data:
        print("错误：未收集到任何有效数据")
        sys.exit(1)

    all_state_data = np.array(all_state_data, dtype=np.float32)
    all_actions_data = np.array(all_actions_data, dtype=np.float32)

    print(f"\n总帧数: {all_state_data.shape[0]}")
    print(f"state 原始维度: {all_state_data.shape[1]}")
    print(f"actions 原始维度: {all_actions_data.shape[1]}")
    print(f"输出 norm_dim: {norm_dim}")

    state_stats = compute_statistics(all_state_data, norm_dim)
    action_stats = compute_statistics(all_actions_data, norm_dim)

    output_data = {
        "norm_stats": {
            "state": state_stats,
            "actions": action_stats,
        }
    }

    try:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)
        print(f"\n统计结果已保存到：{output_path}")
    except Exception as e:
        print(f"错误：无法保存结果到 {output_path}。原因：{e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
