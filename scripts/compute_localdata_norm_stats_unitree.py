#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
compute_norm_stats.py

此脚本递归地遍历指定的一个或多个 data_path 目录，处理所有符合条件的子目录（包含 meta 和 data 子目录）中的数据，
计算指定字段的统计量，并将结果输出为 norm_stats.json 格式。
"""

import argparse
import os
import sys
import json
import pyarrow.parquet as pq
import numpy as np
import pandas as pd
from typing import List, Dict

def parse_arguments():
    """
    解析命令行参数。

    返回:
        argparse.Namespace: 解析后的命令行参数。
    """
    parser = argparse.ArgumentParser(
        description="处理所有符合条件的子目录中的数据，计算统计量并输出结果。"
    )
    parser.add_argument(
        '--data_paths',
        type=str,
        required=True,
        help="一个或多个根目录路径，多个路径可以用逗号分隔，或者多次使用此参数"
    )
    parser.add_argument(
        '--output_path',
        type=str,
        required=True,
        help="输出统计结果的 JSON 文件路径"
    )
    parser.add_argument(
        '--norm_dim',
        type=int,
        default=30,
        help=r"指定 norm_stats 的维度，默认为30，匹配 convert_unitree_data_to_lerobot_v21.py 生成的 30 维 observation_state/actions。如果指定维度大于原始数据维度，后面的维度将填充0，否则将截断。"
    )
    args = parser.parse_args()
    
    # 处理逗号分隔的路径
    if ',' in args.data_paths:
        args.data_paths = [p.strip() for p in args.data_paths.split(',') if p.strip()]
    else:
        args.data_paths = [args.data_paths]
        
    return args

def find_valid_subdirs(root_paths: List[str]) -> List[Dict]:
    """
    在多个根目录下查找所有有效的子目录（包含 meta 和 data 子目录）。

    参数:
        root_paths (List[str]): 根目录路径列表。

    返回:
        list: 包含有效子目录信息的字典列表，每个字典包含子目录路径和对应的 meta 文件路径。
    """
    valid_dirs = []
    
    for root_path in root_paths:
        print(f"\n检查根目录: {root_path}")
        if not os.path.isdir(root_path):
            print(f"警告：目录不存在或不可访问：{root_path}")
            continue
            
        # 检查当前目录是否直接包含meta和data子目录
        meta_dir = os.path.join(root_path, "meta")
        data_dir = os.path.join(root_path, "data")
        info_file = os.path.join(meta_dir, "info.json")
        
        if os.path.isdir(meta_dir) and os.path.isdir(data_dir) and os.path.isfile(info_file):
            # 当前目录就是一个有效的任务目录
            valid_dirs.append({
                "path": root_path,
                "meta_file": info_file,
                "data_dir": data_dir,
                "name": os.path.basename(root_path),
                "root_path": os.path.dirname(root_path)
            })
            print(f"找到有效目录: {os.path.basename(root_path)}")
            continue
            
        # 如果当前目录不是任务目录，则检查其一级子目录
        try:
            subdirs = [d for d in os.listdir(root_path) 
                      if os.path.isdir(os.path.join(root_path, d)) 
                      and d not in ['meta', 'data']]  # 排除meta和data目录
        except Exception as e:
            print(f"错误：无法读取根目录 {root_path}。原因：{e}")
            continue

        for subdir in subdirs:
            subdir_path = os.path.join(root_path, subdir)
            meta_dir = os.path.join(subdir_path, "meta")
            data_dir = os.path.join(subdir_path, "data")
            info_file = os.path.join(meta_dir, "info.json")

            # 检查必要的目录和文件是否存在
            if os.path.isdir(meta_dir) and os.path.isdir(data_dir) and os.path.isfile(info_file):
                valid_dirs.append({
                    "path": subdir_path,
                    "meta_file": info_file,
                    "data_dir": data_dir,
                    "name": subdir,
                    "root_path": root_path
                })
                print(f"找到有效目录: {subdir}")
            else:
                print(f"警告：目录 {subdir_path} 不是有效的任务目录（缺少必要的子目录或文件）")

    return valid_dirs

def find_parquet_files(data_path):
    """
    递归地查找 data_path 目录下的所有 .parquet 文件。

    参数:
        data_path (str): 要搜索的目录路径。

    返回:
        list: 所有找到的 .parquet 文件的完整路径列表。
    """
    parquet_files = []
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith('.parquet'):
                full_path = os.path.join(root, file)
                parquet_files.append(full_path)
    return parquet_files

def read_parquet_file(file_path):
    """
    读取指定的 .parquet 文件中的指定字段。

    参数:
        file_path (str): .parquet 文件的路径。

    返回:
        dict: 包含字段数据的字典。
    """
    try:
        # 首先读取文件的 schema
        schema = pq.read_schema(file_path)
        # 检查文件中实际存在的字段
        available_fields = [field.name for field in schema]
        
        required_fields = ['observation_state', 'actions']
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
    LeRobot v2.1 的 list 字段可能表现为 list、np.ndarray 或 scalar。
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
    计算给定数据的统计量：均值、标准差、第1百分位数、第99百分位数，并根据指定维度进行填充或截断。

    参数:
        data_array (np.ndarray): 要计算统计量的数据数组，形状为 (样本数, 特征数)。
        norm_dim (int): 指定的统计量维度。

    返回:
        dict: 包含统计量的字典。
    """
    stats = {
        "mean": np.mean(data_array, axis=0).tolist(),
        "std": np.std(data_array, axis=0).tolist(),
        "q01": np.percentile(data_array, 1, axis=0).tolist(),
        "q99": np.percentile(data_array, 99, axis=0).tolist()
    }
    
    for key in stats:
        current_length = len(stats[key])
        if current_length < norm_dim:
            padding = [0] * (norm_dim - current_length)
            stats[key].extend(padding)
            print(f"信息：'{key}' 统计量列表长度为 {current_length}，已填充 {norm_dim - current_length} 个0以达到 {norm_dim} 维度。")
        else:
            stats[key] = stats[key][:norm_dim]  # 截断
            if current_length > norm_dim:
                print(f"信息：'{key}' 统计量列表长度为 {current_length}，已截断到 {norm_dim} 维度。")
    return stats

def main():
    args = parse_arguments()
    root_paths = args.data_paths  # 现在是路径列表
    output_path = args.output_path
    norm_dim = args.norm_dim  # 获取 norm_dim 参数

    print("将处理以下目录：")
    for path in root_paths:
        print(f"- {path}")

    # 检查所有根目录
    valid_root_paths = []
    for path in root_paths:
        print(path)
        if os.path.isdir(path):
            valid_root_paths.append(path)
        else:
            print(f"警告：目录不存在或不可访问：{path}")
            continue

    if not valid_root_paths:
        print("错误：未找到任何有效的根目录")
        sys.exit(1)

    # 查找所有有效的子目录
    valid_dirs = find_valid_subdirs(valid_root_paths)
    if not valid_dirs:
        print("错误：未找到任何有效的子目录（需要包含 meta 和 data 子目录）")
        sys.exit(1)

    print(f"\n总共找到 {len(valid_dirs)} 个有效的子目录:")
    for dir_info in valid_dirs:
        print(f"- {dir_info['name']} (在 {dir_info['root_path']} 中)")

    # 用于存储所有数据
    all_state_data = []
    all_actions_data = []

    # 处理每个有效的子目录
    for dir_info in valid_dirs:
        print(f"\n处理目录：{dir_info['name']} (来自 {dir_info['root_path']})")
        
        # 读取 meta 信息
        try:
            with open(dir_info['meta_file'], 'r') as f:
                data_info = json.load(f)
        except Exception as e:
            print(f"警告：无法读取 meta 文件 {dir_info['meta_file']}，跳过此目录。原因：{e}")
            continue

        # 查找该目录下的所有 .parquet 文件
        parquet_files = find_parquet_files(dir_info['data_dir'])
        if not parquet_files:
            print(f"警告：在目录 {dir_info['data_dir']} 中未找到任何 .parquet 文件，跳过此目录")
            continue

        print(f"在 {dir_info['name']} 中找到 {len(parquet_files)} 个 .parquet 文件")

        # 处理每个 .parquet 文件
        for idx, file in enumerate(parquet_files, 1):
            print(f"处理文件 [{idx}/{len(parquet_files)}]：{os.path.basename(file)}")
            df = read_parquet_file(file)
            if df.empty:
                continue

            # 提取 convert_unitree_data_to_lerobot_v21.py 输出的 LeRobot v2.1 字段
            for _, row in df.iterrows():
                state = to_1d_float_list(row['observation_state'], 'observation_state', file)
                act = to_1d_float_list(row['actions'], 'actions', file)
                if state is None or act is None:
                    continue

                all_state_data.append(state)
                all_actions_data.append(act)

    # 检查是否有收集到数据
    if not all_state_data or not all_actions_data:
        print("错误：未收集到任何有效数据")
        sys.exit(1)

    # Debug: 保存 all_state_data 到临时文件以便可视化分析
    import csv
    
    try:
        # 将临时文件保存到与 norm_stats.json 相同的目录
        temp_dir = os.path.dirname(output_path)
        os.makedirs(temp_dir, exist_ok=True) # 确保目录存在
        temp_state_csv_path = os.path.join(temp_dir, "all_state_data_debug.csv")
        
        with open(temp_state_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(all_state_data)
        print(f"\n[DEBUG] all_state_data 已保存到临时CSV文件：{temp_state_csv_path}")
        print("[DEBUG] 你可以使用如下 Python 代码加载并可视化：")
        print(f"import pandas as pd")
        print(f"df = pd.read_csv('{temp_state_csv_path}', header=None)")
        print(f"print(df.describe()) # 查看基本统计信息")
        print(f"import matplotlib.pyplot as plt")
        print(f"import seaborn as sns")
        print(f"for col in df.columns: # 绘制每个维度的直方图和箱线图")
        print(f"    plt.figure(figsize=(12, 5))")
        print(f"    plt.subplot(1, 2, 1)")
        print(f"    sns.histplot(df[col], kde=True)")
        print(f"    plt.title(f'Dimension {col} Histogram')")
        print(f"    plt.subplot(1, 2, 2)")
        print(f"    sns.boxplot(y=df[col])")
        print(f"    plt.title(f'Dimension {col} Boxplot')")
        print(f"    plt.tight_layout()")
        print(f"    plt.show()")

    except Exception as e:
        print(f"警告：无法保存 all_state_data 到临时文件。原因：{e}")


    # 转换为numpy数组并计算统计量
    all_state_data = np.array(all_state_data)
    all_actions_data = np.array(all_actions_data)

    # 计算统计量
    state_stats = compute_statistics(all_state_data, norm_dim)
    action_stats = compute_statistics(all_actions_data, norm_dim)

    # 准备输出数据
    output_data = {
        "norm_stats": {
            "state": state_stats,
            "actions": action_stats
        }
    }

    # 保存结果
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\n统计结果已保存到：{output_path}")
    except Exception as e:
        print(f"错误：无法保存结果到 {output_path}。原因：{e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
