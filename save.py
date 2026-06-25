"""
base model: 
    gs://openpi-assets/checkpoints/pi0_base    
    gs://openpi-assets/checkpoints/pi0_fast_base
    gs://openpi-assets/checkpoints/pi05_base
"""
import os
from openpi.shared import download

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com' # 设置下载源
os.environ["OPENPI_DATA_HOME"] = "/workspace/openpi/checkpoint" # 设置保存路径
path = "gs://openpi-assets/checkpoints/pi05_base" # 下载你需要的base model（pi0/pi05）
checkpoint_dir = download.maybe_download(path) # 开始下载模型，并保存到刚才指定的位置
