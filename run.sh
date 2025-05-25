#!/bin/bash
#SBATCH -J responder6-test
#SBATCH -o job.out.%j
#SBATCH --partition=gpu3090
#SBATCH --qos=low
#SBATCH --get-user-env
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --mail-type=all

# 切换到工作目录
cd /share/home/jyren/ai_hwk1_janestreet

# 加载必要的环境模块
module load anaconda/2023.09
module load cuda/11.8  # 显式加载 CUDA 11.8

# 设置 Conda 环境
export CONDA_ENVS_PATH=/share/home/jyren/.conda/envs
export PATH=/share/home/jyren/.conda/bin:$PATH

# 设置 CUDA 环境变量
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH

# 设置 CUDA 库路径
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/targets/x86_64-linux/lib:$LD_LIBRARY_PATH

# 设置 TensorFlow GPU 配置
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_GPU_ALLOCATOR=cuda_malloc_async
export TF_CPP_MIN_LOG_LEVEL=2  # 屏蔽无关日志
export CUDA_VISIBLE_DEVICES=0  # 明确指定 GPU 0（避免冲突）

# 激活 Conda 环境
source ~/.bashrc
conda activate tf2-gpu
# 检查 GPU 和 CUDA 状态
echo "=== nvidia-smi ==="
nvidia-smi
echo "=== nvcc --version ==="
nvcc --version



# 验证 TensorFlow GPU 支持
/share/home/jyren/.conda/envs/tf2-gpu/bin/python -c "
import tensorflow as tf
import os
print(f'TensorFlow 版本: {tf.__version__}')
print(f'CUDA 是否可用: {tf.test.is_built_with_cuda()}')
print(f'GPU 是否可用: {tf.test.is_gpu_available()}')
print(f'CUDA_HOME: {os.environ.get(\"CUDA_HOME\")}')
print(f'LD_LIBRARY_PATH: {os.environ.get(\"LD_LIBRARY_PATH\")}')
gpus = tf.config.list_physical_devices('GPU')
print(f'可用的 GPU: {gpus}')
if gpus:
    for gpu in gpus:
        print(f'GPU 设备详情: {gpu}')
else:
    print('警告: 未检测到 GPU！')
"

# 运行测试和可视化脚本
/share/home/jyren/.conda/envs/tf2-gpu/bin/python train_responder6.py