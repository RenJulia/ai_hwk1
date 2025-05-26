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
module purge
# 激活 Conda 环境（含 TF 2.12.0）
source activate tf2-gpu

# 首先加载Anaconda并激活Conda环境
module load anaconda/2023.09
module load cuda/11.8      
module load tensorflow-gpu/2.12   
module load cudnn/8.9.7.29-cuda11  



# 修复Python路径，确保Conda环境的优先级
export PYTHONPATH="$(conda info --base)/envs/tf2-gpu/lib/python3.9/site-packages:$PYTHONPATH"
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