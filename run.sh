#!/bin/bash
#SBATCH -J responder6-xgb
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

# 清理环境
module purge

# 加载必要的模块
module load anaconda/2023.09
module load cuda/11.8


# 激活环境
source activate xgb-env

# 检查 GPU 状态
echo "=== nvidia-smi ==="
nvidia-smi


# 运行训练脚本
/share/home/jyren/.conda/envs/xgb-env/bin/python train_responder6.py