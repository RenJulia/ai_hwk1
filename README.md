# Jane Street 训练任务仓库

## 项目简介
本仓库包含 Jane Street 训练任务的代码和结果。**最初实现为多任务神经网络（CNN）模型，目前已切换为 XGBoost 回归模型**，用于预测目标变量（responder_6），并对模型性能进行可视化分析。

## 数据预处理
- 数据来源：`train.parquet` 和 `features.csv`
- 特征处理：剔除缺失率超过15%的特征，对剩余特征进行中位数填充
- 噪声标注：使用滑动窗口（窗口大小为20）计算局部标准差，标记超过3倍全局标准差的样本为噪声
- 数据标准化：对特征进行Z-score标准化和分位数归一化（1%~99%）
- 数据重组：将数据按时间窗口（60个时间步）切割，并只保留能整除的部分

## 模型结构
- **当前模型：XGBoost回归模型**
  - 采用三层加权机制，结合样本权重、特征权重和噪声权重
  - 通过 RandomizedSearchCV 进行参数优化
  - 评估指标包括 MSE、MAE、R² 等

## 训练过程
- 训练集和测试集划分：按8:2的比例划分
- 训练参数：
  - 使用 XGBoostRegressor
  - 参数优化：RandomizedSearchCV 搜索最佳超参数
  - 训练过程自动保存最佳模型
- 训练结果：
  - 目标值预测：MSE、MAE、R² 等指标见可视化结果

## 可视化内容
- 目标值预测指标：MAE、MSE、R² 的变化
- 预测结果可视化：真实值 vs 预测值、残差分布、特征重要性等
- 相关图片文件：`feature_importance.png`、`prediction_scatter.png`、`residual_plot.png`

## 文件说明
- `train_responder6.py`：主训练与评估脚本（已实现XGBoost回归）
- `data_preprocess.py`：数据预处理脚本
- `run.sh`：Slurm作业提交脚本
- `xgb_responder6_model.json`：保存的XGBoost模型
- `feature_importance.png`、`prediction_scatter.png`、`residual_plot.png`：可视化结果图片
- 其他输出文件：如有

## 使用说明
1. 确保数据文件（`train.parquet`和`features.csv`）位于正确路径
2. 运行训练脚本：`python train_responder6.py` 或使用 Slurm 提交作业，脚本见 `run.sh`
3. 查看可视化结果和预测指标

## 注意事项
- 训练数据仅使用部分样本（按窗口切割后能整除的部分）
- 当前模型为 XGBoost 回归，已不再使用 CNN 方案
- 模型对目标变量的解释能力有限，可进一步优化特征工程和模型结构 