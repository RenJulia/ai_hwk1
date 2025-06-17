import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import time
from datetime import datetime
import os
import matplotlib.pyplot as plt
import seaborn as sns

# 0. 设置随机种子
np.random.seed(42)

# 1. 数据路径
DATA_PATH = "/share/home/jyren/ai_hwk1_janestreet/data/train.parquet/"
FEATURES_PATH = "/share/home/jyren/ai_hwk1_janestreet/data/features.csv"

def reduce_mem_usage(df):
    """减少内存使用"""
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    return df

def load_and_preprocess_data(data_path, features_path, target_col='responder_6'):
    """加载和预处理数据"""
    print("[INFO] Loading data...")
    # 读取数据
    df = pd.read_parquet(data_path)
    features = pd.read_csv(features_path)['feature'].tolist()
    
    # 选择特征列
    selected_columns = ['date_id', 'symbol_id', 'time_id', 'weight'] + features + [target_col]
    df = df[selected_columns]
    
    # 减少内存使用
    df = reduce_mem_usage(df)
    
    print("[INFO] Handling missing values...")
    # 处理缺失值
    for col in features:
        if df[col].isna().any():
            # 先按symbol_id进行前向填充
            df[col] = df.groupby("symbol_id")[col].ffill()
            # 对剩余的缺失值使用中位数填充
            df[col] = df.groupby("symbol_id")[col].transform(lambda x: x.fillna(x.median()))
    
    print("[INFO] Applying normalization...")
    # 特征标准化
    df[features] = df[features].astype("float32")
    mean = df[features].mean()
    std = df[features].std() + 1e-5
    df[features] = (df[features] - mean) / std
    
    # 添加噪声标签
    df["is_noise"] = (df[features].abs() > 4.5).any(axis=1).astype(int)
    
    # 按时间划分训练集和验证集
    dates = sorted(df["date_id"].unique())
    train_dates = dates[:-5]  # 使用最后5天作为验证集
    valid_dates = dates[-5:]
    
    train_df = df[df["date_id"].isin(train_dates)]
    valid_df = df[df["date_id"].isin(valid_dates)]
    
    # 准备训练数据
    X_train = train_df[features]
    y_train = train_df[target_col]
    X_valid = valid_df[features]
    y_valid = valid_df[target_col]
    
    return X_train, X_valid, y_train, y_valid, features

def asymmetric_weight(y_true):
    """动态加权机制"""
    weights = np.ones_like(y_true)
    # 普通下跌样本
    downside_mask = y_true < 0
    weights[downside_mask] = 2.5
    # 极端下跌样本（底部10%分位数）
    extreme_down = y_true < np.percentile(y_true[y_true<0], 10)
    weights[extreme_down] = 4.0
    return weights

# 3. 评估指标函数
def weighted_rmse(y_true, y_pred, weights):
    """计算加权RMSE"""
    return np.sqrt(np.mean(weights * (y_true - y_pred) ** 2))

def directional_errors(y_true, y_pred):
    """计算上行和下行误差"""
    up_mask = y_true > 0
    down_mask = y_true < 0
    up_error = np.sqrt(np.mean((y_true[up_mask] - y_pred[up_mask]) ** 2))
    down_error = np.sqrt(np.mean((y_true[down_mask] - y_pred[down_mask]) ** 2))
    return up_error, down_error

# 2. 加载和预处理数据
X_train, X_valid, y_train, y_valid, features = load_and_preprocess_data(DATA_PATH, FEATURES_PATH)

# 3. 实现三级加权机制
train_weights = asymmetric_weight(y_train.values)

# 4. 参数搜索空间
param_dist = {
    'n_estimators': [300, 400, 500],
    'learning_rate': [0.01, 0.03, 0.05],
    'max_depth': [4, 5, 6],
    'subsample': [0.6, 0.7, 0.8],
    'colsample_bytree': [0.6, 0.7, 0.8],
    'gamma': [0, 0.1, 0.2],
    'reg_alpha': [0.1, 0.5, 1.0],
    'reg_lambda': [0.1, 0.5, 1.0],
    'min_child_weight': [1, 3, 5]
}

# 5. 初始化模型
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    random_state=42,
    tree_method='gpu_hist',
    predictor='gpu_predictor',
    gpu_id=0,
    early_stopping_rounds=20,
    eval_metric=['rmse', 'mae']
)

# 6. 随机搜索
print("[INFO] Starting RandomizedSearchCV...")
random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_dist,
    n_iter=20,
    scoring='neg_root_mean_squared_error',
    cv=3,
    verbose=2,
    n_jobs=4
)

# 7. 训练模型
start_time = time.time()
random_search.fit(
    X_train, y_train,
    sample_weight=train_weights,
    eval_set=[(X_valid, y_valid)],
    verbose=True
)
end_time = time.time()
print(f"⏱️ Training completed in {(end_time - start_time):.2f} seconds")

# 8. 获取最佳模型
best_model = random_search.best_estimator_
print("\nBest parameters:", random_search.best_params_)
print("Best RMSE:", -random_search.best_score_)

# 9. 模型评估
y_pred = best_model.predict(X_valid)

# 计算各项评估指标
rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
valid_weights = asymmetric_weight(y_valid.values)
weighted_rmse_val = weighted_rmse(y_valid, y_pred, valid_weights)
up_error, down_error = directional_errors(y_valid, y_pred)
mae = mean_absolute_error(y_valid, y_pred)
r2 = r2_score(y_valid, y_pred)

print("\nModel Performance Metrics:")
print(f"Standard RMSE: {rmse:.4f}")
print(f"Weighted RMSE: {weighted_rmse_val:.4f}")
print(f"Upside Error: {up_error:.4f}")
print(f"Downside Error: {down_error:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R2 Score: {r2:.4f}")

# 10. 特征重要性分析
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': best_model.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(x='importance', y='feature', data=feature_importance.head(20))
plt.title('Top 20 Feature Importance')
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.close()

# 11. 预测结果可视化
plt.figure(figsize=(10, 6))
plt.scatter(y_valid, y_pred, alpha=0.3)
plt.plot([y_valid.min(), y_valid.max()], [y_valid.min(), y_valid.max()], 'r--')
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('True vs Predicted Values')
plt.tight_layout()
plt.savefig("prediction_scatter.png")
plt.close()

# 12. 残差分析
residuals = y_valid - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.3)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.tight_layout()
plt.savefig("residual_plot.png")
plt.close()

# 13. 保存模型
best_model.save_model("xgb_responder6_model.json")
print("✅ Model saved successfully")

# 14. 保存预测结果
results_df = pd.DataFrame({
    'y_true': y_valid,
    'y_pred': y_pred,
    'residuals': residuals
})
results_df.to_csv("prediction_results.csv", index=False)
print("✅ Prediction results saved to prediction_results.csv") 