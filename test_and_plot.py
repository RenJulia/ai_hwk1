import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

# 数据路径
DATA_PATH = "/share/home/jyren/ai_hwk1_janestreet/data/train.parquet/"
FEATURES_PATH = "/share/home/jyren/ai_hwk1_janestreet/data/features.csv"
MODEL_PATH = "responder6_cnn_model.h5"

# 1. 读取数据
print("[INFO] Loading data...")
df = pd.read_parquet(DATA_PATH)
features = pd.read_csv(FEATURES_PATH)['feature'].tolist()

# 2. 缺失值处理
print("[INFO] Handling missing values...")
missing_percentage = df[features].isnull().sum() * 100 / len(df)
drop_features = missing_percentage[missing_percentage > 15].index.tolist()
features = [f for f in features if f not in drop_features]
df[features] = df[features].fillna(df[features].median())

# 3. 噪声标注（滑动窗口局部std）
window_size = 20
std_threshold = 3
noise_label = np.zeros(len(df), dtype=np.int8)
global_std = df[features].stack().std()
for col in features:
    rolling_std = df[col].rolling(window=window_size, min_periods=1, center=True).std()
    noise_label[(rolling_std > std_threshold * global_std).values] = 1
df['noise_label'] = noise_label

# 4. 时序标准化（Z-score + 分位数归一化）
df_z = (df[features] - df[features].mean()) / df[features].std()
q_low = df_z.quantile(0.01)
q_high = df_z.quantile(0.99)
df_z = df_z.clip(q_low, q_high, axis=1)
df[features] = df_z

# 5. 设定标签
print("[INFO] Preparing data for testing...")
target_col = 'responder_6'
df = df[~df[target_col].isnull()]

# 6. 特征重组（窗口切割+重塑为CNN输入）
window_len = 60
num_features = len(features)
total = (len(df) // window_len) * window_len
X = df[features].values[:total].reshape(-1, window_len, num_features)
y = df[target_col].values[:total].reshape(-1, window_len)
noise = df['noise_label'].values[:total].reshape(-1, window_len)
y = y[:, -1]
noise = noise[:, -1]
X = X[..., np.newaxis]

# 7. 划分训练集和测试集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test, noise_train, noise_test = train_test_split(
    X, y, noise, test_size=0.2, random_state=42)

# 8. 加载模型并预测
print("[INFO] Loading model and predicting...")
model = load_model(MODEL_PATH)
y_pred, noise_pred = model.predict(X_test)
y_pred = y_pred.flatten()
y_true = y_test.flatten()

# 9. 可视化
print("[INFO] Plotting results...")
plt.figure(figsize=(8, 6))
plt.scatter(y_true, y_pred, alpha=0.3)
plt.xlabel("True responder_6")
plt.ylabel("Predicted responder_6")
plt.title("True vs Predicted responder_6")
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
plt.tight_layout()
plt.savefig("scatter_true_vs_pred.png")
plt.show()

errors = y_pred - y_true
plt.figure(figsize=(8, 6))
plt.hist(errors, bins=50, alpha=0.7)
plt.xlabel("Prediction Error")
plt.ylabel("Count")
plt.title("Distribution of Prediction Errors")
plt.tight_layout()
plt.savefig("error_hist.png")
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(y_true[:200], label="True")
plt.plot(y_pred[:200], label="Predicted")
plt.xlabel("Sample Index")
plt.ylabel("responder_6")
plt.title("True vs Predicted responder_6 (First 200 samples)")
plt.legend()
plt.tight_layout()
plt.savefig("series_compare.png")
plt.show()

# 10. 评估指标
print("MSE:", mean_squared_error(y_true, y_pred))
print("MAE:", mean_absolute_error(y_true, y_pred))
print("R2 Score:", r2_score(y_true, y_pred))
