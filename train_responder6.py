import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import tensorflow as tf
import time
from datetime import datetime
import os
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# 0. 详细的GPU检测和配置
print("[INFO] TensorFlow version:", tf.__version__)
print("[INFO] CUDA available:", tf.test.is_built_with_cuda())
print("[INFO] GPU devices:", tf.config.list_physical_devices('GPU'))

# 尝试启用GPU内存增长
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("[INFO] Memory growth enabled for GPU.")
    except RuntimeError as e:
        print("[WARN] Error setting memory growth:", e)
else:
    print("[WARN] No GPU detected! Training will be on CPU.")
    print("[INFO] Please check:")
    print("1. NVIDIA drivers are installed (nvidia-smi)")
    print("2. CUDA toolkit is installed (nvcc --version)")
    print("3. cuDNN is installed")
    print("4. tensorflow-gpu is installed (pip install tensorflow-gpu==2.12.0)")

# 1. 数据路径
DATA_PATH = "/share/home/jyren/ai_hwk1_janestreet/data/train.parquet/"
FEATURES_PATH = "/share/home/jyren/ai_hwk1_janestreet/data/features.csv"

# 2. 读取数据
print("[INFO] Loading data...")
df = pd.read_parquet(DATA_PATH)
features = pd.read_csv(FEATURES_PATH)['feature'].tolist()  # features.csv 应有 'feature' 列

# 3. 缺失值处理
print("[INFO] Handling missing values...")
missing_percentage = df[features].isnull().sum() * 100 / len(df)
# 剔除高缺失特征（如 >15%）
drop_features = missing_percentage[missing_percentage > 15].index.tolist()
features = [f for f in features if f not in drop_features]
# 低缺失特征用中位数填充
df[features] = df[features].fillna(df[features].median())

# 4. 噪声标注（滑动窗口局部std）
print("[INFO] Calculating noise labels...")
window_size = 20  # 可调整
std_threshold = 3  # 3倍全局std
noise_label = np.zeros(len(df), dtype=np.int8)
global_std = df[features].stack().std()
for col in features:
    rolling_std = df[col].rolling(window=window_size, min_periods=1, center=True).std()
    noise_label[(rolling_std > std_threshold * global_std).values] = 1
# 可选：将噪声标签加入df
df['noise_label'] = noise_label

# 5. 时序标准化（Z-score + 分位数归一化）
print("[INFO] Applying Z-score and quantile normalization...")
# Z-score标准化
df_z = (df[features] - df[features].mean()) / df[features].std()
# 分位数归一化（clip到1%~99%）
q_low = df_z.quantile(0.01)
q_high = df_z.quantile(0.99)
df_z = df_z.clip(q_low, q_high, axis=1)
df[features] = df_z

# 6. 设定标签
target_col = 'responder_6'

# 7. 只保留有标签的样本
df = df[~df[target_col].isnull()]

# 8. 特征重组（窗口切割+重塑为CNN输入）
print("[INFO] Reshaping features for CNN input...")
window_len = 60  # 每个样本的时间步长，可调整
num_features = len(features)
# 只保留能整除的部分
total = (len(df) // window_len) * window_len
X = df[features].values[:total].reshape(-1, window_len, num_features)
y = df[target_col].values[:total].reshape(-1, window_len)
noise = df['noise_label'].values[:total].reshape(-1, window_len)
# 取每个窗口最后一个时间步的标签作为该窗口的标签
y = y[:, -1]
noise = noise[:, -1]
# 增加通道维度，适配CNN输入
X = X[..., np.newaxis]  # shape: (batch, window_len, num_features, 1)

# 9. 划分训练集和测试集
X_train, X_test, y_train, y_test, noise_train, noise_test = train_test_split(
    X, y, noise, test_size=0.2, random_state=42)

# 10. 构建神经网络模型（多尺度CNN + SE注意力 + 双输出）
def squeeze_excite_block(input_tensor, ratio=16):
    """SE注意力模块"""
    channels = input_tensor.shape[-1]
    se = tf.keras.layers.GlobalAveragePooling2D()(input_tensor)
    se = Dense(channels // ratio, activation='relu')(se)
    se = Dense(channels, activation='sigmoid')(se)
    se = tf.keras.layers.Reshape((1, 1, channels))(se)
    return tf.keras.layers.Multiply()([input_tensor, se])

# 主输入
inputs = Input(shape=(window_len, num_features, 1))

# 3x3 卷积分支
x3 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
x3 = tf.keras.layers.BatchNormalization()(x3)
x3 = squeeze_excite_block(x3)  # 添加SE注意力

# 7x7 卷积分支
x7 = tf.keras.layers.Conv2D(32, (7, 7), activation='relu', padding='same')(inputs)
x7 = tf.keras.layers.BatchNormalization()(x7)
x7 = squeeze_excite_block(x7)  # 添加SE注意力

# 特征融合
x = tf.keras.layers.Concatenate()([x3, x7])
x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = squeeze_excite_block(x)  # 融合后的特征再次通过SE注意力

# 下采样
x = tf.keras.layers.MaxPooling2D((2, 2))(x)

# 共享特征提取
shared_features = tf.keras.layers.GlobalAveragePooling2D()(x)
shared_features = Dense(128, activation='relu')(shared_features)
shared_features = Dropout(0.4)(shared_features)

# 目标值预测分支
target_branch = Dense(64, activation='relu')(shared_features)
target_branch = Dropout(0.3)(target_branch)
target_output = Dense(1, activation='linear', name='target_output')(target_branch)

# 噪声预测分支
noise_branch = Dense(64, activation='relu')(shared_features)
noise_branch = Dropout(0.3)(noise_branch)
noise_output = Dense(1, activation='sigmoid', name='noise_output')(noise_branch)

# 构建模型
model = Model(inputs, [target_output, noise_output])

# 编译模型
model.compile(
    optimizer='adam',
    loss={
        'target_output': 'mse',
        'noise_output': 'binary_crossentropy'
    },
    loss_weights={
        'target_output': 1.0,
        'noise_output': 0.3
    },
    metrics={
        'target_output': ['mae', 'mse'],
        'noise_output': 'accuracy'
    }
)

model.summary()

# 11. TensorBoard
log_dir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
os.makedirs(log_dir, exist_ok=True)
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# 12. 训练
print("[INFO] Start training...")
start_time = time.time()
history = model.fit(
    X_train, {'target_output': y_train, 'noise_output': noise_train},
    validation_data=(X_test, {'target_output': y_test, 'noise_output': noise_test}),
    epochs=20,
    batch_size=64,
    callbacks=[EarlyStopping(patience=3, restore_best_weights=True), tensorboard_callback],
    verbose=1
)
end_time = time.time()
print(f"⏱️ Training completed in {(end_time - start_time):.2f} seconds")

# 13. 保存模型
model.save("responder6_cnn_model.h5")
model.save("responder6_cnn_saved_model")
print("✅ Model saved successfully") 

# ========== 预测与可视化 ========== #
import matplotlib
matplotlib.use('Agg')  # 适配无GUI环境
import matplotlib.pyplot as plt

print("[INFO] Predicting on test set...")
y_pred, noise_pred = model.predict(X_test)
y_pred = y_pred.flatten()
y_true = y_test.flatten()

# 保存预测结果
results_df = pd.DataFrame({
    'y_true': y_true,
    'y_pred': y_pred
})
results_df.to_csv("prediction_results.csv", index=False)
print("[INFO] Prediction results saved to prediction_results.csv")

# 1. 真实值 vs 预测值
plt.figure(figsize=(8, 6))
plt.scatter(y_true, y_pred, alpha=0.3)
plt.xlabel("True responder_6")
plt.ylabel("Predicted responder_6")
plt.title("True vs Predicted responder_6")
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
plt.tight_layout()
plt.savefig("scatter_true_vs_pred.png")
plt.close()

# 2. 残差分布
errors = y_pred - y_true
plt.figure(figsize=(8, 6))
plt.hist(errors, bins=50, alpha=0.7)
plt.xlabel("Prediction Error")
plt.ylabel("Count")
plt.title("Distribution of Prediction Errors")
plt.tight_layout()
plt.savefig("error_hist.png")
plt.close()

# 3. 局部时间序列对比
plt.figure(figsize=(12, 6))
plt.plot(y_true[:200], label="True")
plt.plot(y_pred[:200], label="Predicted")
plt.xlabel("Sample Index")
plt.ylabel("responder_6")
plt.title("True vs Predicted responder_6 (First 200 samples)")
plt.legend()
plt.tight_layout()
plt.savefig("series_compare.png")
plt.close()

# 4. 残差 vs 真实值
plt.figure(figsize=(8, 6))
plt.scatter(y_true, errors, alpha=0.3)
plt.xlabel("True responder_6")
plt.ylabel("Residual (y_pred - y_true)")
plt.title("Residuals vs True responder_6")
plt.tight_layout()
plt.savefig("residuals_vs_true.png")
plt.close()

# 5. 残差 vs 预测值
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, errors, alpha=0.3)
plt.xlabel("Predicted responder_6")
plt.ylabel("Residual (y_pred - y_true)")
plt.title("Residuals vs Predicted responder_6")
plt.tight_layout()
plt.savefig("residuals_vs_pred.png")
plt.close()

# 评估指标
print("MSE:", mean_squared_error(y_true, y_pred))
print("MAE:", mean_absolute_error(y_true, y_pred))
print("R2 Score:", r2_score(y_true, y_pred)) 