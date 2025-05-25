from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, confusion_matrix

model = load_model("responder6_cnn_saved_model")

# 模型评估
test_results = model.evaluate(X_test, {'target_output': y_test, 'noise_output': noise_test}, verbose=0)
print(f"Target MAE: {test_results[3]:.4f}, Noise Accuracy: {test_results[4]:.4f}")

# 模型预测
y_pred, noise_pred = model.predict(X_test, verbose=0)
y_pred = y_pred.flatten()
noise_pred_class = (noise_pred.flatten() > 0.5).astype(int)

# 指标输出
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2:", r2_score(y_test, y_pred))
print("Noise Accuracy:", accuracy_score(noise_test, noise_pred_class))

# 混淆矩阵
sns.heatmap(confusion_matrix(noise_test, noise_pred_class), annot=True, fmt='d', cmap='Blues')
plt.title("Noise Confusion Matrix")
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.show()
