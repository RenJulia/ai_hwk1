import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd

def load_tensorboard_data(log_dir):
    """从TensorBoard日志加载训练数据"""
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    
    # 遍历日志目录
    for root, dirs, files in os.walk(log_dir):
        for file in files:
            if file.endswith('.v2'):
                event_file = os.path.join(root, file)
                for event in tf.compat.v1.train.summary_iterator(event_file):
                    for value in event.summary.value:
                        if value.tag == 'loss':
                            train_loss.append(value.simple_value)
                        elif value.tag == 'val_loss':
                            val_loss.append(value.simple_value)
                        elif value.tag == 'accuracy':
                            train_acc.append(value.simple_value)
                        elif value.tag == 'val_accuracy':
                            val_acc.append(value.simple_value)
    
    return train_loss, val_loss, train_acc, val_acc

def plot_training_history(history, save_dir='.'):
    """绘制训练历史"""
    # 1. 训练过程中的loss下降
    plt.figure(figsize=(12, 4))
    
    # 目标值预测的loss
    plt.subplot(1, 2, 1)
    plt.plot(history['target_output_loss'], label='Training Loss')
    plt.plot(history['val_target_output_loss'], label='Validation Loss')
    plt.title('Target Prediction Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # 噪音预测的loss
    plt.subplot(1, 2, 2)
    plt.plot(history['noise_output_loss'], label='Training Loss')
    plt.plot(history['val_noise_output_loss'], label='Validation Loss')
    plt.title('Noise Prediction Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_loss.png"))
    plt.close()
    
    # 2. 噪音预测效果
    plt.figure(figsize=(12, 4))
    
    # 噪音预测的准确率
    plt.subplot(1, 2, 1)
    plt.plot(history['noise_output_accuracy'], label='Training Accuracy')
    plt.plot(history['val_noise_output_accuracy'], label='Validation Accuracy')
    plt.title('Noise Prediction Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "noise_prediction.png"))
    plt.close()
    
    # 3. 目标值预测的MAE和MSE
    plt.figure(figsize=(12, 4))
    
    # MAE
    plt.subplot(1, 2, 1)
    plt.plot(history['target_output_mae'], label='Training MAE')
    plt.plot(history['val_target_output_mae'], label='Validation MAE')
    plt.title('Target Prediction MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    # MSE
    plt.subplot(1, 2, 2)
    plt.plot(history['target_output_mse'], label='Training MSE')
    plt.plot(history['val_target_output_mse'], label='Validation MSE')
    plt.title('Target Prediction MSE')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "target_metrics.png"))
    plt.close()

def main():
    # 1. 从TensorBoard日志生成可视化
    log_dir = "logs"
    if os.path.exists(log_dir):
        print("[INFO] Loading TensorBoard logs...")
        train_loss, val_loss, train_acc, val_acc = load_tensorboard_data(log_dir)
        
        if train_loss:  # 如果成功加载了数据
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            plt.plot(train_loss, label='Training Loss')
            plt.plot(val_loss, label='Validation Loss')
            plt.title('Training and Validation Loss')
            plt.xlabel('Step')
            plt.ylabel('Loss')
            plt.legend()
            
            plt.subplot(1, 2, 2)
            plt.plot(train_acc, label='Training Accuracy')
            plt.plot(val_acc, label='Validation Accuracy')
            plt.title('Training and Validation Accuracy')
            plt.xlabel('Step')
            plt.ylabel('Accuracy')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig("tensorboard_metrics.png")
            plt.close()
            print("✅ TensorBoard visualization saved as tensorboard_metrics.png")
    
    # 2. 从保存的模型生成可视化
    model_path = "responder6_cnn_model.h5"
    if os.path.exists(model_path):
        print("[INFO] Loading saved model...")
        try:
            model = load_model(model_path)
            # 如果模型有训练历史记录
            if hasattr(model, 'history'):
                print("[INFO] Plotting training history...")
                plot_training_history(model.history)
                print("✅ Training history visualization completed")
        except Exception as e:
            print(f"[WARN] Could not load model history: {e}")
    
    # 3. 从预测结果生成可视化
    results_path = "prediction_results.csv"
    if os.path.exists(results_path):
        print("[INFO] Loading prediction results...")
        results = pd.read_csv(results_path)
        
        # 绘制预测结果散点图
        plt.figure(figsize=(8, 6))
        plt.scatter(results['y_true'], results['y_pred'], alpha=0.3)
        plt.xlabel("True responder_6")
        plt.ylabel("Predicted responder_6")
        plt.title("True vs Predicted responder_6")
        plt.plot([results['y_true'].min(), results['y_true'].max()], 
                [results['y_true'].min(), results['y_true'].max()], 'r--')
        plt.tight_layout()
        plt.savefig("prediction_scatter.png")
        plt.close()
        
        # 绘制残差分布
        errors = results['y_pred'] - results['y_true']
        plt.figure(figsize=(8, 6))
        plt.hist(errors, bins=50, alpha=0.7)
        plt.xlabel("Prediction Error")
        plt.ylabel("Count")
        plt.title("Distribution of Prediction Errors")
        plt.tight_layout()
        plt.savefig("prediction_errors.png")
        plt.close()
        
        print("✅ Prediction visualization completed")

if __name__ == "__main__":
    main() 