import cv2
import mediapipe as mp
import numpy as np
import os
import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import datetime

DB_DIR = 'raf-db'

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# 面部对齐函数
LEFT_EYE_IDX = [33, 133]
RIGHT_EYE_IDX = [362, 263]
def get_eye_center(landmarks, idxs, image_shape):
    h, w = image_shape[:2]
    points = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in idxs])
    return np.mean(points, axis=0)

def align_face(image, landmarks):
    left_eye = get_eye_center(landmarks, LEFT_EYE_IDX, image.shape)
    right_eye = get_eye_center(landmarks, RIGHT_EYE_IDX, image.shape)
    desired_left_eye = (0.35, 0.35)
    desired_right_eye = (0.65, 0.35)
    desired_face_width = 48
    desired_face_height = 48
    src_pts = np.array([left_eye, right_eye, [left_eye[0], left_eye[1]+10]], dtype=np.float32)
    dst_pts = np.array([
        [desired_left_eye[0]*desired_face_width, desired_left_eye[1]*desired_face_height],
        [desired_right_eye[0]*desired_face_width, desired_right_eye[1]*desired_face_height],
        [desired_left_eye[0]*desired_face_width, (desired_left_eye[1]+0.1)*desired_face_height]
    ], dtype=np.float32)
    M = cv2.getAffineTransform(src_pts, dst_pts)
    aligned_face = cv2.warpAffine(image, M, (desired_face_width, desired_face_height), flags=cv2.INTER_CUBIC)
    return aligned_face

class EmotionLandmarkExtractor:
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        
    def extract_landmarks(self, image_path):
        try:
            image = cv2.imread(image_path)
            if image is None:
                return None
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(image_rgb)
            if not results.multi_face_landmarks:
                return None
            landmarks = results.multi_face_landmarks[0].landmark
            # 直接提取全部468点
            landmark_coords = []
            for lm in landmarks:
                landmark_coords.extend([lm.x, lm.y, lm.z])
            return np.array(landmark_coords)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None
    def close(self):
        self.face_mesh.close()

def load_dataset(data_dir):
    """Load dataset and extract landmarks"""
    emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    emotion_to_idx = {emotion: idx for idx, emotion in enumerate(emotions)}
    
    X = []  # Landmarks
    y = []  # Labels
    
    extractor = EmotionLandmarkExtractor()
    
    print("Extracting facial landmarks from dataset...")
    
    for emotion in emotions:
        emotion_dir = os.path.join(data_dir, emotion)
        if not os.path.exists(emotion_dir):
            continue
            
        print(f"Processing {emotion} images...")
        image_files = [f for f in os.listdir(emotion_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for image_file in tqdm(image_files, desc=emotion):
            image_path = os.path.join(emotion_dir, image_file)
            landmarks = extractor.extract_landmarks(image_path)
            
            if landmarks is not None:
                X.append(landmarks)
                y.append(emotion_to_idx[emotion])
    
    extractor.close()
    
    return np.array(X), np.array(y), emotions

def train_mlp_model(X_train, y_train, X_val, y_val):
    """Train MLP model with hyperparameter tuning and loss curve plotting"""
    print("Training MLP model...")
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Define MLP model with verbose output for loss tracking
    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation='relu',
        solver='adam',
        alpha=0.001,
        batch_size=64,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        verbose=True  # Enable verbose output to track training
    )
    
    # Train the model
    mlp.fit(X_train_scaled, y_train)
    
    # Plot loss curves
    plot_loss_curves(mlp)
    
    return mlp, scaler

def plot_loss_curves(model):
    """Plot training and validation loss curves"""
    print("Plotting loss curves...")
    
    # Get loss history from the model
    if hasattr(model, 'loss_curve_'):
        train_loss = model.loss_curve_
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        
        # Plot training loss
        plt.subplot(2, 2, 1)
        plt.plot(train_loss, 'b-', linewidth=2, label='Training Loss')
        plt.title('Training Loss Curve')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot training loss with log scale
        plt.subplot(2, 2, 2)
        plt.semilogy(train_loss, 'b-', linewidth=2, label='Training Loss (Log Scale)')
        plt.title('Training Loss Curve (Log Scale)')
        plt.xlabel('Iteration')
        plt.ylabel('Loss (Log Scale)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot loss improvement
        plt.subplot(2, 2, 3)
        loss_improvement = [train_loss[0] - loss for loss in train_loss]
        plt.plot(loss_improvement, 'g-', linewidth=2, label='Loss Improvement')
        plt.title('Loss Improvement Over Time')
        plt.xlabel('Iteration')
        plt.ylabel('Loss Improvement')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot convergence analysis
        plt.subplot(2, 2, 4)
        if len(train_loss) > 10:
            # Calculate moving average for smoother curve
            window_size = min(10, len(train_loss) // 10)
            moving_avg = [np.mean(train_loss[max(0, i-window_size):i+1]) 
                         for i in range(len(train_loss))]
            plt.plot(moving_avg, 'r-', linewidth=2, label=f'Moving Average (window={window_size})')
            plt.plot(train_loss, 'b-', alpha=0.5, linewidth=1, label='Raw Loss')
        else:
            plt.plot(train_loss, 'r-', linewidth=2, label='Training Loss')
        
        plt.title('Loss Convergence Analysis')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('loss_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print training statistics
        print(f"\nTraining Statistics:")
        print(f"Initial Loss: {train_loss[0]:.6f}")
        print(f"Final Loss: {train_loss[-1]:.6f}")
        print(f"Total Improvement: {train_loss[0] - train_loss[-1]:.6f}")
        print(f"Improvement Percentage: {((train_loss[0] - train_loss[-1]) / train_loss[0] * 100):.2f}%")
        print(f"Total Iterations: {len(train_loss)}")
        
        # Check for convergence
        if len(train_loss) > 20:
            recent_losses = train_loss[-20:]
            loss_std = np.std(recent_losses)
            print(f"Loss Stability (std of last 20 iterations): {loss_std:.6f}")
            
            if loss_std < 1e-6:
                print("✓ Model appears to have converged (stable loss)")
            else:
                print("⚠ Model may still be training (unstable loss)")
        
    else:
        print("Warning: Loss curve not available. Model may not have verbose output enabled.")

def evaluate_model(model, scaler, X_test, y_test, emotions):
    """Evaluate the trained model"""
    print("Evaluating model...")
    
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=emotions))
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=emotions, yticklabels=emotions)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()
    
    return y_pred

def save_model(model, scaler, emotions, filename='emotion_model.pkl'):
    """Save the trained model and scaler"""
    model_data = {
        'model': model,
        'scaler': scaler,
        'emotions': emotions
    }
    
    with open(filename, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"Model saved to {filename}")

def save_test_report(report_dict, filename='test_report.txt'):
    """保存测试报告到txt文件，格式参考用户示例"""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write('============================================================\n')
        f.write('模型测试报告\n')
        f.write(f'生成时间: {report_dict["gen_time"]}\n')
        f.write(f'数据集: {report_dict["dataset_name"]}\n')
        f.write('============================================================\n\n')
        f.write(f'开始测试模型，使用数据集: {report_dict["dataset_name"]}\n\n')
        f.write('测试结果:\n')
        f.write(f'测试样本数量: {report_dict["n_samples"]}\n')
        f.write(f'正确预测数量: {report_dict["n_correct"]}\n')
        f.write(f'测试准确率: {report_dict["accuracy"]:.2f}%\n')
        f.write(f'测试耗时: {report_dict["test_time_sec"]:.2f} 秒\n')
        f.write(f'平均每个样本处理时间: {report_dict["avg_time_ms"]:.2f} 毫秒\n\n')
        f.write('详细分类报告:\n')
        f.write(report_dict["cls_report"] + '\n')
        f.write('\n混淆矩阵:\n')
        f.write(report_dict["cm_str"] + '\n')
        f.write('\n各类别准确率:\n')
        for line in report_dict["per_class_acc_lines"]:
            f.write(line + '\n')
        f.write('\n============================================================\n')
        f.write(f'测试完成时间: {report_dict["end_time"]}\n')
        f.write('============================================================\n')

def evaluate_and_report(model, scaler, X_test, y_test, emotions, dataset_name='unknown'):
    import time
    from sklearn.metrics import classification_report, confusion_matrix
    start_time = time.time()
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    end_time = time.time()
    n_samples = len(y_test)
    n_correct = int((y_pred == y_test).sum())
    accuracy = 100.0 * n_correct / n_samples if n_samples > 0 else 0.0
    test_time_sec = end_time - start_time
    avg_time_ms = (test_time_sec / n_samples * 1000) if n_samples > 0 else 0.0
    # 分类报告
    cls_report = classification_report(y_test, y_pred, target_names=emotions)
    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    # 混淆矩阵字符串
    cm_str = '实际\\预测' + ''.join([f'{name:>10s}' for name in emotions]) + '\n'
    for i, row in enumerate(cm):
        cm_str += f'{emotions[i]:>10s}' + ''.join([f'{v:10d}' for v in row]) + '\n'
    # 各类别准确率
    per_class_acc_lines = []
    for i, name in enumerate(emotions):
        support = cm[i].sum()
        correct = cm[i, i]
        acc = 100.0 * correct / support if support > 0 else 0.0
        per_class_acc_lines.append(f'{name:>10s}: {acc:6.2f}% ({correct}/{support})')
    # 生成时间
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # 汇总
    report_dict = {
        'gen_time': now,
        'dataset_name': dataset_name,
        'n_samples': n_samples,
        'n_correct': n_correct,
        'accuracy': accuracy,
        'test_time_sec': test_time_sec,
        'avg_time_ms': avg_time_ms,
        'cls_report': cls_report,
        'cm_str': cm_str,
        'per_class_acc_lines': per_class_acc_lines,
        'end_time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    save_test_report(report_dict, filename=f'test_report_{dataset_name}.txt')
    print(f"测试报告已保存到 test_report_{dataset_name}.txt")
    return y_pred

def main():
    # Load training data
    print("Loading training dataset...")
    X_train_full, y_train_full, emotions = load_dataset(DB_DIR+'/train')
    
    # Load test data
    print("Loading test dataset...")
    X_test_full, y_test_full, _ = load_dataset(DB_DIR+'/test')
    
    print(f"Training samples: {len(X_train_full)}")
    print(f"Test samples: {len(X_test_full)}")
    print(f"Number of features (landmarks): {X_train_full.shape[1]}")
    print(f"Emotion classes: {emotions}")
    
    # Split training data into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
    )
    
    # Train the model
    model, scaler = train_mlp_model(X_train, y_train, X_val, y_val)
    
    # Evaluate on validation set
    print("\nValidation Set Performance:")
    evaluate_model(model, scaler, X_val, y_val, emotions)
    
    # Evaluate on test set
    print("\nTest Set Performance:")
    evaluate_and_report(model, scaler, X_test_full, y_test_full, emotions, dataset_name= DB_DIR)
    
    # Save the model
    save_model(model, scaler, emotions)
    
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main() 