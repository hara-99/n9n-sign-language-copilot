"""
Model Training Script for AI Sign Language to Speech Copilot
Trains an optimized Multi-Layer Perceptron (MLP) Classifier on the ASL Alphabet dataset.
"""

import os
import glob
import time
import json
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

DATASET_DIR = r"c:\Users\Hara prasad\Downloads\archive (1)\Data"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "models")
IMG_SIZE = (48, 48)
SAMPLES_PER_CLASS = 150  # Balanced high-quality sampling across all 26 classes

def load_and_preprocess_dataset(dataset_dir, samples_per_class=SAMPLES_PER_CLASS):
    classes = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])
    print(f"Discovered {len(classes)} classes: {classes}")
    
    X = []
    y = []
    class_stats = {}
    
    t0 = time.time()
    for cls_idx, cls_name in enumerate(classes):
        cls_folder = os.path.join(dataset_dir, cls_name)
        img_files = glob.glob(os.path.join(cls_folder, "*.jpg"))
        total_files = len(img_files)
        class_stats[cls_name] = total_files
        
        selected_files = img_files[:samples_per_class] if samples_per_class else img_files
        
        for img_path in selected_files:
            try:
                with Image.open(img_path) as im:
                    im_gray = im.convert("L").resize(IMG_SIZE, Image.Resampling.BILINEAR)
                    arr = np.array(im_gray, dtype=np.float32) / 255.0
                    X.append(arr.flatten())
                    y.append(cls_idx)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    print(f"Loaded {len(X)} images from {len(classes)} classes in {time.time() - t0:.2f}s")
    return X, y, classes, class_stats

def train_and_save_model():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    X, y, classes, class_stats = load_and_preprocess_dataset(DATASET_DIR)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")
    
    # Train robust Neural Network classifier
    clf = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        solver="adam",
        max_iter=150,
        random_state=42,
        early_stopping=True,
        n_iter_no_change=10,
        validation_fraction=0.1
    )
    
    t_start = time.time()
    clf.fit(X_train, y_train)
    train_duration = time.time() - t_start
    print(f"Training completed in {train_duration:.2f}s")
    
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model Test Accuracy: {acc * 100:.2f}%")
    
    cm = confusion_matrix(y_test, y_pred)
    
    # Save model and artifacts
    model_path = os.path.join(OUTPUT_DIR, "sign_classifier.joblib")
    joblib.dump(clf, model_path)
    print(f"Model saved to {model_path}")
    
    metadata = {
        "classes": classes,
        "num_classes": len(classes),
        "img_size": IMG_SIZE,
        "test_accuracy": float(acc),
        "train_samples": int(X_train.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "total_dataset_images": sum(class_stats.values()),
        "class_distribution": class_stats,
        "confusion_matrix": cm.tolist(),
        "train_duration_sec": round(train_duration, 2),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    meta_path = os.path.join(OUTPUT_DIR, "model_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {meta_path}")

if __name__ == "__main__":
    train_and_save_model()
