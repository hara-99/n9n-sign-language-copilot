"""
Dataset & Model Inference Service for AI Sign Language Copilot
Interfaces with the dataset in archive (1)/Data and provides fast local inference.
"""

import os
import io
import glob
import json
import base64
import random
from typing import Dict, List, Any, Optional, Tuple
from PIL import Image
import numpy as np
import joblib

DATASET_DIR = r"c:\Users\Hara prasad\Downloads\archive (1)\Data"
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
MODEL_PATH = os.path.join(MODELS_DIR, "sign_classifier.joblib")
META_PATH = os.path.join(MODELS_DIR, "model_metadata.json")
IMG_SIZE = (48, 48)

class DatasetService:
    def __init__(self):
        self.dataset_dir = DATASET_DIR
        self.model = None
        self.metadata = {}
        self.classes = []
        self._load_model_and_metadata()
        self._index_dataset()

    def _load_model_and_metadata(self):
        """Loads trained classifier and metadata if available."""
        if os.path.exists(MODEL_PATH):
            try:
                self.model = joblib.load(MODEL_PATH)
                print(f"[DatasetService] Loaded classifier model from {MODEL_PATH}")
            except Exception as e:
                print(f"[DatasetService] Failed to load model: {e}")

        if os.path.exists(META_PATH):
            try:
                with open(META_PATH, "r") as f:
                    self.metadata = json.load(f)
                    self.classes = self.metadata.get("classes", [])
            except Exception as e:
                print(f"[DatasetService] Failed to load metadata: {e}")

        if not self.classes and os.path.exists(self.dataset_dir):
            self.classes = sorted([
                d for d in os.listdir(self.dataset_dir)
                if os.path.isdir(os.path.join(self.dataset_dir, d))
            ])

    def _index_dataset(self):
        """Indexes available dataset samples for fast lookup."""
        self.class_files: Dict[str, List[str]] = {}
        if os.path.exists(self.dataset_dir):
            for cls in self.classes:
                folder = os.path.join(self.dataset_dir, cls)
                if os.path.isdir(folder):
                    files = glob.glob(os.path.join(folder, "*.jpg"))
                    self.class_files[cls] = files

    def get_dataset_stats(self) -> Dict[str, Any]:
        """Returns comprehensive dataset statistics and model metrics."""
        total_images = sum(len(f) for f in self.class_files.values())
        return {
            "classes": self.classes,
            "total_classes": len(self.classes),
            "total_dataset_images": total_images,
            "class_distribution": {k: len(v) for k, v in self.class_files.items()},
            "model_metadata": self.metadata,
            "is_model_loaded": (self.model is not None)
        }

    def get_sample_image(self, class_name: str, index: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Retrieves a sample image for a given class as Base64 data."""
        files = self.class_files.get(class_name.upper(), [])
        if not files:
            return None
        
        if index is not None and 0 <= index < len(files):
            chosen_file = files[index]
        else:
            chosen_file = random.choice(files)

        try:
            with Image.open(chosen_file) as im:
                buffered = io.BytesIO()
                im.save(buffered, format="JPEG")
                img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                
                # Perform model prediction on sample
                pred_result = self.predict_pil_image(im)

                return {
                    "class_name": class_name.upper(),
                    "filename": os.path.basename(chosen_file),
                    "image_b64": f"data:image/jpeg;base64,{img_b64}",
                    "prediction": pred_result
                }
        except Exception as e:
            print(f"[DatasetService] Error reading sample {chosen_file}: {e}")
            return None

    def predict_pil_image(self, pil_image: Image.Image) -> Dict[str, Any]:
        """Runs inference on a PIL Image."""
        if self.model is None:
            return {"error": "Model not loaded"}

        try:
            # Resize and convert to grayscale
            im_gray = pil_image.convert("L").resize(IMG_SIZE, Image.Resampling.BILINEAR)
            arr = np.array(im_gray, dtype=np.float32) / 255.0
            x_vec = arr.flatten().reshape(1, -1)

            # Predict probabilities
            probs = self.model.predict_proba(x_vec)[0]
            top_indices = np.argsort(probs)[::-1][:5]
            
            top_candidates = []
            for idx in top_indices:
                cls_name = self.classes[idx] if idx < len(self.classes) else f"Class_{idx}"
                top_candidates.append({
                    "token": cls_name,
                    "confidence": float(probs[idx]),
                    "confidence_pct": round(float(probs[idx]) * 100, 1)
                })

            best_match = top_candidates[0]
            return {
                "predicted_token": best_match["token"],
                "confidence": best_match["confidence"],
                "confidence_pct": best_match["confidence_pct"],
                "top_candidates": top_candidates,
                "entropy": float(-np.sum(probs * np.log(probs + 1e-12)))
            }
        except Exception as e:
            return {"error": str(e)}

    def predict_base64_frame(self, b64_str: str) -> Dict[str, Any]:
        """Decodes base64 image data and runs inference."""
        try:
            if "," in b64_str:
                b64_str = b64_str.split(",", 1)[1]
            img_data = base64.b64decode(b64_str)
            with Image.open(io.BytesIO(img_data)) as pil_img:
                return self.predict_pil_image(pil_img)
        except Exception as e:
            return {"error": f"Failed to parse base64 image: {e}"}
