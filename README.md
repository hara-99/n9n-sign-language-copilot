# 🌐 VoxSign AI Copilot — Sign Language to Multilingual Speech Engine

<div align="center">

**Real-Time AI-Powered Sign Language Recognition, Context Understanding & Multilingual Speech**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python\&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.141%2B-009688?logo=fastapi\&logoColor=white)](https://fastapi.tiangolo.com/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-MLP%20Classifier-F7931E?logo=scikit-learn\&logoColor=white)](https://scikit-learn.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-21%203D%20Landmarks-00A651?logo=google\&logoColor=white)](https://developers.google.com/mediapipe)
[![Accuracy](https://img.shields.io/badge/Accuracy-98.85%25-success)](#-model--dataset-benchmarks)
[![License](https://img.shields.io/badge/License-MIT-purple)](#-license)
[![Deployment](https://img.shields.io/badge/Deployment-Live-brightgreen)](https://n9n-sign-language-copilot.onrender.com)

### 🚀 [LIVE DEMO](https://n9n-sign-language-copilot.onrender.com)

</div>

---

## 📖 Overview

**VoxSign AI Copilot** is a real-time multimodal AI system designed to bridge communication between Deaf/Hard-of-Hearing users and the hearing world.

Instead of simple letter-by-letter recognition, VoxSign combines computer vision, machine learning, temporal processing, contextual NLP, decision intelligence, and multilingual text-to-speech into a unified pipeline.

### What VoxSign Does

1. 👁️ Tracks hand landmarks in real time using **MediaPipe**.
2. 🧠 Recognizes **ASL alphabet signs** using an optimized MLP classifier.
3. ⏱️ Uses temporal buffering and intelligent debouncing to reduce recognition noise.
4. 💬 Converts recognized signs into meaningful conversational sentences.
5. 🔊 Produces natural speech in multiple languages.
6. 🚨 Detects emergency phrases and activates an alert state.
7. ♿ Provides signer profiles for different signing speeds and stability requirements.
8. 📊 Includes a dataset and model evaluation hub.

---

# ✨ Key Features

## 👁️ 1. Real-Time Sign Recognition

* MediaPipe-based hand landmark tracking
* 21 3D hand landmarks
* Real-time camera input
* Motion and kinematic analysis
* Palm orientation and gesture measurements
* Webcam mirror mode
* Virtual ASL simulator

---

## 🧠 2. Five-Way AI Decision Engine

VoxSign does not simply predict a letter and immediately speak it.

The decision engine evaluates the recognition result and chooses one of five states:

| State          | Purpose                                                   |
| -------------- | --------------------------------------------------------- |
| 🟢 **SPEAK**   | Confident sign → generate speech                          |
| 🔵 **WAIT**    | Gesture still changing → continue buffering               |
| 🟡 **CLARIFY** | Ambiguous prediction → show alternatives                  |
| 🟣 **CORRECT** | Context indicates a previous interpretation should change |
| 🔴 **ALERT**   | Emergency phrase detected                                 |

This helps prevent unwanted speech caused by individual noisy video frames.

---

## 🔤 3. Context-Aware NLP

The NLP engine converts isolated recognized tokens into more natural communication.

### Example

```text
Input:
W A T E R

Output:
"May I please have a glass of water?"
```

It also supports:

* Duplicate-token filtering
* Word boundary detection
* Typo correction
* Context-aware sentence reconstruction
* Conversational memory
* Emergency keyword detection

Example corrections:

```text
HLP  → HELP
WATR → WATER
THX  → THANKS
```

---

# 🗣️ 4. Multilingual Speech

VoxSign supports multilingual speech synthesis including:

* 🇺🇸 English
* 🇬🇧 English UK
* 🇪🇸 Spanish
* 🇫🇷 French
* 🇩🇪 German
* 🇮🇳 Hindi
* 🇯🇵 Japanese
* 🇨🇳 Mandarin Chinese
* 🇸🇦 Arabic
* 🇮🇳 Telugu
* 🇮🇳 Tamil
* 🇮🇹 Italian

Speech controls include:

* Speech rate
* Pitch modulation
* Language selection
* Real-time browser speech synthesis

---

# ♿ 5. Signer Intelligence Profiles

Different users can sign at different speeds and with different levels of motion stability.

VoxSign provides:

| Profile               | Purpose                                      |
| --------------------- | -------------------------------------------- |
| **Balanced**          | General everyday signing                     |
| **Fast Signer**       | Faster recognition for fluent users          |
| **Novice / Learning** | More tolerance and longer confirmation       |
| **Tremor Assist**     | Additional smoothing and stability filtering |

---

# 🚨 6. Emergency Detection

VoxSign includes an emergency detection subsystem.

Recognized emergency keywords include:

```text
HELP
PAIN
DANGER
HOSPITAL
FIRE
CHOKING
POLICE
DOCTOR
MEDICINE
ACCIDENT
BREATHE
```

When an emergency phrase is detected, the system can enter:

```text
ALERT
```

The interface provides visual and audio alert feedback.

There is also a manual:

```text
🚨 SOS ALERT
```

button.

---

# 📊 Model & Dataset Benchmarks

The recognition engine uses a **Multi-Layer Perceptron (MLP)** classifier trained for ASL alphabet recognition.

| Metric                     | Specification               |
| -------------------------- | --------------------------- |
| Model                      | Multi-Layer Perceptron      |
| Hidden Layers              | `(256, 128)`                |
| Activation                 | ReLU                        |
| Optimizer                  | Adam                        |
| Classes                    | 26 (A–Z)                    |
| Dataset                    | 10,676 images               |
| Input                      | 48 × 48 grayscale           |
| Reported Test Accuracy     | **98.85%**                  |
| Reported Inference Latency | **< 12 ms** on standard CPU |

---

# 🏛️ System Architecture

VoxSign is organized into an **8-layer AI pipeline**:

```text
┌─────────────────────────────────────────────┐
│ Layer 1 — Input                             │
│ Camera • Dataset Hub • ASL Simulator        │
└──────────────────────┬──────────────────────┘
                       ↓
┌─────────────────────────────────────────────┐
│ Layer 2 — Multimodal Perception             │
│ MediaPipe • Face/Affect • Motion Velocity   │
└──────────────────────┬──────────────────────┘
                       ↓
┌─────────────────────────────────────────────┐
│ Layer 3 — Data Processing                   │
│ Normalization • Scaling • Smoothing         │
└──────────────────────┬──────────────────────┘
                       ↓
┌─────────────────────────────────────────────┐
│ Layer 4 — Temporal Understanding            │
│ Rolling Buffer • Debounce • Word Boundary  │
└──────────────────────┬──────────────────────┘
                       ↓
┌─────────────────────────────────────────────┐
│ Layer 5 — Context & Memory                  │
│ Working Memory • NLP • Auto-Correction     │
└──────────────────────┬──────────────────────┘
                       ↓
┌─────────────────────────────────────────────┐
│ Layer 6 — Semantic Interpretation           │
│ Intent • Urgency • Emergency Scanner       │
└──────────────────────┬──────────────────────┘
                       ↓
┌─────────────────────────────────────────────┐
│ Layer 7 — Decision Engine                   │
│ 5-Way FSM • Signer Profiles                │
└──────────────────────┬──────────────────────┘
                       ↓
┌─────────────────────────────────────────────┐
│ Layer 8 — Output                            │
│ Speech • Captions • Emergency Alerts       │
└─────────────────────────────────────────────┘
```

---

# 📁 Project Structure

```text
n9n-sign-language-copilot/
│
├── app.py
├── train_model.py
├── requirements.txt
├── render.yaml
├── Procfile
│
├── engine/
│   ├── dataset_service.py
│   ├── decision_engine.py
│   └── nlp_engine.py
│
├── models/
│   ├── sign_classifier.joblib
│   └── model_metadata.json
│
└── static/
    ├── index.html
    ├── style.css
    ├── app.js
    ├── mediapipe_tracker.js
    ├── dataset_hub.js
    └── architecture_visualizer.js
```

---

# 🚀 Live Demo

## Try VoxSign Online

### 👉 https://n9n-sign-language-copilot.onrender.com

No local installation is required to view the deployed application.

> **Note:** Webcam functionality requires browser camera permission.

---

# 💻 Run Locally

If you want to run the project on your own computer:

### 1. Clone

```bash
git clone https://github.com/hara-99/n9n-sign-language-copilot.git
cd n9n-sign-language-copilot
```

### 2. Create Virtual Environment

#### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

#### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the Server

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:8000
```

### Local vs Online

```text
LOCAL
Your Computer
     ↓
python app.py
     ↓
127.0.0.1:8000


ONLINE
GitHub Repository
     ↓
Render
     ↓
https://n9n-sign-language-copilot.onrender.com
```

---

# ☁️ Deployment

The application is deployed using **Render**.

### Production Start Command

```bash
uvicorn app:app --host 0.0.0.0 --port $PORT
```

### Build Command

```bash
pip install -r requirements.txt
```

Live deployment:

**https://n9n-sign-language-copilot.onrender.com**

---

# 📡 API Reference

VoxSign exposes FastAPI endpoints for programmatic integration.

### Prediction

```http
POST /api/predict
```

Example request:

```json
{
  "image_b64": "data:image/jpeg;base64,..."
}
```

Example response:

```json
{
  "predicted_class": "A",
  "confidence": 0.9885,
  "top_candidates": [
    {
      "token": "A",
      "probability": 0.9885
    },
    {
      "token": "E",
      "probability": 0.0092
    },
    {
      "token": "S",
      "probability": 0.0015
    }
  ],
  "inference_time_ms": 11.4
}
```

### Decision Engine

```http
POST /api/decision
```

### NLP Reconstruction

```http
POST /api/nlp/reconstruct
```

Example:

```json
{
  "tokens": ["H", "E", "L", "P"]
}
```

### System Status

```http
GET /api/status
```

### Dataset Statistics

```http
GET /api/dataset/stats
```

### Dataset Sample

```http
GET /api/dataset/sample/{class_name}
```

---

# 🧪 Virtual ASL Simulator

No webcam?

Use the built-in virtual ASL keyboard to test the system.

You can:

* Select letters A–Z
* Create words
* Insert spaces
* Test sentence reconstruction
* Trigger emergency phrases
* Test the decision engine
* Test speech synthesis

Example:

```text
H → E → L → P
        ↓
     "HELP"
        ↓
      ALERT
```

---

# 🔬 Technologies Used

### Artificial Intelligence

* Machine Learning
* Multi-Layer Perceptron
* Natural Language Processing
* Context-aware inference
* Decision state machine

### Computer Vision

* MediaPipe
* Hand landmark detection
* 3D spatial features
* Motion analysis

### Backend

* Python
* FastAPI
* Uvicorn
* REST API

### Frontend

* HTML5
* CSS3
* JavaScript
* WebRTC
* Browser Speech Synthesis

### Machine Learning

* Scikit-learn
* MLP Classifier
* Joblib

### Deployment

* GitHub
* Render

---

# 🎯 Objectives

The main objectives of VoxSign AI Copilot are:

1. Enable real-time sign recognition.
2. Convert signs into meaningful language.
3. Reduce errors caused by individual video frames.
4. Generate natural conversational speech.
5. Support multiple spoken languages.
6. Provide adaptive recognition for different signing styles.
7. Detect emergency communication quickly.
8. Provide an accessible and user-friendly communication interface.

---

# 🌍 Impact

VoxSign aims to make digital communication more accessible by reducing the communication barrier between sign-language users and people who do not understand sign language.

Potential applications include:

* 🏥 Hospitals
* 🏫 Educational institutions
* 🏢 Workplaces
* 🏠 Personal communication
* 🚨 Emergency situations
* 🏛️ Public service environments
* 🌐 Accessibility-focused applications

---

# 🔮 Future Improvements

Possible future extensions include:

* Full ASL word and sentence recognition
* Continuous sign-language recognition
* More sign languages such as ISL
* Transformer-based temporal models
* Cloud model inference
* Mobile application
* Offline speech synthesis
* Personalized model fine-tuning
* Advanced facial-expression understanding
* Improved emergency communication workflows

---

# 🤝 Contributing

Contributions are welcome.

```bash
# Fork the repository

git checkout -b feature/AmazingFeature

git add .

git commit -m "Add AmazingFeature"

git push origin feature/AmazingFeature
```

Then open a Pull Request.

---

# 📄 License

This project is distributed under the **MIT License**.

---

<div align="center">

### 🌐 VoxSign AI Copilot

**AI • Accessibility • Computer Vision • NLP • Speech**

Built to make communication more inclusive.

⭐ If you find this project useful, consider giving the repository a star!

</div>
