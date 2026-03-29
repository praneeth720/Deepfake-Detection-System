# Deepfake Detection System

## 📌 Overview
This project is a **Deepfake Detection System** that analyzes **images, videos, and audio** to determine whether they are real or AI-generated.  
It combines **deep learning models, signal processing, and heuristic analysis** with a web-based interface.

---

## 🚀 Features

- 🖼️ Image Deepfake Detection (CLIP + Heuristic Analysis)
- 🎥 Video Deepfake Detection (Frame + Face Consistency)
- 🎙️ Audio Deepfake Detection (Feature Extraction + Rule-Based + ML)
- 🌐 Web Interface (Flask)
- 🔐 User Login & Registration
- 📊 Logging System (SQLite)

---

## 🧠 Technologies Used

### 🔹 Programming Language
- Python 3.x

### 🔹 Backend Framework
- Flask (Web Application Framework)

### 🔹 Machine Learning / Deep Learning
- PyTorch (Model execution)
- Transformers (CLIP, Vision Transformer - ViT)
- Scikit-learn (Random Forest for audio classification)

### 🔹 Computer Vision
- OpenCV (Image and Video Processing)
- Pillow (Image handling)
- scikit-image (Feature extraction)

### 🔹 Audio Processing
- Librosa (Audio feature extraction)
- NumPy (Numerical computations)

### 🔹 Vision Models
- CLIP Model (Image understanding and classification)
- Vision Transformer (ViT)

### 🔹 Database
- SQLite (Lightweight database for logging and user data)

### 🔹 Web & UI
- HTML5
- CSS3 (Glassmorphism-based UI design)

### 🔹 Other Tools
- FFmpeg (Audio format conversion)
- Pyngrok (Public URL tunneling)
- Werkzeug (File handling and utilities)

---

## 🏗️ Project Structure
```
Deepfake_Detection_System/
│
├── app.py
├── database.py
├── audio_service.py
├── image_detector.py
├── video_detector.py
├── requirements.txt
├── README.md
│
├── ai_detector/
│   └── audio/
│       ├── audio_detector.py
│       ├── processor.py
│       ├── models/
│
├── templates/
│   ├── login.html
│   ├── register.html
│   ├── dashboard.html
│   ├── results.html
│
├── static/
│   └── style.css
│
├── uploads/
│
└── .gitignore
```

---

## ⚙️ Installation

1. Clone the repository:

git clone https://github.com/praneeth720/Deepfake-Detection-System
cd Deepfake-Detection-System

2. Create virtual environment:

python -m venv venv
venv\Scripts\activate

3. Install dependencies:
pip install -r requirements.txt

4. Install FFmpeg (for audio conversion)

- Download from: https://ffmpeg.org/download.html
- Update path in `app.py`

---

## ▶️ Run the Application
python app.py


Then open:
http://127.0.0.1:5000


---

## ⚙️ How It Works

### 🖼️ Image Deepfake Detection

The image detection module combines deep learning and traditional forensic analysis to identify AI-generated content. It primarily uses a pretrained CLIP (Contrastive Language–Image Pretraining) model to understand the semantic relationship between the input image and descriptive prompts such as “real photograph” or “AI-generated image.” This helps in estimating how likely the image belongs to either category.

In addition to semantic classification, the system performs statistical and forensic feature analysis. It evaluates properties such as noise distribution, texture variance, edge density, frequency domain energy (FFT), and brightness levels. Real camera images typically contain natural sensor noise and irregular textures, while AI-generated images often appear overly smooth or contain synthetic artifacts.

The final decision is made using a weighted combination of CLIP-based probabilities and heuristic scores derived from these features. This hybrid approach improves robustness by reducing reliance on a single model and helps in detecting both obvious and subtle manipulations.

---

### 🎥 Video Deepfake Detection

The video detection module focuses on both spatial and temporal inconsistencies present in manipulated videos. The process begins by extracting a set of representative frames from the video to reduce computational cost while preserving important visual information.

From these frames, faces are detected using classical computer vision techniques. The detected face regions are then passed through a deep learning classification model to estimate the probability of manipulation. If no faces are detected, the system falls back to analyzing full frames.

A key component of this module is temporal consistency analysis. In real videos, facial features remain stable across consecutive frames, while deepfake videos often exhibit subtle flickering, warping, or inconsistencies. The system calculates a face consistency score by measuring pixel-level differences between consecutive frames.

Additional checks include:
- Brightness and noise consistency
- Edge sharpness and realism
- Presence of watermarks or synthetic artifacts
- Detection of stylized or anime-like content using CLIP

The final confidence score is computed by combining model predictions, temporal consistency, and heuristic corrections. Based on this score, the video is classified as real, fake, or suspicious.

---

### 🎙️ Audio Deepfake Detection

The audio detection module analyzes the acoustic characteristics of speech to differentiate between real human voices and AI-generated audio. Instead of relying solely on deep learning models, it uses a hybrid approach combining signal processing and machine learning.

First, the system extracts key audio features using Librosa, including:
- MFCC (Mel-Frequency Cepstral Coefficients) variance, representing timbral characteristics
- Pitch variance, indicating natural fluctuations in human speech
- Energy variance, capturing loudness dynamics
- Zero Crossing Rate (ZCR), reflecting signal noisiness
- Spectral centroid variance, used as a proxy for noise level

Human speech typically contains natural variations in pitch, energy, and background noise, whereas AI-generated voices tend to be smoother and more uniform.

A rule-based scoring mechanism evaluates these features to detect unnatural patterns. For example, extremely low variance in pitch or energy may indicate synthetic generation. At the same time, a Random Forest model acts as a fallback classifier to provide probabilistic predictions.

The final decision is made by combining rule-based confidence and machine learning output. If the system detects low confidence or ambiguous signals, it labels the result as uncertain, encouraging better-quality input for accurate analysis.

---

### 🔄 Overall System Behavior

Across all three modalities (image, video, and audio), the system follows a unified principle:
1. Extract meaningful features from the input
2. Apply deep learning models for semantic understanding
3. Use statistical and heuristic analysis for forensic validation
4. Combine results using a weighted decision mechanism
5. Output a confidence score along with a human-readable verdict

This multi-layered approach ensures higher reliability and reduces false positives compared to single-model systems.
---

## 📊 Output

- Confidence Score
- Deepfake / Real Verdict
- Feature Analysis

---

## 🔐 Authentication

- User login system
- SQLite database
- Logs user activity

---

## ⚠️ Limitations

- Model accuracy depends on data quality
- Very noisy audio may reduce performance
- Real-time detection not implemented

---

## 🔮 Future Work

- Real-time detection
- Improved deep learning models
- Cloud deployment
- Better UI/UX
- Advanced audio deepfake detection

---

## 📜 License

This project is for educational purposes.

---

## 👨‍💻 Author

Developed as a major project on Deepfake Detection System.
