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

- Python
- Flask
- OpenCV
- PyTorch
- Transformers (CLIP, ViT)
- Librosa (Audio Processing)
- Scikit-learn
- SQLite

---

## 🏗️ Project Structure

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

## 🔍 How It Works

### Image Detection
- Uses CLIP model to classify AI vs real images
- Combines with noise, texture, and edge analysis

### Video Detection
- Extracts frames from video
- Detects faces and analyzes temporal consistency
- Uses deep learning model for classification

### Audio Detection
- Extracts features like:
  - MFCC variance
  - Pitch variance
  - Energy variance
  - Zero Crossing Rate
  - Noise level
- Uses rule-based + machine learning approach

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
