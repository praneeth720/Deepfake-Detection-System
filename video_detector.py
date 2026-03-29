import os
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    CLIPProcessor,
    CLIPModel
)
import pytesseract

# ================= OCR =================
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ================= CONFIG =================
MODEL_NAME = "dima806/deepfake_vs_real_image_detection"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_FRAMES = 12
RESIZE_WIDTH = 256

AI_THRESHOLD = 50
SUSPICIOUS_THRESHOLD = 30

# ================= LOAD MODELS =================
print("🔄 Loading models...")

processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForImageClassification.from_pretrained(MODEL_NAME).to(DEVICE).eval()

clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE).eval()

print("✅ Models ready")

# ================= FACE DETECTOR =================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ================= FRAME EXTRACTION =================
def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        return []

    step = max(total_frames // MAX_FRAMES, 1)
    frames = []

    for i in range(0, total_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()

        if not ret:
            continue

        h, w = frame.shape[:2]
        scale = RESIZE_WIDTH / w
        frame = cv2.resize(frame, (RESIZE_WIDTH, int(h * scale)))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frames.append(frame)

        if len(frames) >= MAX_FRAMES:
            break

    cap.release()
    return frames


# ================= FACE EXTRACTION =================
def extract_faces_with_positions(frames):
    face_data = []

    for idx, frame in enumerate(frames):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        faces = face_cascade.detectMultiScale(
            gray, 1.2, 5, minSize=(40, 40)
        )

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            face = cv2.resize(face, (224, 224))
            face_data.append((idx, face, (x, y, w, h)))

    return face_data


# ================= FACE CONSISTENCY =================
def face_consistency_score(face_data):
    if len(face_data) < 2:
        return 1.0  # assume stable

    diffs = []

    for i in range(len(face_data) - 1):
        _, f1, _ = face_data[i]
        _, f2, _ = face_data[i + 1]

        f1_gray = cv2.cvtColor(f1, cv2.COLOR_RGB2GRAY)
        f2_gray = cv2.cvtColor(f2, cv2.COLOR_RGB2GRAY)

        diff = np.mean(np.abs(f1_gray.astype("float") - f2_gray.astype("float")))
        diffs.append(diff)

    avg_diff = np.mean(diffs)

    # normalize score
    consistency = max(0, 1 - (avg_diff / 50))

    return consistency  # 0 = inconsistent, 1 = stable


# ================= MODEL PREDICTION =================
def predict_frames(frames):
    images = [Image.fromarray(f) for f in frames]

    inputs = processor(images=images, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)
    return probs[:, 1].cpu().numpy()


# ================= CLIP =================
def detect_anime(frames):
    images = [Image.fromarray(f) for f in frames[:5]]

    texts = [
        "anime style image",
        "cartoon image",
        "AI generated face",
        "real human photo"
    ]

    inputs = clip_processor(
        text=texts,
        images=images,
        return_tensors="pt",
        padding=True
    ).to(DEVICE)

    with torch.no_grad():
        outputs = clip_model(**inputs)

    probs = outputs.logits_per_image.softmax(dim=1).cpu().numpy()

    scores = [(p[0] + p[1] + p[2]) - p[3] for p in probs]
    return float(np.mean(scores))


# ================= WATERMARK =================
def detect_watermark(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    text = pytesseract.image_to_string(gray)
    return len(text.strip().split()) >= 3


# ================= REALISM =================
def realism_score(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    noise = np.var(gray)
    edges = cv2.Laplacian(gray, cv2.CV_64F).var()
    return noise / 255.0, edges / 255.0


# ================= BRIGHTNESS =================
def brightness_level(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    return np.mean(gray) / 255.0


# ================= MAIN =================
def detect_video(video_path):

    if not os.path.exists(video_path):
        return 0, "Error", "File not found"

    frames = extract_frames(video_path)
    if len(frames) == 0:
        return 0, "Error", "No frames"

    # FACE DATA
    face_data = extract_faces_with_positions(frames)
    faces = [f[1] for f in face_data]

    # PREDICTION
    if len(faces) > 0:
        probs = predict_frames(faces)
    else:
        probs = predict_frames(frames)

    avg_fake_prob = float(np.mean(probs))

    # STYLE
    anime_prob = detect_anime(frames)
    is_stylized = anime_prob > 0.6

    # REALISM
    noise_vals, edge_vals, bright_vals = [], [], []

    for f in frames:
        n, e = realism_score(f)
        b = brightness_level(f)
        noise_vals.append(n)
        edge_vals.append(e)
        bright_vals.append(b)

    avg_noise = np.mean(noise_vals)
    avg_edges = np.mean(edge_vals)
    avg_brightness = np.mean(bright_vals)

    # FACE CONSISTENCY
    consistency = face_consistency_score(face_data)

    # ================= CONFIDENCE =================
    confidence = avg_fake_prob * 100

    # MODEL PRIORITY
    if avg_fake_prob < 0.35:
        confidence -= 15
    if avg_fake_prob > 0.75:
        confidence += 15

    # STYLE (controlled)
    if is_stylized and avg_fake_prob > 0.5:
        confidence += 6

    # WATERMARK
    if detect_watermark(frames[0]):
        confidence += 6

    # REALISM (only bright scenes)
    if avg_brightness > 0.3:
        if avg_noise < 0.02 and avg_edges < 0.02:
            confidence += 6

    # 🔥 FACE INCONSISTENCY BOOST
    if consistency < 0.6:
        confidence += 12

    confidence = np.clip(confidence, 0, 100)

    # ================= VERDICT =================
    if avg_fake_prob > 0.8:
        verdict = "Fake Video ❌"

    elif avg_fake_prob < 0.3 and avg_brightness > 0.2:
        verdict = "Real Video ✅"

    elif confidence >= AI_THRESHOLD:
        verdict = "Fake Video ❌"

    elif confidence >= SUSPICIOUS_THRESHOLD:
        verdict = "Suspicious ⚠️"

    else:
        verdict = "Real Video ✅"

    # ================= DETAILS =================
    details = (
        f"Frames: {len(frames)}\n"
        f"Faces: {len(faces)}\n"
        f"Fake Prob: {avg_fake_prob*100:.2f}%\n"
        f"Consistency: {consistency:.2f}\n"
        f"Brightness: {avg_brightness:.2f}\n"
        f"Noise: {avg_noise:.4f}, Edges: {avg_edges:.4f}\n"
        f"Confidence: {confidence:.2f}%"
    )

    return confidence, verdict, details