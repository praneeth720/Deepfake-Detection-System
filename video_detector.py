import os
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification, CLIPProcessor, CLIPModel
import pytesseract

# ================= OCR CONFIG =================
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ================= CONFIG =================
MODEL_NAME = "dima806/deepfake_vs_real_image_detection"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_FRAMES = 12
RESIZE_WIDTH = 256
OCR_FRAME_INDEX = 0
WATERMARK_TEXT_THRESHOLD = 5

# Thresholds
STYLIZED_BOOST = 0.10
WATERMARK_BOOST = 0.10
EDGE_DENSITY_THRESHOLD = 0.12
TEXTURE_VARIANCE_THRESHOLD = 800
FFT_RATIO_THRESHOLD = 1.8

# ================= LOAD MODELS =================
print("🔄 Loading models...")
processor = AutoImageProcessor.from_pretrained(MODEL_NAME, use_fast=True)
model = AutoModelForImageClassification.from_pretrained(MODEL_NAME).to(DEVICE).eval()

anime_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
anime_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE).eval()
print("✅ Models ready")

# ================= FRAME EXTRACTION =================
def extract_frames(video_path, max_frames=MAX_FRAMES):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []

    frame_idxs = np.linspace(0, total_frames - 1, max_frames, dtype=int)
    frames = []
    for idx in frame_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        h, w = frame.shape[:2]
        scale = RESIZE_WIDTH / w
        frame = cv2.resize(frame, (RESIZE_WIDTH, int(h * scale)))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return frames

# ================= FRAME PREDICTION =================
def predict_frames(frames):
    pil_images = [Image.fromarray(f) for f in frames]
    inputs = processor(images=pil_images, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)
    return probs[:, 1].cpu().numpy()  # fake probabilities

# ================= WATERMARK DETECTION =================
def detect_watermark(frame):
    small = cv2.resize(frame, (256, 256))
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    text = pytesseract.image_to_string(gray)
    return len(text.strip()) > WATERMARK_TEXT_THRESHOLD

# ================= EDGE & TEXTURE =================
def edge_density(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return np.sum(edges > 0) / edges.size

def texture_variance(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    return np.var(gray)

def noise_level(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    return np.var(gray)

# ================= MOBILE CAMERA DETECTION =================
def is_mobile_like_video(frames):
    edge_vals = [edge_density(f) for f in frames[:10]]
    texture_vals = [texture_variance(f) for f in frames[:10]]
    return np.mean(edge_vals) > EDGE_DENSITY_THRESHOLD and np.mean(texture_vals) > TEXTURE_VARIANCE_THRESHOLD

# ================= STYLIZED DETECTION =================
def detect_stylized(frames, frame_probs):
    votes = 0
    if np.mean(frame_probs) < 0.08:
        votes += 1
    fft_scores, edge_scores, texture_scores = [], [], []
    for frame in frames[:10]:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude = np.log(np.abs(fshift) + 1)
        high = np.mean(magnitude[magnitude > np.percentile(magnitude, 75)])
        low = np.mean(magnitude[magnitude < np.percentile(magnitude, 25)])
        fft_scores.append(high / (low + 1e-6))
        edge_scores.append(edge_density(frame))
        texture_scores.append(texture_variance(frame))
    if np.mean(fft_scores) > FFT_RATIO_THRESHOLD:
        votes += 1
    if np.mean(edge_scores) > EDGE_DENSITY_THRESHOLD:
        votes += 1
    if np.mean(texture_scores) < TEXTURE_VARIANCE_THRESHOLD:
        votes += 1
    return votes >= 2

# ================= ANIME DETECTION =================
def is_anime_frame(frame):
    return edge_density(frame) > EDGE_DENSITY_THRESHOLD and texture_variance(frame) < TEXTURE_VARIANCE_THRESHOLD

def detect_anime_clip(frames):
    pil_images = [Image.fromarray(f) for f in frames]
    inputs = anime_processor(text=["anime style", "real photograph"], images=pil_images, return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        outputs = anime_model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1).cpu().numpy()
    return float(np.mean(probs[:, 0]))  # probability of anime

# ================= MAIN DETECTOR =================
def detect_video(video_path):
    if not os.path.exists(video_path):
        return 0, "Error: Video file not found", "Video file does not exist"

    frames = extract_frames(video_path)
    if not frames:
        return 0, "Error: No frames extracted", "No frames could be extracted from video"

    # ----- ANIME HEURISTIC -----
    anime_count = sum(1 for f in frames[:20] if is_anime_frame(f))
    if anime_count > len(frames[:20]) * 0.5:
        return 100, "🎌 Anime / Stylized Content (Heuristic)", f"Anime frames: {anime_count}/{len(frames[:20])}"

    # ----- ANIME MODEL -----
    anime_prob = detect_anime_clip(frames[:20])
    if anime_prob > 0.8:
        return anime_prob*100, "🎌 Anime / Stylized Content (ML)", f"Anime probability: {anime_prob*100:.2f}%"

    # ----- DEEPFAKE DETECTION -----
    frame_probs = predict_frames(frames)
    avg_fake_prob = float(np.mean(frame_probs))
    confidence = avg_fake_prob * 100

    # ----- STYLIZED CONTENT -----
    stylized = detect_stylized(frames, frame_probs)
    if stylized:
        confidence += STYLIZED_BOOST * 100

    # ----- WATERMARK -----
    if detect_watermark(frames[OCR_FRAME_INDEX]):
        confidence += WATERMARK_BOOST * 100

    # ----- NOISE CHECK -----
    avg_noise = np.mean([noise_level(f) for f in frames])
    if avg_noise > 500:
        confidence -= 10

    # ----- MOBILE CAMERA SAFETY -----
    if is_mobile_like_video(frames) and confidence < 35:
        confidence *= 0.5

    confidence = min(max(confidence, 0), 100)

    # ----- FINAL VERDICT -----
    if confidence >= 40:
        verdict = "Fake Video ❌"
    elif confidence >= 35:
        verdict = "Suspicious ⚠️"
    else:
        verdict = "Real Video ✅"

    details = (
        f"Frames analyzed: {len(frames)}\n"
        f"Base fake probability: {avg_fake_prob*100:.2f}%\n"
        f"Stylized content detected: {stylized}\n"
        f"Mobile-like video: {is_mobile_like_video(frames)}\n"
        f"Average noise level: {avg_noise:.2f}\n"
        f"Final confidence: {confidence:.2f}%"
    )

    return confidence, verdict, details