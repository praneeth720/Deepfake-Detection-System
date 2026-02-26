import cv2
import torch
import numpy as np
from PIL import Image, ExifTags
from transformers import CLIPProcessor, CLIPModel
from skimage.filters import sobel
from skimage.feature import local_binary_pattern
from skimage.util import img_as_float

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

AI_THRESHOLD = 0.55
SUSPICIOUS_THRESHOLD = 0.40
ANIME_EDGE_THRESHOLD = 0.15
LOW_NOISE_THRESHOLD = 0.005

# Load CLIP
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Utilities
def load_image(path):
    return Image.open(path).convert("RGB")

def get_exif_info(img):
    try:
        exif = img._getexif()
        if not exif:
            return {}
        return {ExifTags.TAGS.get(k, k): v for k, v in exif.items() if k in ExifTags.TAGS}
    except:
        return {}

def estimate_noise(gray):
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return lap.var() / 100000

def edge_density(gray):
    return np.mean(sobel(gray))

def texture_variance(gray):
    return local_binary_pattern(gray, P=8, R=1, method="uniform").var()

def fft_energy(gray):
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    return np.mean(np.abs(fshift))

def clip_ai_probability(image):
    prompts = [
        "a real photograph taken by a camera",
        "a photo taken by a mobile phone",
        "a DSLR camera photo",
        "an AI generated image",
        "a synthetic image",
        "a digitally created image",
        "an anime illustration",
        "a cartoon drawing"
    ]
    inputs = clip_processor(text=prompts, images=image, return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        outputs = clip_model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1).cpu().numpy()[0]
    ai_score = probs[3] + probs[4] + probs[5] + 0.6*probs[6] + 0.6*probs[7]
    anime_score = probs[6] + probs[7]
    return ai_score, anime_score

def detect_ai_image(path):
    img = load_image(path)
    exif = get_exif_info(img)
    img_np = np.array(img)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    gray_f = img_as_float(gray)

    noise = estimate_noise(gray)
    edges = edge_density(gray_f)
    texture = texture_variance(gray)
    fft_val = fft_energy(gray)
    ai_clip, anime_clip = clip_ai_probability(img)

    heuristic_ai = 0
    if noise < LOW_NOISE_THRESHOLD: heuristic_ai += 0.15
    if fft_val > 35: heuristic_ai += 0.15
    if texture < 20: heuristic_ai += 0.10
    if edges > ANIME_EDGE_THRESHOLD: heuristic_ai += 0.10

    camera_type = "Unknown"
    if "Model" in exif:
        model = str(exif["Model"]).lower()
        if any(x in model for x in ["iphone","samsung","xiaomi","oneplus"]):
            camera_type = "Mobile Camera"
            heuristic_ai -= 0.05
        elif any(x in model for x in ["canon","nikon","sony","fuji"]):
            camera_type = "DSLR Camera"
            heuristic_ai -= 0.10

    final_ai = np.clip(0.7*ai_clip + 0.3*heuristic_ai, 0, 1)

    if anime_clip > 0.55 and edges > ANIME_EDGE_THRESHOLD:
        verdict = "🎌 Anime / Stylized Image"
    elif final_ai >= AI_THRESHOLD:
        verdict = "❌ AI-Generated Image"
    elif final_ai >= SUSPICIOUS_THRESHOLD:
        verdict = "⚠️ Suspicious / Edited Image"
    else:
        verdict = "✅ Real Camera Image"

    return {
        "confidence": round(final_ai*100,2),
        "verdict": verdict,
        "details": {
            "Camera_Type": camera_type,
            "Noise_Level": round(noise,6),
            "Edge_Density": round(edges,4),
            "Texture_Variance": round(texture,2),
            "FFT_Energy": round(fft_val,2)
        }
    }