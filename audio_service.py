from ai_detector.audio.audio_detector import DeepfakeDetector
from ai_detector.audio.processor import AudioProcessor

detector = DeepfakeDetector()
processor = AudioProcessor()


# ================= RULE-BASED =================
def rule_based_detection(features):
    mfcc = features["mfcc_var"]
    pitch = features["pitch_var"]
    energy = features["energy_var"]
    zcr = features["zcr_var"]
    noise = features["noise_level"]

    score = 0

    # AI tends to be too smooth
    if mfcc < 200:
        score += 0.3

    if pitch < 600:
        score += 0.25

    if energy < 0.0008:
        score += 0.2

    if zcr < 0.0004:
        score += 0.2

    # 🔥 Noise correction (VERY IMPORTANT)
    if noise > 3000:
        score -= 0.3   # reduce false AI detection

    # Clamp score between 0–1
    score = max(0, min(score, 1))

    is_fake = score >= 0.6
    confidence = score if is_fake else (1 - score)

    return is_fake, confidence


# ================= MAIN =================
def analyze_audio(file_path):
    try:
        features = processor.extract_features(file_path)

        if features is None:
            return {
                "confidence": 0,
                "verdict": "Invalid or Too Short Audio"
            }

        # STEP 1: Rule-based
        rule_pred, rule_conf = rule_based_detection(features)

        # STEP 2: ML fallback
        model_result = detector.predict(features)

        # FINAL DECISION
        if rule_conf >= 0.65:
            is_fake = rule_pred
            confidence = rule_conf
            method = "Rule-Based (Primary)"
        else:
            is_fake = model_result["is_deepfake"]
            confidence = model_result["confidence"]
            method = "ML Fallback"

        # 🔥 Uncertain handling
        if confidence < 0.6:
            verdict = "⚠️ Uncertain (Better Audio Needed)"
        else:
            verdict = "❌ AI Generated Voice" if is_fake else "✅ Real Human Voice"

        return {
            "confidence": round(confidence * 100, 2),
            "verdict": verdict,
            "method": method,
            "analysis": features
        }

    except Exception as e:
        print("Audio Service Error:", e)
        return {
            "confidence": 0,
            "verdict": "Processing Failed"
        }