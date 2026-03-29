import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path

MODEL_DIR = Path("ai_detector/audio/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "audio_model.pkl"
CONFIDENCE_THRESHOLD = 0.55

FEATURE_ORDER = [
    "mfcc_var",
    "pitch_var",
    "energy_var",
    "zcr_var",
    "noise_level"
]


class DeepfakeDetector:

    def __init__(self):
        self.model = None
        self.load_model()

    def load_model(self):
        if MODEL_PATH.exists():
            self.model = joblib.load(MODEL_PATH)
        else:
            self._create_model()

    def _create_model(self):
        # Dummy model (placeholder)
        self.model = RandomForestClassifier(n_estimators=200, random_state=42)

        X = np.random.randn(1000, 50)
        y = np.random.randint(0, 2, 1000)

        self.model.fit(X, y)
        joblib.dump(self.model, MODEL_PATH)

    def prepare_features(self, features):
        values = [features.get(f, 0) for f in FEATURE_ORDER]

        # pad to 50
        values += [0] * (50 - len(values))
        return np.array(values[:50]).reshape(1, -1)

    def predict(self, features):
        try:
            input_data = self.prepare_features(features)
            probs = self.model.predict_proba(input_data)[0]

            real = float(probs[0])
            fake = float(probs[1])

            return {
                "is_deepfake": fake > CONFIDENCE_THRESHOLD,
                "confidence": max(real, fake),
                "deepfake_probability": fake,
                "authentic_probability": real,
            }

        except Exception as e:
            print("Prediction Error:", e)
            return {
                "is_deepfake": False,
                "confidence": 0.0,
                "deepfake_probability": 0.0,
                "authentic_probability": 0.0,
            }