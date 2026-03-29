from pathlib import Path


class AudioSettings:
    SAMPLE_RATE = 22050
    FRAME_LENGTH = 2048
    HOP_LENGTH = 512

    CONFIDENCE_THRESHOLD = 0.5

    BASE_DIR = Path(__file__).resolve().parent
    MODEL_DIR = BASE_DIR / "saved_models"
    MODEL_NAME = "audio_model.pkl"


audio_settings = AudioSettings()

audio_settings.MODEL_DIR.mkdir(exist_ok=True)