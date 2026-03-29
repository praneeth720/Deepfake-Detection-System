import librosa
import numpy as np

class AudioProcessor:
    def __init__(self):
        self.sample_rate = 16000

    def extract_features(self, file_path):
        try:
            audio, sr = librosa.load(file_path, sr=self.sample_rate)

            # Reject very short audio
            if len(audio) < sr * 1:
                return None

            # ===== MFCC =====
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            mfcc_var = np.var(mfcc)

            # ===== Pitch =====
            pitches, _ = librosa.piptrack(y=audio, sr=sr)
            pitch_values = pitches[pitches > 0]
            pitch_var = np.var(pitch_values) if len(pitch_values) > 0 else 0

            # ===== Energy =====
            energy_frames = librosa.feature.rms(y=audio)[0]
            energy_var = np.var(energy_frames)

            # ===== ZCR =====
            zcr = librosa.feature.zero_crossing_rate(audio)[0]
            zcr_var = np.var(zcr)

            # ===== Noise Level (NEW 🔥) =====
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            noise_level = np.var(spectral_centroid)

            return {
                "mfcc_var": float(mfcc_var),
                "pitch_var": float(pitch_var),
                "energy_var": float(energy_var),
                "zcr_var": float(zcr_var),
                "noise_level": float(noise_level),
            }

        except Exception as e:
            print("Feature Extraction Error:", e)
            return None