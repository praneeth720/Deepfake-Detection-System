from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory
from werkzeug.utils import secure_filename
from pyngrok import ngrok
import os
import subprocess

from database import init_db, save_log, register_user, login_user
from image_detector import detect_ai_image
from video_detector import detect_video
from audio_service import analyze_audio

app = Flask(__name__)
app.secret_key = "secretkey"

# ================= FOLDERS =================
UPLOAD_FOLDER = "uploads"
IMAGE_FOLDER = os.path.join(UPLOAD_FOLDER, "images")
VIDEO_FOLDER = os.path.join(UPLOAD_FOLDER, "videos")
AUDIO_FOLDER = os.path.join(UPLOAD_FOLDER, "audios")

app.config["IMAGE_UPLOAD_FOLDER"] = IMAGE_FOLDER
app.config["VIDEO_UPLOAD_FOLDER"] = VIDEO_FOLDER
app.config["AUDIO_UPLOAD_FOLDER"] = AUDIO_FOLDER

os.makedirs(IMAGE_FOLDER, exist_ok=True)
os.makedirs(VIDEO_FOLDER, exist_ok=True)
os.makedirs(AUDIO_FOLDER, exist_ok=True)

# Init DB
init_db()

# ================= AUDIO CONVERSION =================
def convert_to_wav(input_path):
    output_path = input_path.rsplit(".", 1)[0] + ".wav"

    ffmpeg_path = r"C:\ffmpeg\bin\ffmpeg.exe"

    command = [
        ffmpeg_path,
        "-y",
        "-i", input_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        output_path
    ]

    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_path


# ================= ROUTES =================

@app.route("/")
def home():
    return render_template("login.html")


# -------- LOGIN --------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        role = login_user(username, password)

        if role:
            session["user"] = username
            return redirect(url_for("dashboard"))
        else:
            return render_template("login.html", error="❌ Invalid username or password")

    return render_template("login.html")


# -------- REGISTER --------
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        success = register_user(username, password)

        if success:
            return render_template("register.html", message="✅ Account created successfully!")
        else:
            return render_template("register.html", error="❌ Username already exists!")

    return render_template("register.html")


# -------- DASHBOARD --------
@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("dashboard.html", user=session["user"])


@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))


# ================= DETECTION =================

@app.route("/detect", methods=["POST"])
def detect():
    if "user" not in session:
        return redirect(url_for("login"))

    result = None

    # IMAGE
    if "image" in request.files and request.files["image"].filename:
        file = request.files["image"]
        filename = secure_filename(file.filename)
        path = os.path.join(app.config["IMAGE_UPLOAD_FOLDER"], filename)
        file.save(path)

        output = detect_ai_image(path)

        save_log(session["user"], "Image", output["confidence"], output["verdict"])

        result = {
            "type": "Image",
            "confidence": output["confidence"],
            "verdict": output["verdict"],
            "file_path": url_for("uploaded_image", filename=filename)
        }

    # VIDEO
    elif "video" in request.files and request.files["video"].filename:
        file = request.files["video"]
        filename = secure_filename(file.filename)
        path = os.path.join(app.config["VIDEO_UPLOAD_FOLDER"], filename)
        file.save(path)

        confidence, verdict, _ = detect_video(path)

        save_log(session["user"], "Video", confidence, verdict)

        result = {
            "type": "Video",
            "confidence": confidence,
            "verdict": verdict,
            "file_path": url_for("uploaded_video", filename=filename)
        }

    # AUDIO
    elif "audio" in request.files and request.files["audio"].filename:
        file = request.files["audio"]
        filename = secure_filename(file.filename)
        path = os.path.join(app.config["AUDIO_UPLOAD_FOLDER"], filename)
        file.save(path)

        wav_path = convert_to_wav(path)
        output = analyze_audio(wav_path)

        confidence = output["confidence"]
        verdict = output["verdict"]

        save_log(session["user"], "Audio", confidence, verdict)

        result = {
            "type": "Audio",
            "confidence": confidence,
            "verdict": verdict,
            "file_path": url_for("uploaded_audio", filename=filename)
        }

    return render_template("results.html", result=result)


# ================= FILE SERVING =================

@app.route("/uploads/images/<filename>")
def uploaded_image(filename):
    return send_from_directory(app.config["IMAGE_UPLOAD_FOLDER"], filename)


@app.route("/uploads/videos/<filename>")
def uploaded_video(filename):
    return send_from_directory(app.config["VIDEO_UPLOAD_FOLDER"], filename)


@app.route("/uploads/audios/<filename>")
def uploaded_audio(filename):
    return send_from_directory(app.config["AUDIO_UPLOAD_FOLDER"], filename)


# ================= RUN =================

if __name__ == "__main__":
    port = 5000

    ngrok.kill()
    public_url = ngrok.connect(port)
    print(f"🚀 Public URL: {public_url}")

    app.run(port=port, debug=True, use_reloader=False)