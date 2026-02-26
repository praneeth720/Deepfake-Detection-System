import os
from flask import (
    Flask, render_template, request,
    redirect, url_for, session, send_from_directory
)
from werkzeug.utils import secure_filename
from pyngrok import ngrok

# ---------- INTERNAL MODULES ----------
from database import register_user, login_user, init_db, save_log
from image_detector import detect_ai_image
from video_detector import detect_video

# ---------- FLASK CONFIG ----------
app = Flask(__name__)
app.secret_key = "supersecretkey"

# ---------- BASE DIRECTORY ----------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# ---------- UPLOAD FOLDERS ----------
IMAGE_UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads", "images")
VIDEO_UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads", "videos")

os.makedirs(IMAGE_UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VIDEO_UPLOAD_FOLDER, exist_ok=True)

app.config["IMAGE_UPLOAD_FOLDER"] = IMAGE_UPLOAD_FOLDER
app.config["VIDEO_UPLOAD_FOLDER"] = VIDEO_UPLOAD_FOLDER

# ---------- INIT DATABASE ----------
init_db()

# =========================================================
# ======================= ROUTES ==========================
# =========================================================

# ---------- LOGIN ----------
@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        role = login_user(username, password)
        if role:
            session["user"] = username
            return redirect(url_for("dashboard"))
        else:
            return render_template("login.html", error="Invalid credentials")

    return render_template("login.html")


# ---------- REGISTER ----------
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        success = register_user(username, password)
        if success:
            return redirect(url_for("login"))
        else:
            return render_template(
                "register.html",
                message="Username already exists"
            )

    return render_template("register.html")


# ---------- DASHBOARD ----------
@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    if "user" not in session:
        return redirect(url_for("login"))

    result = None

    if request.method == "POST":

        # ================= IMAGE UPLOAD =================
        if "image" in request.files and request.files["image"].filename:
            file = request.files["image"]
            filename = secure_filename(file.filename)
            image_path = os.path.join(
                app.config["IMAGE_UPLOAD_FOLDER"], filename
            )
            file.save(image_path)

            # ---- IMAGE DETECTION ----
            output = detect_ai_image(image_path)

            confidence = output["confidence"]
            verdict = output["verdict"]
            details = output["details"]

            save_log(session["user"], "Image", confidence, verdict)

            result = {
                "type": "Image",
                "confidence": confidence,
                "verdict": verdict,
                "details": details,
                "file_path": url_for(
                    "uploaded_image", filename=filename
                )
            }

        # ================= VIDEO UPLOAD =================
        elif "video" in request.files and request.files["video"].filename:
            file = request.files["video"]
            filename = secure_filename(file.filename)
            video_path = os.path.join(
                app.config["VIDEO_UPLOAD_FOLDER"], filename
            )
            file.save(video_path)

            # ---- VIDEO DETECTION ----
            confidence, verdict, details = detect_video(video_path)

            save_log(session["user"], "Video", confidence, verdict)

            result = {
                "type": "Video",
                "confidence": round(float(confidence), 2),
                "verdict": verdict,
                "details": details,
                "file_path": url_for(
                    "uploaded_video", filename=filename
                )
            }

    return render_template("dashboard.html", result=result)


# ---------- SERVE UPLOADED IMAGE ----------
@app.route("/uploads/images/<filename>")
def uploaded_image(filename):
    return send_from_directory(
        app.config["IMAGE_UPLOAD_FOLDER"], filename
    )


# ---------- SERVE UPLOADED VIDEO ----------
@app.route("/uploads/videos/<filename>")
def uploaded_video(filename):
    return send_from_directory(
        app.config["VIDEO_UPLOAD_FOLDER"], filename
    )


# ---------- LOGOUT ----------
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# =========================================================
# ======================= RUN APP =========================
# =========================================================
if __name__ == "__main__":
    from pyngrok import ngrok

    ngrok.kill()   # 🔥 kill any old tunnels safely
    public_url = ngrok.connect(5000)
    print("🌍 Public URL:", public_url)

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        use_reloader=False
    )