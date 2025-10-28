import os
from flask import Flask, request, render_template
from PIL import Image
from werkzeug.utils import secure_filename
from utils import load_model, predict_image

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
MODEL_PATH = "models/resnet50_deepfake.pth"

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB limit

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model once at startup
model, classes, device = load_model(MODEL_PATH)

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "image" not in request.files:
            return render_template("index.html", error="No file part")

        file = request.files["image"]
        if file.filename == "":
            return render_template("index.html", error="No selected file")

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            # Open and predict
            pil_img = Image.open(filepath).convert("RGB")
            label, confidence, probs = predict_image(model, classes, pil_img, device=device)

            # Convert NumPy arrays to Python floats
            probs = [float(p) for p in probs]
            confidence = float(confidence)
            confidence_pct = round(confidence * 100, 2)

            # Render results
            return render_template(
                "result.html",
                filename=filename,
                label=label,
                confidence=confidence_pct,
                probs=probs,
                classes=classes
            )
        else:
            return render_template("index.html", error="File type not allowed")

    return render_template("index.html")


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return app.send_static_file(os.path.join("..", app.config["UPLOAD_FOLDER"], filename))


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
