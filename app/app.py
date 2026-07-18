"""
Flask App - Klasifikasi Ras Anjing dengan Transfer Learning VGG16
TUGAS 11 - Dataset: Kaggle Dog Breed Identification

Cara menjalankan:
    pip install -r ../requirements.txt
    python app.py
Lalu buka http://127.0.0.1:5000
"""

import os
import json
import urllib.request
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "dogbreed_vgg16_final.h5")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "..", "model", "class_names.json")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
ALLOWED_EXT = {"png", "jpg", "jpeg"}

# ============================================================
# GANTI URL INI dengan link asli hasil upload GitHub Release kamu
# Contoh: https://github.com/reynaen/dogbreed-classification-vgg16-flask/releases/download/v1.0/dogbreed_vgg16_final.h5
# ============================================================
MODEL_URL = "https://github.com/reynaen/dogbreed-classification-vgg16-flask/releases/tag/v1.0"

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.secret_key = "dogbreed-vgg16-secret-key"  # ganti dengan key acak saat deploy production

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def pastikan_model_tersedia():
    """Download model dari GitHub Release kalau belum ada / rusak (misal cuma pointer LFS)."""
    perlu_download = (
        not os.path.exists(MODEL_PATH)
        or os.path.getsize(MODEL_PATH) < 1_000_000  # kurang dari 1MB = kemungkinan pointer LFS rusak
    )
    if perlu_download:
        print(f"Model tidak ditemukan / tidak valid di {MODEL_PATH}, mengunduh dari GitHub Release...")
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Download model selesai.")


# ============================================================
# Siapkan & load model & daftar kelas sekali saat aplikasi start
# ============================================================
pastikan_model_tersedia()

print("Memuat model VGG16 Dog Breed...")
model = load_model(MODEL_PATH)

with open(CLASS_NAMES_PATH) as f:
    class_names = json.load(f)

print(f"Model berhasil dimuat. Total {len(class_names)} ras anjing.")


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def predict_breed(img_path, top_k=3):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    preds = model.predict(img_array)[0]
    top_indices = preds.argsort()[-top_k:][::-1]

    results = [
        {"breed": class_names[i].replace("_", " ").title(), "confidence": round(float(preds[i]) * 100, 2)}
        for i in top_indices
    ]
    return results


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        flash("Tidak ada file yang diunggah.")
        return redirect(url_for("index"))

    file = request.files["file"]
    if file.filename == "":
        flash("Silakan pilih gambar terlebih dahulu.")
        return redirect(url_for("index"))

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        results = predict_breed(filepath)

        return render_template(
            "result.html",
            image_file=filename,
            results=results,
        )
    else:
        flash("Format file tidak didukung. Gunakan PNG, JPG, atau JPEG.")
        return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)