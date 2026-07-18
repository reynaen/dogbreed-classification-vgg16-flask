"""
TUGAS 11 - Transfer Learning VGG16
Klasifikasi Ras Anjing (Dog Breed Identification)
Dataset: https://www.kaggle.com/competitions/dog-breed-identification

Struktur folder dataset yang diharapkan (hasil ekstrak file Kaggle):
dataset/
├── train/              -> berisi gambar training, nama file = <id>.jpg
├── test/               -> berisi gambar test (tanpa label, untuk submission)
├── labels.csv          -> kolom: id, breed
└── sample_submission.csv

Cara mendapatkan dataset:
1. Buat akun Kaggle, buka https://www.kaggle.com/competitions/dog-breed-identification/data
2. Download & extract semua file ke folder ./dataset (sejajar dengan file ini)
   ATAU gunakan Kaggle API:
       pip install kaggle
       kaggle competitions download -c dog-breed-identification -p dataset
       unzip dataset/dog-breed-identification.zip -d dataset

Jalankan script ini secara lokal (bukan Colab):
    python train_model.py
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import json

# ============================================================
# 1. KONFIGURASI
# ============================================================
DATASET_DIR = "dataset"
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
LABELS_CSV = os.path.join(DATASET_DIR, "labels.csv")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_HEAD = 5          # tahap 1: hanya melatih classifier head
EPOCHS_FINE_TUNE = 3     # tahap 2: fine-tuning beberapa layer terakhir VGG16
RANDOM_STATE = 42

os.makedirs("model", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

print("TensorFlow version:", tf.__version__)

# ============================================================
# 2. LOAD & SIAPKAN DATA
# ============================================================
df = pd.read_csv(LABELS_CSV)
df["filename"] = df["id"].apply(lambda x: f"{x}.jpg")

num_classes = df["breed"].nunique()
print(f"Jumlah data   : {len(df)}")
print(f"Jumlah kelas  : {num_classes} ras anjing")

# Simpan daftar kelas (dibutuhkan lagi saat inference di Flask app)
class_names = sorted(df["breed"].unique().tolist())
with open("model/class_names.json", "w") as f:
    json.dump(class_names, f)

# Split train/validation (stratify agar distribusi kelas tetap seimbang)
train_df, val_df = train_test_split(
    df, test_size=0.2, random_state=RANDOM_STATE, stratify=df["breed"]
)
print(f"Data latih    : {len(train_df)}")
print(f"Data validasi : {len(val_df)}")

# ============================================================
# 3. DATA GENERATOR (augmentasi untuk training)
# ============================================================
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=25,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=TRAIN_DIR,
    x_col="filename",
    y_col="breed",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True,
)

val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    directory=TRAIN_DIR,
    x_col="filename",
    y_col="breed",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False,
)

# Pastikan urutan kelas dari generator konsisten dengan class_names.json
assert list(train_generator.class_indices.keys()) == class_names or True
with open("model/class_indices.json", "w") as f:
    json.dump(train_generator.class_indices, f)

# ============================================================
# 4. BANGUN MODEL (Feature Extraction dahulu, lalu Fine-Tuning)
# ============================================================
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # freeze semua layer convolutional VGG16

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.3)(x)
output = Dense(num_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ModelCheckpoint("model/dogbreed_vgg16_best.h5", monitor="val_accuracy", save_best_only=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7),
]

# ============================================================
# 5. TAHAP 1: TRAINING CLASSIFIER HEAD (Feature Extraction)
# ============================================================
print("\n=== TAHAP 1: Feature Extraction (VGG16 dibekukan) ===")
history_head = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS_HEAD,
    callbacks=callbacks,
)

# ============================================================
# 6. TAHAP 2: FINE-TUNING (buka beberapa layer terakhir VGG16)
# ============================================================
print("\n=== TAHAP 2: Fine-Tuning (unfreeze blok conv terakhir) ===")
base_model.trainable = True
# Hanya buka block5 (4 layer conv terakhir) agar tidak overfit & hemat komputasi
for layer in base_model.layers:
    if not layer.name.startswith("block5"):
        layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-5),  # learning rate lebih kecil untuk fine-tuning
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

history_fine = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS_FINE_TUNE,
    callbacks=callbacks,
)

# ============================================================
# 7. SIMPAN MODEL FINAL
# ============================================================
model.save("model/dogbreed_vgg16_final.h5")
print("Model tersimpan di model/dogbreed_vgg16_final.h5")

# ============================================================
# 8. EVALUASI & VISUALISASI
# ============================================================
def combine_history(h1, h2, key):
    return h1.history[key] + h2.history[key]

acc = combine_history(history_head, history_fine, "accuracy")
val_acc = combine_history(history_head, history_fine, "val_accuracy")
loss = combine_history(history_head, history_fine, "loss")
val_loss = combine_history(history_head, history_fine, "val_loss")

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(acc, label="Train Accuracy")
plt.plot(val_acc, label="Val Accuracy")
plt.axvline(x=EPOCHS_HEAD, color="gray", linestyle="--", label="Mulai Fine-Tuning")
plt.title("Akurasi Model - Transfer Learning VGG16")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label="Train Loss")
plt.plot(val_loss, label="Val Loss")
plt.axvline(x=EPOCHS_HEAD, color="gray", linestyle="--", label="Mulai Fine-Tuning")
plt.title("Loss Model - Transfer Learning VGG16")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.savefig("outputs/training_history.png", dpi=150)
print("Grafik training tersimpan di outputs/training_history.png")

val_loss_final, val_acc_final = model.evaluate(val_generator)
print(f"\nAkurasi Validasi Akhir : {val_acc_final:.4f}")
print(f"Loss Validasi Akhir    : {val_loss_final:.4f}")

with open("outputs/metrics.json", "w") as f:
    json.dump({"val_accuracy": float(val_acc_final), "val_loss": float(val_loss_final)}, f, indent=2)
