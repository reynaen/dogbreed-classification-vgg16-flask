# DogBreedAI — Transfer Learning VGG16 (Tugas 11)

Aplikasi klasifikasi ras anjing menggunakan **Transfer Learning VGG16**, dilatih dengan dataset
[Kaggle Dog Breed Identification](https://www.kaggle.com/competitions/dog-breed-identification)
(120 ras anjing), lalu di-deploy sebagai aplikasi web berbasis **Flask + Bootstrap**.

> Project ini dijalankan secara lokal (bukan Google Colab), sesuai kebutuhan Tugas 11.

## Struktur Folder

```
dogbreed-vgg16/
├── dataset/                  # download & extract dataset Kaggle di sini (tidak disertakan)
│   ├── train/
│   ├── test/
│   └── labels.csv
├── model/                    # hasil training (dibuat otomatis oleh train_model.py)
│   ├── dogbreed_vgg16_final.h5
│   ├── dogbreed_vgg16_best.h5
│   ├── class_names.json
│   └── class_indices.json
├── outputs/                  # grafik & metrik training
│   ├── training_history.png
│   └── metrics.json
├── app/                      # aplikasi Flask
│   ├── app.py
│   ├── templates/
│   │   ├── index.html
│   │   └── result.html
│   └── static/uploads/
├── train_model.py            # script training Transfer Learning VGG16
├── requirements.txt
├── Procfile                  # untuk deploy ke Railway/Heroku
└── README.md
```

## Langkah Pengerjaan

### 1. Siapkan Environment
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download Dataset dari Kaggle
Opsi A — manual:
1. Buka https://www.kaggle.com/competitions/dog-breed-identification/data
2. Download `train.zip`, `test.zip`, `labels.csv`
3. Extract semua ke folder `dataset/` sehingga strukturnya:
   ```
   dataset/train/xxxx.jpg
   dataset/test/xxxx.jpg
   dataset/labels.csv
   ```

Opsi B — Kaggle API:
```bash
pip install kaggle
# taruh kaggle.json (API token) di ~/.kaggle/kaggle.json
kaggle competitions download -c dog-breed-identification -p dataset
unzip "dataset/*.zip" -d dataset
```

### 3. Training Model
```bash
python train_model.py
```
Script akan melakukan:
- **Tahap 1 (Feature Extraction):** VGG16 dibekukan, hanya melatih classifier head baru.
- **Tahap 2 (Fine-Tuning):** membuka blok convolutional terakhir (`block5`) VGG16 dan melatih ulang dengan learning rate kecil.
- Menyimpan model (`model/dogbreed_vgg16_final.h5`), daftar kelas, grafik akurasi/loss, dan metrik evaluasi.

Estimasi waktu training tergantung spesifikasi PC/GPU — disarankan menggunakan GPU (CUDA) agar tidak terlalu lama karena dataset berisi ±10.000 gambar dan 120 kelas.

### 4. Jalankan Aplikasi Web
```bash
cd app
python app.py
```
Buka `http://127.0.0.1:5000` di browser. Upload foto anjing → aplikasi menampilkan top-3 prediksi ras beserta persentase keyakinan.

### 5. Deploy ke Hosting Gratis (contoh: Railway)
1. Push seluruh folder project ini ke GitHub.
2. Buat project baru di [Railway](https://railway.app), hubungkan ke repo GitHub.
3. Railway otomatis membaca `Procfile` dan `requirements.txt`.
4. Set start command (jika perlu): `cd app && gunicorn app:app --bind 0.0.0.0:$PORT`.
5. Setelah deploy selesai, catat URL publik untuk dilampirkan di laporan.

> Catatan: file model `.h5` hasil training biasanya berukuran puluhan-ratusan MB. Jika melebihi limit GitHub (100MB), gunakan **Git LFS** — sesuai pengalaman sebelumnya saat deploy model CNN.

## Tips Penulisan Laporan
- Jelaskan mengapa VGG16 dipilih (arsitektur, pretrained ImageNet, feature extraction vs fine-tuning).
- Tampilkan grafik `outputs/training_history.png` (akurasi & loss, termasuk garis pemisah tahap fine-tuning).
- Sertakan tabel/confusion matrix top-N kelas dengan akurasi tertinggi/terendah (dataset 120 kelas terlalu besar untuk confusion matrix penuh — bisa ditampilkan top 10-15 kelas paling sering salah klasifikasi).
- Cantumkan link GitHub, link aplikasi hosting, dan link video YouTube.
