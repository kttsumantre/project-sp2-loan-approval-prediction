# 🏦 Loan Approval Prediction System

**Sistem Prediksi Persetujuan Pinjaman Menggunakan Algoritma Decision Tree Classifier Berbasis Streamlit**

## 1. Deskripsi Project

Project ini adalah sistem Machine Learning sederhana yang memprediksi apakah pengajuan pinjaman seorang nasabah
akan **Approved (Disetujui)** atau **Rejected (Ditolak)**. Project terdiri dari dua bagian yang saling terhubung:

1. **Google Colab** (`colab/Loan_Approval_Prediction.ipynb`): pengolahan dataset, preprocessing, training, evaluasi, dan export model.
2. **Streamlit** (`app.py`): aplikasi web interaktif yang memakai model hasil training dari Colab.

Model yang dipakai Streamlit adalah file `models/loan_approval_pipeline.pkl` hasil training notebook (bukan model dummy).

> ⚠️ **Disclaimer:** hasil prediksi hanya untuk tujuan edukasi dan bukan keputusan persetujuan kredit yang sebenarnya.

## 2. Tujuan Sistem

- Mempelajari alur lengkap proyek klasifikasi: dari data mentah sampai aplikasi web.
- Memprediksi status pinjaman berdasarkan data pemohon.
- Menampilkan probabilitas prediksi dan hasil evaluasi model secara visual.

## 3. Dataset

File: `dataset/loan_approval_dataset.csv` (dataset asli, tidak diubah).

| Informasi | Nilai |
|---|---|
| Jumlah baris | 4269 |
| Jumlah kolom | 13 (12 + `loan_id`) |
| Missing value | 0 |
| Data duplikat | 0 |
| Kelas Approved | 2656 (62.22%) |
| Kelas Rejected | 1613 (37.78%) |

### Fitur Dataset

| Fitur | Keterangan |
|---|---|
| `loan_id` | Nomor identitas (tidak dipakai sebagai fitur) |
| `no_of_dependents` | Jumlah tanggungan |
| `education` | Pendidikan: Graduate / Not Graduate |
| `self_employed` | Wiraswasta: Yes / No |
| `income_annum` | Pendapatan tahunan |
| `loan_amount` | Jumlah pinjaman |
| `loan_term` | Jangka waktu pinjaman (tahun) |
| `cibil_score` | Skor kredit (300–900) |
| `residential_assets_value` | Nilai aset tempat tinggal |
| `commercial_assets_value` | Nilai aset komersial |
| `luxury_assets_value` | Nilai aset mewah |
| `bank_asset_value` | Nilai aset bank |
| `loan_status` | **Target**: Approved / Rejected |

> Dataset tidak mencantumkan mata uang, sehingga nilai uang ditampilkan apa adanya.

**Catatan kualitas data (ditemukan saat pemeriksaan):**
- Nama kolom dan nilai teks memiliki spasi di depan (mis. `' education'`) sehingga dirapikan dengan `strip()`.
- Terdapat 28 nilai `residential_assets_value` yang negatif. Aset tidak mungkin negatif, sehingga nilainya diubah menjadi 0 (baris **tidak dihapus**).

## 4. Algoritma Decision Tree

Decision Tree membentuk "pohon keputusan" berisi pertanyaan ya/tidak pada fitur, misalnya *apakah CIBIL score di atas suatu ambang?*
Data dipisahkan terus-menerus sampai mencapai **daun** yang berisi keputusan akhir (Approved/Rejected).
Pemisahan dipilih agar kelas pada tiap cabang semakin "murni" (ukuran: Gini impurity).

- **Kelebihan:** mudah dipahami dan divisualisasikan, tidak perlu scaling data.
- **Kekurangan:** mudah overfitting jika pohon terlalu dalam, karena itu `max_depth` dan `min_samples_leaf` dicari dengan GridSearchCV.

## 5. Tahapan Pengolahan Data di Google Colab

1. Import library dan membaca dataset.
2. Pemeriksaan dataset: ukuran, tipe data, statistik deskriptif.
3. EDA dan visualisasi distribusi target.
4. Pemeriksaan missing value, duplikasi, dan anomali.
5. Data cleaning (hapus kolom `loan_id`, perbaiki nilai aset negatif, rapikan spasi).
6. Preprocessing: One-Hot Encoding untuk `education` dan `self_employed` (`ColumnTransformer`); fitur numerik diteruskan apa adanya.
7. Pemisahan X dan y, lalu **stratified train-test split 80:20** (`random_state=42`).
8. Training `Pipeline(preprocessor → DecisionTreeClassifier)` dengan GridSearchCV 5-fold **hanya pada data latih** (menghindari data leakage).
9. Evaluasi pada data uji: Accuracy, Precision, Recall, F1-Score, Confusion Matrix, Classification Report.
10. Visualisasi Decision Tree dan Feature Importance.
11. Export `loan_approval_pipeline.pkl` dan `loan_model_metrics.json`.

Karena preprocessing berada di dalam pipeline yang sama dengan model, Streamlit otomatis memakai preprocessing yang **persis sama** dengan saat training.

## 6. Cara Menjalankan Notebook

**Opsi A — Google Colab**
1. Buka [colab.research.google.com](https://colab.research.google.com) → *File → Upload notebook* → pilih `colab/Loan_Approval_Prediction.ipynb`.
2. Jalankan semua sel (*Runtime → Run all*). Saat diminta, upload `dataset/loan_approval_dataset.csv`.
3. Di akhir notebook, file `loan_approval_pipeline.pkl` dan `loan_model_metrics.json` otomatis terunduh.
4. Letakkan kedua file ke folder `models/` (menimpa file lama).

**Opsi B — Lokal (Jupyter / VS Code)**
Jalankan notebook dari folder `colab/`. Dataset dibaca dari `../dataset/` dan hasilnya langsung disimpan ke `../models/`.

> File di folder `models/` sudah merupakan hasil menjalankan notebook ini, jadi aplikasi bisa langsung dipakai tanpa menjalankan ulang notebook.

## 7. Cara Menjalankan Website Streamlit

```bash
# (opsional) buat virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate        # macOS / Linux

pip install -r requirements.txt
streamlit run app.py
```

Aplikasi terbuka di `http://localhost:8501`.

Halaman yang tersedia: **Dashboard**, **Prediksi Pinjaman**, **Evaluasi Model**, **Informasi Sistem**.

**Jika muncul error saat memuat model** (mis. versi scikit-learn berbeda): model ini dilatih dengan scikit-learn **1.9.1**.
Pasang versi yang sama (`pip install -r requirements.txt`) atau jalankan ulang notebook di lingkungan Anda lalu ganti isi folder `models/`.

## 8. Struktur Folder

```
LOAN_APPROVAL_SYSTEM/
├── app.py                          # Aplikasi web Streamlit
├── requirements.txt                # Daftar library
├── README.md                       # Dokumentasi
├── .gitignore
├── colab/
│   └── Loan_Approval_Prediction.ipynb   # Notebook training & evaluasi
├── dataset/
│   └── loan_approval_dataset.csv        # Dataset asli
├── models/
│   ├── loan_approval_pipeline.pkl       # Pipeline (preprocessing + model)
│   └── loan_model_metrics.json          # Metrik evaluasi untuk Streamlit
├── utils/
│   ├── __init__.py
│   └── model_loader.py                  # Pemuat model, metrik, dan fungsi prediksi
└── assets/                              # Gambar hasil notebook (EDA, confusion matrix, tree, dll.)
```

## 9. Library yang Digunakan

Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Joblib, Streamlit.

## 10. Hasil Klasifikasi

Model terbaik: Decision Tree dengan `max_depth = 10` dan `min_samples_leaf = 5`
(kedalaman pohon 10, 31 daun). Data latih: 3415 baris, data uji: 854 baris.

| Metrik (data uji) | Nilai |
|---|---|
| Accuracy | 97.89% |
| Precision (weighted) | 97.90% |
| Recall (weighted) | 97.89% |
| F1-Score (weighted) | 97.89% |

Akurasi data latih 98.74% dan akurasi cross validation 98.13%, sehingga selisih dengan data uji kecil (tidak ada overfitting berat).

**Confusion Matrix** (baris = aktual, kolom = prediksi):

| | Prediksi Approved | Prediksi Rejected |
|---|---|---|
| **Aktual Approved** | 526 | 5 |
| **Aktual Rejected** | 13 | 310 |

Per kelas: Approved (precision 97.59%, recall 99.06%),
Rejected (precision 98.41%, recall 95.98%).

**Feature Importance:**

| Fitur | Importance |
|---|---|
| cibil_score | 0.8454 |
| loan_term | 0.0858 |
| loan_amount | 0.0295 |
| income_annum | 0.0193 |
| luxury_assets_value | 0.0092 |
| commercial_assets_value | 0.0084 |
| self_employed_Yes | 0.0015 |
| residential_assets_value | 0.0010 |
| education_Not Graduate | 0.0000 |
| no_of_dependents | 0.0000 |
| bank_asset_value | 0.0000 |

**Interpretasi:** `cibil_score` adalah fitur yang paling menentukan keputusan, diikuti `loan_term` dan `loan_amount`.
Pendidikan dan status wiraswasta hampir tidak berpengaruh pada dataset ini.

Catatan: probabilitas pada Decision Tree berasal dari proporsi kelas pada daun, sehingga banyak prediksi bernilai 100% / 0%.
Nilai ini bukan "tingkat keyakinan" statistik seperti pada model lain.
Dataset ini kemungkinan bersifat sintetis (CIBIL score sangat dominan), sehingga akurasi tinggi tidak otomatis berarti akan sama pada data kredit nyata.
