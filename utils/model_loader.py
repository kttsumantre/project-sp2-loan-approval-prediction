"""
Modul pemuat model dan fungsi prediksi.

Semua file model dibaca dari folder `models/` hasil training di Google Colab:
- loan_approval_pipeline.pkl  : Pipeline (preprocessing + Decision Tree)
- loan_model_metrics.json     : metrik evaluasi & informasi dataset
"""
import json
from pathlib import Path

import joblib
import pandas as pd

# Lokasi file dihitung dari posisi file ini, jadi aman dijalankan dari folder mana pun
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "loan_approval_pipeline.pkl"
METRICS_PATH = BASE_DIR / "models" / "loan_model_metrics.json"
DATASET_PATH = BASE_DIR / "dataset" / "loan_approval_dataset.csv"

# Nama dan urutan fitur harus sama dengan saat training di notebook
FEATURE_COLUMNS = [
    "no_of_dependents",
    "education",
    "self_employed",
    "income_annum",
    "loan_amount",
    "loan_term",
    "cibil_score",
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value",
]


def load_pipeline():
    """Memuat pipeline hasil training dari Colab."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"File model tidak ditemukan: {MODEL_PATH}. "
            "Jalankan notebook Colab lalu simpan loan_approval_pipeline.pkl ke folder models/."
        )
    return joblib.load(MODEL_PATH)


def load_metrics():
    """Memuat metrik evaluasi (JSON) hasil training dari Colab."""
    if not METRICS_PATH.exists():
        raise FileNotFoundError(
            f"File metrik tidak ditemukan: {METRICS_PATH}. "
            "Jalankan notebook Colab lalu simpan loan_model_metrics.json ke folder models/."
        )
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_dataset_preview():
    """Memuat dataset asli (dirapikan spasinya) untuk pratinjau. Mengembalikan None jika tidak ada."""
    if not DATASET_PATH.exists():
        return None
    df = pd.read_csv(DATASET_PATH, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    for kolom in ["education", "self_employed", "loan_status"]:
        df[kolom] = df[kolom].str.strip()
    return df


def predict_loan(pipeline, data: dict) -> dict:
    """
    Memprediksi satu pemohon.

    data : dict berisi 11 fitur (kunci sesuai FEATURE_COLUMNS)
    return: {"status": "Approved"/"Rejected", "probabilities": {"Approved": p1, "Rejected": p2}}
    """
    input_df = pd.DataFrame([data])[FEATURE_COLUMNS]      # urutan kolom sama dengan training
    prediksi = pipeline.predict(input_df)[0]
    proba = pipeline.predict_proba(input_df)[0]
    probabilities = {str(kelas): float(p) for kelas, p in zip(pipeline.classes_, proba)}
    return {"status": str(prediksi), "probabilities": probabilities}
