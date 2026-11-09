"""
LOAN APPROVAL PREDICTION SYSTEM
Aplikasi web Streamlit untuk memprediksi persetujuan pinjaman
menggunakan model Decision Tree yang dilatih di Google Colab.

Jalankan dengan:  streamlit run app.py
"""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from utils.model_loader import (
    load_pipeline, load_metrics, load_dataset_preview, predict_loan,
)

# ---------------------------------------------------------------------------
# Konfigurasi halaman & warna
# ---------------------------------------------------------------------------
NAVY = "#1F3A5F"
GREEN = "#2E9E5B"
RED = "#D64545"
LIGHT_GRAY = "#F3F5F8"

st.set_page_config(page_title="Loan Approval Prediction System", page_icon="🏦", layout="wide")

st.markdown(
    f"""
    <style>
    .block-container {{ padding-top: 2rem; max-width: 1200px; }}
    section[data-testid="stSidebar"] {{ background-color: {NAVY}; }}
    section[data-testid="stSidebar"] * {{ color: #FFFFFF !important; }}
    .hero {{
        background: linear-gradient(120deg, {NAVY} 0%, #2F5A8F 100%);
        color: #FFFFFF; padding: 1.6rem 2rem; border-radius: 14px; margin-bottom: 1.5rem;
    }}
    .hero h1 {{ color: #FFFFFF; margin: 0; font-size: 2rem; }}
    .hero p {{ color: #DCE6F2; margin: 0.4rem 0 0 0; }}
    .card {{
        background: {LIGHT_GRAY}; border-radius: 12px; padding: 1rem 1.2rem;
        border-left: 5px solid {NAVY}; margin-bottom: 0.8rem; color: #1B2733;
    }}
    .card h4 {{ margin: 0 0 0.3rem 0; color: {NAVY}; }}
    .result-approved {{
        background: #E8F6EE; border: 2px solid {GREEN}; color: {GREEN};
        border-radius: 14px; padding: 1.2rem; text-align: center;
        font-size: 1.8rem; font-weight: 700;
    }}
    .result-rejected {{
        background: #FBEAEA; border: 2px solid {RED}; color: {RED};
        border-radius: 14px; padding: 1.2rem; text-align: center;
        font-size: 1.8rem; font-weight: 700;
    }}
    div[data-testid="stMetric"] {{
        background: {LIGHT_GRAY}; border-radius: 12px; padding: 0.8rem 1rem;
    }}
    div[data-testid="stMetric"] * {{ color: #1B2733 !important; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Memuat model & metrik (di-cache agar tidak dimuat ulang setiap interaksi)
# ---------------------------------------------------------------------------
@st.cache_resource
def get_pipeline():
    return load_pipeline()


@st.cache_data
def get_metrics():
    return load_metrics()


@st.cache_data
def get_dataset():
    return load_dataset_preview()


try:
    pipeline = get_pipeline()
    metrics = get_metrics()
except Exception as e:  # file model belum ada / versi library tidak cocok
    st.error(f"Gagal memuat model: {e}")
    st.info(
        "Pastikan file `models/loan_approval_pipeline.pkl` dan `models/loan_model_metrics.json` tersedia, "
        "dan versi scikit-learn sama dengan versi saat training (lihat requirements.txt)."
    )
    st.stop()


# ---------------------------------------------------------------------------
# Fungsi bantu
# ---------------------------------------------------------------------------
def hero(judul, deskripsi):
    st.markdown(f'<div class="hero"><h1>{judul}</h1><p>{deskripsi}</p></div>', unsafe_allow_html=True)


def card(judul, isi):
    st.markdown(f'<div class="card"><h4>{judul}</h4>{isi}</div>', unsafe_allow_html=True)


def fmt(angka):
    """Format angka dengan pemisah ribuan."""
    return f"{int(angka):,}".replace(",", ".")


def pct(x):
    return f"{x * 100:.2f}%"


def plot_distribusi(distribusi, judul):
    """Bar chart distribusi kelas (Approved hijau, Rejected merah)."""
    label = ["Approved", "Rejected"]
    nilai = [distribusi.get(k, 0) for k in label]
    fig, ax = plt.subplots(figsize=(5, 3.6))
    bars = ax.bar(label, nilai, color=[GREEN, RED])
    for b, v in zip(bars, nilai):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}".replace(",", "."), ha="center", va="bottom")
    ax.set_title(judul)
    ax.set_ylabel("Jumlah data")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Sidebar navigasi
# ---------------------------------------------------------------------------
st.sidebar.markdown("## 🏦 LOAN APPROVAL")
st.sidebar.markdown("**Prediction System**")
halaman = st.sidebar.radio(
    "Menu",
    ["Dashboard", "Prediksi Pinjaman", "Evaluasi Model", "Informasi Sistem"],
    label_visibility="collapsed",
)
st.sidebar.markdown("---")
st.sidebar.caption("Algoritma: Decision Tree Classifier")
st.sidebar.caption("⚠️ Untuk tujuan edukasi, bukan keputusan kredit yang sebenarnya.")


# ===========================================================================
# HALAMAN 1: DASHBOARD
# ===========================================================================
if halaman == "Dashboard":
    hero("LOAN APPROVAL PREDICTION SYSTEM",
         "Sistem prediksi persetujuan pinjaman menggunakan algoritma Decision Tree Classifier.")

    ds, sp, md, mt = metrics["dataset"], metrics["split"], metrics["model"], metrics["metrics"]

    st.subheader("Informasi Dataset")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Data", fmt(ds["jumlah_baris"]))
    c2.metric("Data Training", fmt(sp["jumlah_train"]))
    c3.metric("Data Testing", fmt(sp["jumlah_test"]))
    c4.metric("Jumlah Fitur", ds["jumlah_fitur"])

    st.subheader("Metrik Evaluasi Model")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", pct(mt["accuracy"]))
    m2.metric("Precision", pct(mt["precision"]))
    m3.metric("Recall", pct(mt["recall"]))
    m4.metric("F1-Score", pct(mt["f1_score"]))
    st.caption("Metrik dihitung pada data uji dengan rata-rata weighted.")

    kiri, kanan = st.columns([1, 1])
    with kiri:
        st.subheader("Distribusi Status Pinjaman")
        st.pyplot(plot_distribusi(ds["distribusi_kelas"], "Seluruh Dataset"))
    with kanan:
        st.subheader("Informasi Algoritma")
        depth = md["max_depth"] if md["max_depth"] is not None else "tanpa batas"
        card("Decision Tree Classifier",
             f"Model memisahkan data berdasarkan aturan if-else pada fitur (terutama CIBIL score).<br>"
             f"Pengaturan terbaik: <b>max_depth = {depth}</b>, <b>min_samples_leaf = {md['min_samples_leaf']}</b>.<br>"
             f"Kedalaman pohon: <b>{md['kedalaman_pohon']}</b> &nbsp;|&nbsp; Jumlah daun: <b>{md['jumlah_daun']}</b>.")
        card("Pembagian Data",
             f"Train : Test = {int((1 - sp['test_size']) * 100)} : {int(sp['test_size'] * 100)} "
             f"(stratified, random_state = {sp['random_state']}).")

    st.subheader("Cuplikan Dataset")
    df = get_dataset()
    if df is not None:
        st.dataframe(df.head(10))
    else:
        st.info("File dataset tidak ditemukan di folder dataset/.")


# ===========================================================================
# HALAMAN 2: PREDIKSI
# ===========================================================================
elif halaman == "Prediksi Pinjaman":
    hero("Prediksi Persetujuan Pinjaman",
         "Isi data calon peminjam, lalu tekan tombol prediksi untuk melihat hasilnya.")

    rg = metrics["feature_ranges"]
    kat = metrics["kategori"]
    LABEL_EDU = {"Graduate": "Lulusan Perguruan Tinggi (Graduate)", "Not Graduate": "Bukan Lulusan Perguruan Tinggi (Not Graduate)"}
    LABEL_SE = {"Yes": "Ya (Wiraswasta)", "No": "Tidak"}

    st.caption("Nilai uang menggunakan satuan yang sama dengan dataset (dataset tidak mencantumkan mata uang). "
               "Rentang input mengikuti rentang data yang dipakai saat training.")

    with st.form("form_prediksi"):
        a, b = st.columns(2)
        with a:
            st.markdown("**Data Pribadi**")
            tanggungan = st.slider("Jumlah tanggungan", rg["no_of_dependents"]["min"], rg["no_of_dependents"]["max"],
                                   rg["no_of_dependents"]["median"])
            pendidikan = st.selectbox("Pendidikan", kat["education"], format_func=lambda x: LABEL_EDU.get(x, x))
            mandiri = st.selectbox("Status pekerjaan mandiri (self employed)", kat["self_employed"],
                                   format_func=lambda x: LABEL_SE.get(x, x))
            cibil = st.slider("CIBIL score", rg["cibil_score"]["min"], rg["cibil_score"]["max"],
                              rg["cibil_score"]["median"])

            st.markdown("**Data Pinjaman**")
            pendapatan = st.number_input("Pendapatan tahunan", min_value=0, max_value=rg["income_annum"]["max"],
                                         value=rg["income_annum"]["median"], step=100000)
            pinjaman = st.number_input("Jumlah pinjaman", min_value=0, max_value=rg["loan_amount"]["max"],
                                       value=rg["loan_amount"]["median"], step=100000)
            tenor = st.slider("Jangka waktu pinjaman (tahun)", rg["loan_term"]["min"], rg["loan_term"]["max"],
                              rg["loan_term"]["median"])
        with b:
            st.markdown("**Informasi Aset**")
            a_tinggal = st.number_input("Nilai aset tempat tinggal", min_value=0,
                                        max_value=rg["residential_assets_value"]["max"],
                                        value=rg["residential_assets_value"]["median"], step=100000)
            a_komersial = st.number_input("Nilai aset komersial", min_value=0,
                                          max_value=rg["commercial_assets_value"]["max"],
                                          value=rg["commercial_assets_value"]["median"], step=100000)
            a_mewah = st.number_input("Nilai aset mewah", min_value=0, max_value=rg["luxury_assets_value"]["max"],
                                      value=rg["luxury_assets_value"]["median"], step=100000)
            a_bank = st.number_input("Nilai aset bank", min_value=0, max_value=rg["bank_asset_value"]["max"],
                                     value=rg["bank_asset_value"]["median"], step=100000)

        kirim = st.form_submit_button("Prediksi Persetujuan Pinjaman")

    if kirim:
        data_input = {
            "no_of_dependents": int(tanggungan),
            "education": pendidikan,
            "self_employed": mandiri,
            "income_annum": int(pendapatan),
            "loan_amount": int(pinjaman),
            "loan_term": int(tenor),
            "cibil_score": int(cibil),
            "residential_assets_value": int(a_tinggal),
            "commercial_assets_value": int(a_komersial),
            "luxury_assets_value": int(a_mewah),
            "bank_asset_value": int(a_bank),
        }
        hasil = predict_loan(pipeline, data_input)
        status = hasil["status"]
        p_app = hasil["probabilities"].get("Approved", 0.0)
        p_rej = hasil["probabilities"].get("Rejected", 0.0)

        st.markdown("---")
        st.subheader("Hasil Prediksi")
        if status == "Approved":
            st.markdown('<div class="result-approved">✅ APPROVED — Pinjaman Disetujui</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-rejected">❌ REJECTED — Pinjaman Ditolak</div>', unsafe_allow_html=True)
        st.write("")

        k1, k2 = st.columns(2)
        k1.metric("Probabilitas Approved", pct(p_app))
        k2.metric("Probabilitas Rejected", pct(p_rej))

        fig, ax = plt.subplots(figsize=(8, 2.2))
        ax.barh(["Rejected", "Approved"], [p_rej * 100, p_app * 100], color=[RED, GREEN])
        ax.set_xlim(0, 100)
        ax.set_xlabel("Probabilitas (%)")
        for i, v in enumerate([p_rej * 100, p_app * 100]):
            ax.text(min(v + 1, 88), i, f"{v:.1f}%", va="center")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)

        st.subheader("Ringkasan Data Pemohon")
        ringkasan = pd.DataFrame({
            "Data": ["Jumlah tanggungan", "Pendidikan", "Wiraswasta (self employed)", "Pendapatan tahunan",
                     "Jumlah pinjaman", "Jangka waktu pinjaman", "CIBIL score", "Aset tempat tinggal",
                     "Aset komersial", "Aset mewah", "Aset bank"],
            "Nilai": [str(data_input["no_of_dependents"]), pendidikan, mandiri, fmt(pendapatan), fmt(pinjaman),
                      f"{tenor} tahun", str(cibil), fmt(a_tinggal), fmt(a_komersial), fmt(a_mewah), fmt(a_bank)],
        })
        st.table(ringkasan.set_index("Data"))
        st.caption("⚠️ Hasil prediksi hanya untuk tujuan edukasi dan bukan keputusan persetujuan kredit yang sebenarnya.")


# ===========================================================================
# HALAMAN 3: EVALUASI MODEL
# ===========================================================================
elif halaman == "Evaluasi Model":
    hero("Evaluasi Model", "Hasil pengujian model Decision Tree pada data uji (hasil training di Google Colab).")

    mt = metrics["metrics"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", pct(mt["accuracy"]))
    c2.metric("Precision", pct(mt["precision"]))
    c3.metric("Recall", pct(mt["recall"]))
    c4.metric("F1-Score", pct(mt["f1_score"]))
    st.caption(f"Precision, Recall, dan F1-Score memakai rata-rata {mt['average']}. "
               f"Akurasi data latih: {pct(mt['akurasi_train'])}, akurasi cross validation: "
               f"{pct(metrics['model']['akurasi_cross_validation'])}.")

    kiri, kanan = st.columns(2)
    with kiri:
        st.subheader("Confusion Matrix")
        cm = metrics["confusion_matrix"]
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm["matrix"], annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=cm["labels"], yticklabels=cm["labels"], ax=ax)
        ax.set_xlabel("Prediksi")
        ax.set_ylabel("Aktual")
        fig.tight_layout()
        st.pyplot(fig)
    with kanan:
        st.subheader("Classification Report")
        laporan = pd.DataFrame(metrics["classification_report"]).T
        laporan = laporan.rename(columns={"precision": "Precision", "recall": "Recall",
                                          "f1-score": "F1-Score", "support": "Support"})
        laporan["Support"] = laporan["Support"].astype(int)
        st.dataframe(laporan.round(4))

    st.subheader("Feature Importance")
    fi = pd.DataFrame(metrics["feature_importance"]).sort_values("importance", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(data=fi, x="importance", y="feature", color=NAVY, ax=ax)
    ax.set_xlabel("Importance")
    ax.set_ylabel("")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig)
    st.caption(f"Fitur paling berpengaruh: **{fi.iloc[0]['feature']}** "
               f"({fi.iloc[0]['importance'] * 100:.1f}% dari total importance).")


# ===========================================================================
# HALAMAN 4: INFORMASI SISTEM
# ===========================================================================
else:
    hero("Informasi Sistem", "Penjelasan project, algoritma, dan fitur dataset.")

    card("Deskripsi Project",
         "Sistem Prediksi Persetujuan Pinjaman (Loan Approval Prediction) adalah aplikasi Machine Learning sederhana "
         "yang memprediksi apakah pengajuan pinjaman akan <b>Approved</b> (disetujui) atau <b>Rejected</b> (ditolak). "
         "Model dilatih di Google Colab, lalu dipakai oleh aplikasi web Streamlit ini.")
    card("Algoritma: Decision Tree Classifier",
         "Decision Tree membentuk struktur seperti pohon yang berisi serangkaian pertanyaan ya/tidak pada fitur "
         "(contoh: <i>apakah CIBIL score &gt; 549?</i>). Data dipisahkan terus-menerus hingga sampai pada daun yang "
         "berisi keputusan akhir. Kelebihannya mudah dipahami dan tidak memerlukan scaling data. "
         "Kekurangannya cenderung overfitting jika pohon terlalu dalam, sehingga kedalamannya dibatasi.")

    st.subheader("Penjelasan Fitur Dataset")
    fitur = pd.DataFrame({
        "Fitur": ["no_of_dependents", "education", "self_employed", "income_annum", "loan_amount", "loan_term",
                  "cibil_score", "residential_assets_value", "commercial_assets_value", "luxury_assets_value",
                  "bank_asset_value", "loan_status (target)"],
        "Keterangan": ["Jumlah tanggungan pemohon", "Pendidikan: Graduate / Not Graduate",
                       "Apakah pemohon wiraswasta (Yes / No)", "Pendapatan tahunan",
                       "Jumlah pinjaman yang diajukan", "Jangka waktu pinjaman (tahun)",
                       "Skor kredit (300 - 900); makin tinggi makin baik", "Nilai aset tempat tinggal",
                       "Nilai aset komersial", "Nilai aset mewah", "Nilai aset di bank",
                       "Status pinjaman: Approved / Rejected"],
    })
    st.table(fitur.set_index("Fitur"))

    st.subheader("Alur Sistem")
    st.markdown(
        "1. Dataset diproses di Google Colab (cleaning, encoding, split 80:20).\n"
        "2. Decision Tree dilatih dan dievaluasi, lalu disimpan sebagai `loan_approval_pipeline.pkl`.\n"
        "3. Metrik evaluasi disimpan sebagai `loan_model_metrics.json`.\n"
        "4. Streamlit memuat kedua file tersebut untuk prediksi dan menampilkan hasil evaluasi."
    )
    st.info(f"Model dilatih dengan scikit-learn versi {metrics['model']['sklearn_version']}.")
    st.warning("**Disclaimer:** hasil prediksi hanya untuk tujuan edukasi dan bukan keputusan persetujuan "
               "kredit yang sebenarnya.")
