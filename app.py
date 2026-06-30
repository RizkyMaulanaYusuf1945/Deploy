import streamlit as st
import pandas as pd
import pickle
import os

# Set konfigurasi halaman web
st.set_page_config(
    page_title="Analisis Sentimen Vacuum Cleaner",
    page_icon="🧹",
    layout="wide"
)

# Fungsi untuk memuat model secara aman
@st.cache_resource
def load_models():
    tfidf_path = 'tfidf_model.pkl'
    knn_path = 'knn_model.pkl'
    
    if os.path.exists(tfidf_path) and os.path.exists(knn_path):
        with open(tfidf_path, 'rb') as f_tfidf:
            tfidf = pickle.load(f_tfidf)
        with open(knn_path, 'rb') as f_knn:
            knn = pickle.load(f_knn)
        return tfidf, knn
    else:
        return None, None

tfidf_model, knn_model = load_models()

# --- HEADER APLIKASI WEB ---
st.title("🧹 Sistem Analisis Sentimen Ulasan Vacuum Cleaner Portable (Shopee)")
st.markdown("### Menggunakan Algoritma K-Nearest Neighbor (KNN) berparameter $K=7$")
st.write("---")

if tfidf_model is None or knn_model is None:
    st.error("⚠️ File 'tfidf_model.pkl' atau 'knn_model.pkl' tidak ditemukan di folder proyek lu, bro! Tolong download dari Drive dan taruh di folder yang sama dengan file ini.")
else:
    # Membuat dua tab menu interaktif di halaman web
    tab1, tab2 = st.tabs(["🔍 Prediksi Teks Tunggal", "📊 Prediksi Massal (Batch File Upload)"])
    
    # ==========================================
    # TAB 1: PREDIKSI TEKS TUNGGAL (REAL-TIME)
    # ==========================================
    with tab1:
        st.subheader("Input Teks Ulasan Baru")
        ulasan_user = st.text_area(
            "Masukkan teks ulasan pembeli Shopee di sini:", 
            placeholder="Contoh: Barangnya bagus banget, daya hisap kuat dan baterai awet pisaan..."
        )
        
        if st.button("Analisis Sentimen", key="btn_tunggal"):
            if ulasan_user.strip() == "":
                st.warning("Teks ulasan tidak boleh kosong, bro!")
            else:
                vektor_teks = tfidf_model.transform([ulasan_user])
                prediksi = knn_model.predict(vektor_teks)[0]
                
                st.write("#### Hasil Analisis Sistem:")
                if prediksi == 1:
                    st.success("✨ **SENTIMEN POSITIF** — Ulasan mengindikasikan kepuasan terhadap kualitas produk.")
                else:
                    st.error("🚨 **SENTIMEN NEGATIF** — Ulasan mengindikasikan keluhan atau ketidakpuasan pelanggan.")

    # ==========================================
    # TAB 2: PREDIKSI MASSAL (BATCH UPLOAD EXCEL/CSV)
    # ==========================================
    with tab2:
        st.subheader("Analisis Skala Besar Berbasis File Dokumen")
        st.markdown("Unggah file `.xlsx` atau `.csv` lu yang berisi kolom ulasan pembeli untuk diprediksi massal.")
        
        uploaded_file = st.file_uploader("Pilih file dataset ulasan:", type=['csv', 'xlsx'])
        
        if uploaded_file is not None:
            if uploaded_file.name.endswith('.csv'):
                df_batch = pd.read_csv(uploaded_file)
            else:
                df_batch = pd.read_excel(uploaded_file)
                
            st.write("📂 **Preview Data Yang Berhasil Diunggah:**")
            st.dataframe(df_batch.head(5))
            
            nama_kolom = st.selectbox("Pilih kolom nama yang berisi teks ulasan produk:", df_batch.columns)
            
            if st.button("Proses Klasifikasi Massal", key="btn_batch"):
                df_clean = df_batch.dropna(subset=[nama_kolom]).copy()
                fitur_batch = tfidf_model.transform(df_clean[nama_kolom].astype(str))
                hasil_prediksi_batch = knn_model.predict(fitur_batch)
                
                df_clean['Hasil_Prediksi_Sentimen'] = hasil_prediksi_batch
                df_clean['Status_Sentimen'] = df_clean['Hasil_Prediksi_Sentimen'].map({1: 'Positif', 0: 'Negatif'})
                
                st.write("---")
                st.write("### 🎉 Hasil Klasifikasi Massal Berhasil Diselesaikan!")
                
                total_positif = int((df_clean['Hasil_Prediksi_Sentimen'] == 1).sum())
                total_negatif = int((df_clean['Hasil_Prediksi_Sentimen'] == 0).sum())
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Data Diproses", f"{len(df_clean)} ulasan")
                col2.metric("Total Sentimen Positif", f"🟢 {total_positif}")
                col3.metric("Total Sentimen Negatif", f"🔴 {total_negatif}")
                
                st.write("#### Grafik Distribusi Sentimen Produk:")
                df_counts = df_clean['Status_Sentimen'].value_counts()
                st.bar_chart(df_counts)
                
                st.write("#### Tabel Hasil Prediksi Lengkap:")
                st.dataframe(df_clean[[nama_kolom, 'Status_Sentimen']])