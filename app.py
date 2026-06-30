import streamlit as st
import pandas as pd
import pickle
import os
import plotly.express as px  # Ditambahkan untuk visualisasi yang jauh lebih interaktif

# --- 1. SET KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Analisis Sentimen Vacuum Cleaner",
    page_icon="🧹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. CUSTOM CSS UNTUK UI MODERN (10X BETTER LOOK) ---
st.markdown("""
    <style>
    /* Mengubah font global dan background soft */
    .main {
        background-color: #f8f9fa;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Styling Card untuk metrics & hasil */
    .custom-card {
        background-color: #ffffff;
        padding: 24px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        margin-bottom: 20px;
        border-left: 5px solid #4F46E5;
    }
    
    /* Styling Header */
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: #1E1B4B;
        margin-bottom: 5px;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 25px;
    }
    
    /* Modifikasi tab active indicator */
    .stTabs [data-baseweb="tab"] {
        font-size: 16px;
        font-weight: 600;
        color: #4B5563;
        padding: 12px 24px;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #4F46E5;
    }
    .stTabs [aria-selected="true"] {
        color: #4F46E5 !important;
        border-bottom-color: #4F46E5 !important;
    }
    
    /* Tombol Utama */
    div.stButton > button:first-child {
        background-color: #4F46E5;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
        width: 100%;
    }
    div.stButton > button:first-child:hover {
        background-color: #4338CA;
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3);
        transform: translateY(-1px);
    }
    </style>
""", unsafe_allow_html=True)

# --- 3. FUNGSI MEMUAT MODEL (CACHE) ---
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
    return None, None

tfidf_model, knn_model = load_models()

# --- 4. SIDEBAR PANEL (EFEKTIF UNTUK STATS & INFO MODEL) ---
with st.sidebar:
    st.markdown("### ⚙️ Informasi Model")
    st.info("Aplikasi ini menggunakan pipeline Machine Learning untuk mengklasifikasikan ulasan secara otomatis.")
    
    st.markdown("---")
    st.markdown("**Spesifikasi Sistem:**")
    st.markdown("- **Ekstraksi Fitur:** `TF-IDF Vectorizer`")
    st.markdown("- **Algoritma:** `K-Nearest Neighbor`")
    st.markdown("- **Hyperparameter:** `$K = 7$`")
    st.markdown("- **Sumber Data:** Shopee Scraper")
    st.markdown("---")
    st.caption("Sentimen Sistem v2.0 • Dioptimalkan untuk Vacuum Cleaner")

# --- 5. HEADER UTAMA ---
st.markdown('<p class="main-title">🧹 Sentiment Analytics Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Sistem Klasifikasi Ulasan Komparatif Vacuum Cleaner Portable Menggunakan Algoritma KNN</p>', unsafe_allow_html=True)

# --- 6. VALIDASI & LOGIKA APLIKASI ---
if tfidf_model is None or knn_model is None:
    st.error("🚨 **Kritis:** File `tfidf_model.pkl` atau `knn_model.pkl` tidak terdeteksi di server/direktori aktif.")
    st.warning("👉 **Solusi:** Pastikan file model tersebut sudah Anda commit dan push ke GitHub di folder yang sama dengan skrip `app.py` ini.")
else:
    # Memisahkan Menu Menggunakan Tab Modern
    tab1, tab2 = st.tabs(["🔍 Analisis Teks Tunggal", "📊 Analisis Massal (Batch Processing)"])
    
    # ==========================================
    # TAB 1: ANALISIS TEKS TUNGGAL
    # ==========================================
    with tab1:
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Grid layout untuk memisahkan input dan output
        col_in, col_out = st.columns([1.2, 0.8], gap="large")
        
        with col_in:
            st.markdown("#### 💬 Input Ulasan Konsumen")
            ulasan_user = st.text_area(
                "Tulis atau tempel ulasan produk di bawah ini:", 
                placeholder="Contoh: Vacuum-nya ringkih banget, baru dipakai 5 menit baterainya langsung drop...",
                height=150,
                label_visibility="collapsed"
            )
            
            btn_analisis = st.button("Mulai Analisis Sentimen", key="btn_tunggal")
            
        with col_out:
            st.markdown("#### 🎯 Hasil Keputusan Sistem")
            
            if btn_analisis:
                if ulasan_user.strip() == "":
                    st.toast("Isi teks ulasannya dulu ya, bro!", icon="⚠️")
                else:
                    with st.spinner("Mengkalkulasi matriks jarak Jaccard/Euclidean..."):
                        # Transformasi & Prediksi
                        vektor_teks = tfidf_model.transform([ulasan_user])
                        prediksi = knn_model.predict(vektor_teks)[0]
                        
                        # Tampilan Berdasarkan Hasil Prediksi
                        if prediksi == 1:
                            st.markdown("""
                            <div class="custom-card" style="border-left: 5px solid #10B981; background-color: #F0FDF4;">
                                <h3 style='color: #065F46; margin:0;'>🟢 SENTIMEN POSITIF</h3>
                                <p style='color: #047857; font-size:14px; margin-top:8px;'>
                                Konsumen puas dengan performa produk. Indikator mencakup aspek kualitas, efisiensi fungsional, atau pengiriman cepat.
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                            st.balloons()
                        else:
                            st.markdown("""
                            <div class="custom-card" style="border-left: 5px solid #EF4444; background-color: #FEF2F2;">
                                <h3 style='color: #991B1B; margin:0;'>🔴 SENTIMEN NEGATIF</h3>
                                <p style='color: #B91C1C; font-size:14px; margin-top:8px;'>
                                Terdeteksi keluhan komplain pembeli. Segera evaluasi kecacatan produk atau layanan logistik vendor.
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
            else:
                st.info("Silakan ketik teks di sebelah kiri lalu klik tombol analisis untuk melihat hasil perkiraan klasifikasi.")

    # ==========================================
    # TAB 2: ANALISIS MASSAL (BATCH PROCESSING)
    # ==========================================
    with tab2:
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("#### 📂 Pemrosesan Dokumen Skala Besar")
        st.caption("Format file yang didukung: Excel (.xlsx) atau CSV (.csv) yang memiliki kolom ulasan mentah.")
        
        uploaded_file = st.file_uploader("Unggah dataset ulasan Anda:", type=['csv', 'xlsx'], label_visibility="collapsed")
        
        if uploaded_file is not None:
            # Baca file secara aman
            if uploaded_file.name.endswith('.csv'):
                df_batch = pd.read_csv(uploaded_file)
            else:
                df_batch = pd.read_excel(uploaded_file)
                
            st.markdown("---")
            
            # Layout pembagian konfigurasi kolom dan preview
            c_conf, c_prev = st.columns([0.8, 1.2], gap="medium")
            
            with c_conf:
                st.markdown("##### ⚙️ Target Mapping")
                nama_kolom = st.selectbox("Pilih kolom target yang berisi teks ulasan:", df_batch.columns)
                st.markdown("<br>", unsafe_allow_html=True)
                btn_batch = st.button("Eksekusi Klasifikasi Massal", key="btn_batch")
                
            with c_prev:
                st.markdown("##### 📄 Cuplikan Data (Top 3 Baris)")
                st.dataframe(df_batch.head(3), use_container_width=True)
                
            if btn_batch:
                with st.spinner("Sedang memproses seluruh baris data via KNN..."):
                    # Bersihkan NaN khusus pada kolom target ulasan
                    df_clean = df_batch.dropna(subset=[nama_kolom]).copy()
                    
                    # Prediksi Massal
                    fitur_batch = tfidf_model.transform(df_clean[nama_kolom].astype(str))
                    hasil_prediksi_batch = knn_model.predict(fitur_batch)
                    
                    # Mapping Hasil
                    df_clean['Hasil_Prediksi_Sentimen'] = hasil_prediksi_batch
                    df_clean['Status_Sentimen'] = df_clean['Hasil_Prediksi_Sentimen'].map({1: 'Positif', 0: 'Negatif'})
                    
                    st.markdown("---")
                    st.markdown("### 📊 Laporan Hasil Analisis Batch")
                    
                    # Metrics Modern Cards
                    total_data = len(df_clean)
                    total_positif = int((df_clean['Hasil_Prediksi_Sentimen'] == 1).sum())
                    total_negatif = int((df_clean['Hasil_Prediksi_Sentimen'] == 0).sum())
                    
                    m1, m2, m3 = st.columns(3)
                    with m1:
                        st.markdown(f'<div class="custom-card"><h5>📦 Total Data</h5><h2>{total_data} <span style="font-size:14px;color:#6B7280;">Ulasan</span></h2></div>', unsafe_allow_html=True)
                    with m2:
                        st.markdown(f'<div class="custom-card" style="border-left:5px solid #10B981"><h5>🟢 Sentimen Positf</h5><h2>{total_positif} <span style="font-size:14px;color:#10B981;">({(total_positif/total_data)*100:.1f}%)</span></h2></div>', unsafe_allow_html=True)
                    with m3:
                        st.markdown(f'<div class="custom-card" style="border-left:5px solid #EF4444"><h5>🔴 Sentimen Negatif</h5><h2>{total_negatif} <span style="font-size:14px;color:#EF4444;">({(total_negatif/total_data)*100:.1f}%)</span></h2></div>', unsafe_allow_html=True)
                    
                    # Grafik Distribusi & Detail Data Grid
                    g_chart, g_table = st.columns([1, 1], gap="large")
                    
                    with g_chart:
                        st.markdown("##### 📈 Distribusi Rasio")
                        df_counts = df_clean['Status_Sentimen'].value_counts().reset_index()
                        df_counts.columns = ['Sentimen', 'Jumlah']
                        
                        # Menggunakan Plotly Pie Chart agar jauh lebih interaktif dan estetik dibanding bar_chart standar
                        fig = px.pie(
                            df_counts, 
                            values='Jumlah', 
                            names='Sentimen', 
                            color='Sentimen',
                            color_discrete_map={'Positif': '#10B981', 'Negatif': '#EF4444'},
                            hole=0.4
                        )
                        fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=260)
                        st.plotly_chart(fig, use_container_width=True)
                        
                    with g_table:
                        st.markdown("##### 📋 Tabel Output Prediksi")
                        st.dataframe(
                            df_clean[[nama_kolom, 'Status_Sentimen']], 
                            use_container_width=True, 
                            height=260
                        )
                        
                    # Sediakan tombol download hasil download CSV
                    csv_data = df_clean.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Unduh Seluruh Hasil Analisis (.csv)",
                        data=csv_data,
                        file_name="hasil_analisis_sentimen_vacuum.csv",
                        mime="text/csv"
                    )
