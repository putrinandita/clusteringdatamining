import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, mean_squared_error, r2_score

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Vending Machine Location Profiling", layout="wide")

# =========================
# TITLE & INTRO
# =========================
st.title("📊 Vending Machine Location Profiling")

st.markdown("""
Aplikasi ini melakukan analisis pada data penjualan mesin vending Anda,
dengan memperlakukan setiap lokasi sebagai entitas yang diprofilkan:

1. **Segmentasi Lokasi (Clustering)** → Mengelompokkan lokasi berdasarkan kemiripan kinerja penjualan dan bauran produk.
2. **Prediksi Kinerja Penjualan (Regresi)** → Memprediksi **Total Sales** lokasi menggunakan metrik kinerja lainnya.
""")

# =========================
# FUNGSIONALITAS FEATURE ENGINEERING
# =========================

def perform_feature_engineering(df_raw):
    """Membersihkan dan mengubah data transaksi menjadi metrik kinerja per Lokasi."""
    
    # 1. Data Cleaning
    df = df_raw.copy()
    df.dropna(subset=['Category', 'Product', 'MPrice'], inplace=True)
    
    # Hitung metrik penjualan per lokasi (Analogi Profil Pelanggan)
    location_sales = df.groupby('Location').agg(
        # Target Regresi dan Fitur Clustering
        Total_Sales=('TransTotal', 'sum'),
        Total_Qty=('MQty', 'sum'),
        Num_Transactions=('Transaction', 'count'),
        Avg_Trans_Value=('TransTotal', 'mean'),
    ).reset_index()

    # Hitung fitur 'Product Mix' (Contoh: Proporsi Carbonated)
    carbonated_qty = df[df['Category'] == 'Carbonated'].groupby('Location')['MQty'].sum().reset_index(name='Qty_Carbonated')
    location_sales = pd.merge(location_sales, carbonated_qty, on='Location', how='left').fillna(0)
    location_sales['Prop_Carbonated'] = location_sales['Qty_Carbonated'] / location_sales['Total_Qty']
    location_sales.drop(columns=['Qty_Carbonated'], inplace=True)
    
    return location_sales

# =========================
# DATASET UPLOAD
# =========================
st.header("📁 Upload Dataset")
uploaded_file = st.file_uploader("Upload file CSV 'vending_machine_sales.csv'", type=["csv"])

if uploaded_file is not None:
    df_raw = pd.read_csv(uploaded_file)
    st.success("Dataset berhasil diupload. Memulai Feature Engineering...")

    try:
        df_profile = perform_feature_engineering(df_raw)
        
        if df_profile.shape[0] < 4:
             st.error(f"Hanya ditemukan {df_profile.shape[0]} lokasi unik. Tidak cukup data untuk clustering dan regresi yang bermakna.")
             st.stop()

        st.subheader("🔍 Preview Data Profil Lokasi")
        st.dataframe(df_profile.head())

        # =========================
        # FEATURE SELECTION
        # =========================
        # Fitur yang digunakan untuk Clustering dan Regresi
        CLUSTERING_FEATURES = [
            'Total_Sales',
            'Total_Qty',
            'Num_Transactions',
            'Avg_Trans_Value',
            'Prop_Carbonated'
        ]

        X = df_profile[CLUSTERING_FEATURES]

        # SCALING
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # ==========================================================
        # ANALISIS 1: CLUSTERING (Segmentasi Lokasi)
        # ==========================================================
        st.header("🔹 Analisis 1: Segmentasi Lokasi (K-Means)")

        num_locations = df_profile.shape[0]
        max_k = min(num_locations - 1, 10)
        
        # UI untuk memilih K
        k = st.slider("Pilih Jumlah Cluster (k):", min_value=2, max_value=max_k, value=min(3, max_k))
        
        try:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X_scaled)

            df_profile['Cluster'] = cluster_labels

            # Evaluasi K-Means
            sil_score = silhouette_score(X_scaled, cluster_labels)
            
            st.write(f"**Jumlah Cluster yang Dipilih:** {k}")
            st.write(f"**Silhouette Score:** {sil_score:.4f} (Semakin dekat ke 1, semakin baik pemisahan cluster)")
            
            st.subheader("Distribusi Cluster")
            st.bar_chart(df_profile['Cluster'].value_counts())

            st.subheader("Ringkasan Profil Cluster")
            cluster_summary = df_profile.groupby('Cluster')[CLUSTERING_FEATURES].mean().style.format("{:,.2f}")
            st.dataframe(cluster_summary)
            st.markdown("---")

        except ValueError as e:
            st.error(f"Gagal melakukan clustering: {e}. Coba kurangi jumlah klaster.")
            st.stop()
        
        # ==========================================================
        # ANALISIS 2: REGRESI (Prediksi Total Sales)
        # ==========================================================
        st.header("🔹 Analisis 2: Prediksi Total Sales (Regresi)")

        st.markdown("""
        Model **Random Forest Regressor** digunakan untuk memprediksi
        **Total Sales** lokasi, berdasarkan metrik kinerja lainnya.
        """)

        # Target & fitur regresi
        REGRESSION_TARGET = 'Total_Sales'
        REGRESSION_FEATURES = [f for f in CLUSTERING_FEATURES if f != REGRESSION_TARGET]
        
        y = df_profile[REGRESSION_TARGET]
        X_reg = df_profile[REGRESSION_FEATURES]

        # Scaling Regresi
        scaler_reg = StandardScaler()
        X_reg_scaled = scaler_reg.fit_transform(X_reg)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_reg_scaled, y, test_size=0.2, random_state=42
        )

        # Model regresi
        model = RandomForestRegressor(
            n_estimators=100, # Dikurangi agar lebih cepat
            random_state=42
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Evaluasi
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        col1, col2, col3 = st.columns(3)
        col1.metric("MSE", f"{mse:,.2f}")
        col2.metric("RMSE", f"{rmse:,.2f}")
        col3.metric("R² Score", f"{r2:.4f}")

        # =========================
        # VISUALISASI REGRESI
        # =========================
        st.subheader("📊 Visualisasi Prediksi")

        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, alpha=0.7, color='teal')
        ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2) # Garis ideal (y=x)
        ax.set_xlabel("Actual Total Sales")
        ax.set_ylabel("Predicted Total Sales")
        ax.set_title("Actual vs Predicted Total Sales")
        st.pyplot(fig)
        
        # =========================
        # INTERPRETASI SINGKAT
        # =========================
        st.info(f"""
        🔍 **Interpretasi:**
        - Model Random Forest Regressor telah dilatih untuk memprediksi **Total Sales**.
        - Nilai **R² Score ({r2:.4f})** menunjukkan proporsi variasi penjualan yang dapat dijelaskan oleh model.
        - Titik-titik yang dekat dengan garis diagonal merah menunjukkan prediksi yang akurat.
        """)

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses data: {e}")

else:
    st.info("Silakan upload dataset CSV untuk memulai analisis.")
