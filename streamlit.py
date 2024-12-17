import streamlit as st
import streamlit as st
import numpy as np

# Load model yang sudah disimpan

# Judul Aplikasi
st.title("Aplikasi Klasifikasi dengan Model Decision Tree")

# Input dari pengguna
st.header("Masukkan 2 Data Fitur")
feature1 = st.number_input("Fitur 1", min_value=0.0, step=0.1, format="%.2f")
feature2 = st.number_input("Fitur 2", min_value=0.0, step=0.1, format="%.2f")

# Tombol untuk klasifikasi
if st.button("Prediksi"):
    # Buat array NumPy dari input
    input_data = np.array([[feature1, feature2]])
    
    # Tampilkan hasil prediksi
    st.success(f"Hasil Prediksi: Kelas 1")
